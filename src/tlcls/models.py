"""精确指定权重、规范 Adapter 与基于模块对象的参数白名单。"""

import math

import timm
import torch
from torch import nn
from safetensors.torch import load_file

from .common import ROOT, derive_seed, load_json, rng_stream, tensor_hash


class ViTAdapter(nn.Module):
    def __init__(self, channels=768):
        super().__init__()
        self.down = nn.Linear(channels, channels // 4)
        self.up = nn.Linear(channels // 4, channels)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, value):
        return value + self.up(torch.nn.functional.gelu(self.down(value)))


class R50Adapter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.down = nn.Conv2d(channels, channels // 4, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels // 4, eps=1e-5, momentum=0.1)
        self.up = nn.Conv2d(channels // 4, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.1)
        nn.init.kaiming_normal_(self.down.weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(self.up.weight, mode='fan_out', nonlinearity='relu')
        nn.init.ones_(self.bn1.weight)
        nn.init.zeros_(self.bn1.bias)
        nn.init.zeros_(self.bn2.weight)
        nn.init.zeros_(self.bn2.bias)

    def forward(self, value):
        return value + self.bn2(self.up(torch.relu(self.bn1(self.down(value)))))


class AdaptedBlock(nn.Module):
    def __init__(self, block, adapter):
        super().__init__()
        self.block = block
        self.adapter = adapter

    def forward(self, value):
        # 冻结的后续层仍保留 autograd，确保前端 Adapter 的梯度可达。
        return self.adapter(self.block(value))


def head_of(model, model_key):
    return model.fc if model_key == 'r50' else model.head


def adapters_of(model):
    return [module for module in model.modules() if isinstance(module, (ViTAdapter, R50Adapter))]


def insert_adapters(model, model_key, seed):
    with rng_stream(derive_seed('adapter', seed, model_key)):
        if model_key == 'r50':
            for stage_id, channels in enumerate([256, 512, 1024, 2048], start=1):
                stage = getattr(model, f'layer{stage_id}')
                for index in range(len(stage)):
                    stage[index] = AdaptedBlock(stage[index], R50Adapter(channels))
        else:
            for index in range(len(model.blocks)):
                model.blocks[index] = AdaptedBlock(model.blocks[index], ViTAdapter())


def apply_strategy(model, model_key, method, training=True):
    head = head_of(model, model_key)
    adapters = adapters_of(model)
    modules = [head] + adapters
    allowed = ({id(parameter) for parameter in model.parameters()} if method in ['scratch', 'full_ft']
               else {id(parameter) for module in modules for parameter in module.parameters()})
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in allowed)
    if not training:
        model.eval()
    elif method in ['scratch', 'full_ft']:
        model.train()
    else:
        model.eval()
        head.train()
        for adapter in adapters:
            adapter.train()
    return allowed


def build_model(config, device='cpu'):
    protocol = load_json(ROOT / 'configs/protocol.json')
    key, method, seed = config['model_key'], config['method'], config['seed']
    classes = protocol['datasets'][config['dataset']]['num_classes']
    kwargs = {'num_classes': classes, 'drop_rate': 0.0, 'drop_path_rate': 0.0}
    if key == 'vitb16':
        kwargs.update(attn_drop_rate=0.0, proj_drop_rate=0.0, global_pool='token')
    # 只构造本地模型，显式加载已验真文件；不使用任何隐式 Hub 下载入口。
    with rng_stream(derive_seed('model', seed, key)):
        model = timm.create_model(protocol['models'][key]['model_id'], pretrained=False, **kwargs)
    original_classes = None
    if method != 'scratch':
        state = load_file(str(ROOT / 'weights' / key / 'model.safetensors'))
        classifier = 'fc' if key == 'r50' else 'head'
        original_classes = state[classifier + '.weight'].shape[0]
        assert original_classes == (1000 if key == 'r50' else 21843)
        del state[classifier + '.weight'], state[classifier + '.bias']
        result = model.load_state_dict(state, strict=False)
        assert set(result.missing_keys) == {classifier + '.weight', classifier + '.bias'}
        assert not result.unexpected_keys
        del state
    head = head_of(model, key)
    with rng_stream(derive_seed('head', seed, config['dataset'], key)):
        nn.init.trunc_normal_(head.weight, std=0.02)
        nn.init.zeros_(head.bias)
    info = {'initial_classifier_sha256': tensor_hash(head.state_dict().items()),
            'original_classes': original_classes, 'target_classes': classes,
            'constructor_kwargs': kwargs, 'pretrained_loaded': method != 'scratch'}
    if method == 'adapter':
        insert_adapters(model, key, seed)
    apply_strategy(model, key, method)
    return model.to(device), info


def parameter_counts(model, key):
    head = head_of(model, key)
    added = sum(p.numel() for module in adapters_of(model) for p in module.parameters())
    head_n = sum(p.numel() for p in head.parameters())
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'base_params': total - head_n - added, 'added_params': added, 'head_params': head_n,
            'total_params': total, 'trainable_params': trainable, 'trainable_ratio': trainable / total,
            'buffer_elements': sum(p.numel() for p in model.buffers())}


def make_optimizer(model, config):
    key, method = config['model_key'], config['method']
    head_ids = {id(p) for p in head_of(model, key).parameters()}
    no_decay_names = set(model.no_weight_decay()) if hasattr(model, 'no_weight_decay') else set()
    groups = {}
    manifests = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        # 所有 Norm/bias 均为一维；位置/类别嵌入使用模型自己的免衰减白名单。
        decay = 1e-4 if parameter.ndim >= 2 and name not in no_decay_names else 0.0
        multiplier = 10.0 if method == 'full_ft' and id(parameter) in head_ids else 1.0
        group = groups.setdefault((decay, multiplier), {'params': [], 'weight_decay': decay,
                                    'lr': config['learning_rate'] * multiplier,
                                    'peak_lr': config['learning_rate'] * multiplier})
        group['params'].append(parameter)
        manifests.append({'name': name, 'shape': list(parameter.shape), 'weight_decay': decay,
                          'peak_lr': config['learning_rate'] * multiplier})
    optimizer = torch.optim.AdamW(list(groups.values()), betas=(0.9, 0.999), eps=1e-8)
    ids = [id(p) for group in optimizer.param_groups for p in group['params']]
    allowed = {id(p) for p in model.parameters() if p.requires_grad}
    assert len(ids) == len(set(ids)) and set(ids) == allowed
    return optimizer, manifests
