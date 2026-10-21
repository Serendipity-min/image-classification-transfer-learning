"""共享训练更新、有限值门禁、累积尾窗口与完整断点恢复。"""

import math
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .common import atomic_torch_save, finite_or_raise, restore_rng, rng_state
from .models import apply_strategy


class UpdateScheduler:
    def __init__(self, optimizer, updates_per_epoch):
        self.optimizer = optimizer
        self.warmup = 5 * updates_per_epoch
        self.horizon = 50 * updates_per_epoch
        self.updates = 0
        self.apply()

    def factor(self):
        index = self.updates
        if index < self.warmup:
            return 0.01 + 0.99 * index / max(1, self.warmup - 1)
        progress = (index - self.warmup) / max(1, self.horizon - self.warmup - 1)
        return 0.01 + 0.99 * (1 + math.cos(math.pi * min(1.0, progress))) / 2

    def apply(self):
        for group in self.optimizer.param_groups:
            group['lr'] = group['peak_lr'] * self.factor()

    def step(self):
        self.updates += 1
        self.apply()

    def state_dict(self):
        return {'updates': self.updates, 'warmup': self.warmup, 'horizon': self.horizon}

    def load_state_dict(self, state):
        assert state['warmup'] == self.warmup and state['horizon'] == self.horizon
        self.updates = state['updates']
        self.apply()


def finite_parameters_and_optimizer(parameters, optimizer):
    # foreach 减少逐参数同步开销；先检参数，再检 Adam 动量，异常位置可明确区分。
    norms = torch.stack(torch._foreach_norm(parameters))
    finite_or_raise(norms, 'updated_parameters')
    values = [value for state in optimizer.state.values() for name, value in state.items()
              if isinstance(value, torch.Tensor) and value.ndim > 0]
    if values:
        finite_or_raise(torch.stack(torch._foreach_norm(values)), 'optimizer_state')


def effective_update(model, optimizer, scheduler, batches, device, bf16=True):
    optimizer.zero_grad(set_to_none=True)
    samples = sum(len(batch[1]) for batch in batches)
    assert samples > 0
    loss_sum, correct = 0.0, 0
    histogram = None
    logits_min, logits_max = math.inf, -math.inf
    sample_ids = []
    for images, labels, identifiers in batches:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        finite_or_raise(images, 'input')
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
            logits = model(images)
        logits = logits.float()
        finite_or_raise(logits, 'forward_logits')
        assert int(labels.min()) >= 0 and int(labels.max()) < logits.shape[1]
        loss = F.cross_entropy(logits, labels, reduction='sum')
        finite_or_raise(loss, 'loss')
        # 当前窗口按真实样本数归一化，末尾不足四个 micro-batch 仍正确更新。
        (loss / samples).backward()
        loss_sum += float(loss.detach())
        predicted = logits.argmax(dim=1)
        correct += int((predicted == labels).sum())
        counts = torch.bincount(predicted, minlength=logits.shape[1]).cpu()
        histogram = counts if histogram is None else histogram + counts
        logits_min = min(logits_min, float(logits.detach().min()))
        logits_max = max(logits_max, float(logits.detach().max()))
        sample_ids.extend(identifiers)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    try:
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True, foreach=True)
    except RuntimeError as exc:
        if 'non-finite' in str(exc):
            raise FloatingPointError('unscaled_gradient_before_step') from exc
        raise
    grad_before = float(norm)
    lr = [group['lr'] for group in optimizer.param_groups]
    optimizer.step()
    finite_parameters_and_optimizer(parameters, optimizer)
    scheduler.step()
    return {'samples': samples, 'loss_sum': loss_sum, 'correct': correct,
            'grad_norm_before_clip': grad_before, 'grad_norm_after_clip_upper_bound': min(grad_before, 1.0),
            'lr_by_group': lr, 'logits_min': logits_min, 'logits_max': logits_max,
            'predicted_class_histogram': histogram.tolist(), 'sample_ids': sample_ids}


def train_epoch(model, optimizer, scheduler, dataloader, sampler, config, epoch, device,
                deadline=None, heartbeat=None):
    apply_strategy(model, config['model_key'], config['method'], training=True)
    sampler.epoch = epoch
    window, loss_sum, correct, samples, updates = [], 0.0, 0, 0, 0
    latest = None
    started = time.monotonic()
    for micro_step, batch in enumerate(dataloader):
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError('gpu_budget_stop_at_accumulation_boundary')
        window.append(batch)
        if len(window) == 4 or micro_step + 1 == len(dataloader):
            latest = effective_update(model, optimizer, scheduler, window, device)
            samples += latest['samples']
            loss_sum += latest['loss_sum']
            correct += latest['correct']
            updates += 1
            window = []
            if heartbeat and (updates % 25 == 0 or micro_step + 1 == len(dataloader)):
                heartbeat({'epoch': epoch + 1, 'micro_step': micro_step + 1,
                           'optimizer_step': scheduler.updates, 'epoch_samples': samples,
                           'epoch_train_seconds': time.monotonic() - started,
                           'latest_loss': latest['loss_sum'] / latest['samples']})
    assert samples == len(dataloader.dataset)
    assert updates == math.ceil(samples / 128)
    torch.cuda.synchronize()
    return {'loss': loss_sum / samples, 'top1_percent': 100 * correct / samples,
            'samples': samples, 'optimizer_updates': updates, 'last_update': latest}


@torch.no_grad()
def validate(model, dataloader, device):
    model.eval()
    loss_sum, correct, samples = 0.0, 0, 0
    for images, labels, _ in dataloader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(images)
        logits = logits.float()
        finite_or_raise(logits, 'validation_logits')
        loss = F.cross_entropy(logits, labels, reduction='sum')
        finite_or_raise(loss, 'validation_loss')
        loss_sum += float(loss)
        correct += int((logits.argmax(1) == labels).sum())
        samples += len(labels)
    assert samples == len(dataloader.dataset)
    torch.cuda.synchronize()
    return {'loss': loss_sum / samples, 'top1_percent': 100 * correct / samples,
            'samples': samples, 'correct': correct}


def save_resume(path, model, optimizer, scheduler, config, completed_epoch, best_value, best_epoch,
                costs, provenance, extra=None):
    payload = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
               'scheduler': scheduler.state_dict(), 'rng': rng_state(), 'config': config,
               'completed_epoch': completed_epoch, 'best_value': best_value, 'best_epoch': best_epoch,
               'costs': dict(costs), 'provenance': provenance,
               'sampler_reconstruction': 'numpy.default_rng.PCG64_SHA256_dataset_seed_epoch_v1',
               'checkpoint_boundary': 'completed_epoch', 'extra': extra or {}}
    atomic_torch_save(path, payload)


def load_resume(path, model, optimizer, scheduler, config):
    # 只加载本项目生成的可信状态；包括 NumPy/Python RNG，不能用 weights_only=True。
    payload = torch.load(path, map_location='cpu', weights_only=False)
    for name in ['run_id', 'dataset', 'model_key', 'method', 'seed']:
        assert payload['config'][name] == config[name]
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    scheduler.load_state_dict(payload['scheduler'])
    apply_strategy(model, config['model_key'], config['method'], training=True)
    restore_rng(payload['rng'])
    return payload
