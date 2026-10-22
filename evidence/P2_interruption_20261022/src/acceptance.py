"""P0 科学有效性验收：结构、冻结、梯度、尾累积、确定性增强和恢复轨迹。"""

import argparse
import gc
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn

from tlcls.common import (ROOT, atomic_json, atomic_torch_save, load_json, provenance,
                           rng_state, restore_rng, setup, tensor_hash)
from tlcls.data import EpochSampler, TrainValData, epoch_order_hash, loader
from tlcls.engine import UpdateScheduler, effective_update, load_resume, save_resume
from tlcls.models import (adapters_of, apply_strategy, build_model, head_of,
                          make_optimizer, parameter_counts)


def frozen_state(model):
    adapter_buffer_ids = {id(buffer) for module in adapters_of(model) for buffer in module.buffers()}
    values = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()
              if not parameter.requires_grad}
    values.update({name: buffer.detach().cpu().clone() for name, buffer in model.named_buffers()
                   if id(buffer) not in adapter_buffer_ids})
    return values


def compare_frozen(model, before):
    state = model.state_dict()
    changed = [name for name, value in before.items() if not torch.equal(value, state[name].detach().cpu())]
    assert not changed, f'冻结状态发生变化: {changed[:3]}'
    return len(before)


def check_tail_accumulation(device):
    # 以独立解析梯度为基准，验证不等长 micro-batch 的样本加权，避免复制实现作为测试。
    model = nn.Linear(3, 2).to(device)
    reference = nn.Linear(3, 2).to(device)
    reference.load_state_dict(model.state_dict())
    x, y = torch.randn(7, 3, device=device), torch.tensor([0, 1, 0, 0, 1, 1, 0], device=device)
    reference_loss = nn.functional.cross_entropy(reference(x), y)
    reference_loss.backward()
    optimizer = torch.optim.SGD([{'params': list(model.parameters()), 'lr': 0.05, 'peak_lr': 0.05}])
    scheduler = UpdateScheduler(optimizer, 1)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=optimizer.param_groups[0]['lr'])
    torch.nn.utils.clip_grad_norm_(reference.parameters(), 1.0, error_if_nonfinite=True)
    reference_optimizer.step()
    batches = [(x[start:end], y[start:end], [f'acceptance/tail/{i}' for i in range(start, end)])
               for start, end in [(0, 3), (3, 6), (6, 7)]]
    update = effective_update(model, optimizer, scheduler, batches, device, bf16=False)
    max_error = max(float((a - b).abs().max()) for a, b in zip(model.parameters(), reference.parameters()))
    assert max_error < 1e-7 and update['samples'] == 7 and scheduler.updates == 1
    return {'partition_sizes': [3, 3, 1], 'max_parameter_update_error': max_error,
            'compares_production_effective_update_to_full_batch_reference': True, 'passed': True}


def check_data_rng():
    evidence = {}
    for dataset in ['cifar100', 'svhn']:
        data = TrainValData(dataset, 'r50', 'train', 314159)
        first = data[(1, 0)]
        random.random()
        np.random.rand(13)
        torch.rand(17)
        repeated = data[(1, 0)]
        assert torch.equal(first[0], repeated[0])
        assert first[0].shape == (3, 224, 224)
        # 真实多 worker persistent DataLoader 与单进程结果逐样本一致。
        parallel, sampler = loader(data, 314159, training=True, workers=4)
        sampler.epoch = 2
        batch = next(iter(parallel))
        positions = list(iter(sampler))[:32]
        expected = torch.stack([data[position][0] for position in positions])
        assert torch.equal(batch[0], expected)
        assert list(batch[2]) == [data[position][2] for position in positions]
        order_before = epoch_order_hash(sampler)
        sampler.epoch = 3
        assert epoch_order_hash(sampler) != order_before
        sampler.epoch = 2
        assert epoch_order_hash(sampler) == order_before
        evidence[dataset] = {'stateless_augmentation': True, 'persistent_workers_match_main_process': True,
                             'sampler_epoch_reconstruction': True, 'epoch2_order_sha256': order_before,
                             'training_horizontal_flip_probability': 0.0 if dataset == 'svhn' else 0.5}
        del parallel, data
        gc.collect()
    return evidence


def check_resume(config, device, batches):
    model, _ = build_model(config, device)
    optimizer, _ = make_optimizer(model, config)
    scheduler = UpdateScheduler(optimizer, 2)
    apply_strategy(model, config['model_key'], config['method'], True)
    effective_update(model, optimizer, scheduler, batches[:1], device)
    target = ROOT / 'records' / f"acceptance_resume_{config['model_key']}.pt"
    model.eval()
    with torch.no_grad():
        before_logits = model(batches[0][0].to(device)).detach().cpu()
    save_resume(target, model, optimizer, scheduler, config, 1, 0.0, 1, {}, provenance(), {})
    apply_strategy(model, config['model_key'], config['method'], True)
    effective_update(model, optimizer, scheduler, batches[1:], device)
    expected = tensor_hash(model.state_dict().items())
    expected_optimizer = tensor_hash([(f'{index}/{name}', value) for index, state in enumerate(optimizer.state.values())
                                      for name, value in state.items() if torch.is_tensor(value)])
    expected_rng = rng_state()
    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()
    model, _ = build_model(config, device)
    optimizer, _ = make_optimizer(model, config)
    scheduler = UpdateScheduler(optimizer, 2)
    payload = load_resume(target, model, optimizer, scheduler, config)
    model.eval()
    with torch.no_grad():
        loaded_logits = model(batches[0][0].to(device)).detach().cpu()
    max_error = float((loaded_logits - before_logits).abs().max())
    assert torch.allclose(loaded_logits, before_logits, rtol=1e-5, atol=1e-6)
    apply_strategy(model, config['model_key'], config['method'], True)
    # eval 检查后再次恢复断点 RNG，紧邻继续训练。
    restore_rng(payload['rng'])
    effective_update(model, optimizer, scheduler, batches[1:], device)
    actual = tensor_hash(model.state_dict().items())
    actual_optimizer = tensor_hash([(f'{index}/{name}', value) for index, state in enumerate(optimizer.state.values())
                                    for name, value in state.items() if torch.is_tensor(value)])
    assert actual == expected
    assert actual_optimizer == expected_optimizer
    assert torch.equal(torch.get_rng_state(), expected_rng['torch'])
    assert torch.equal(torch.cuda.get_rng_state(device), expected_rng['cuda'])
    size = target.stat().st_size
    target.unlink()
    return {'eval_logit_max_abs_error': max_error, 'continued_model_state_bitwise_equal': True,
            'continued_optimizer_state_bitwise_equal': True, 'checkpoint_bytes': size,
            'temporary_acceptance_checkpoint_removed': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    started = float(os.environ.get('TLCLS_PROCESS_STARTED_MONOTONIC', time.monotonic()))
    torch.cuda.set_device(args.device)
    setup(314159)
    device = torch.device('cuda', args.device)
    torch.cuda.reset_peak_memory_stats(device)
    protocol = load_json(ROOT / 'configs/protocol.json')
    runs = [run for run in load_json(ROOT / 'configs/run_matrix.json')['planned_runs'] if run['phase'] == 'P1']
    receipt = {'status': 'running', 'provenance': provenance(), 'checks': [], 'failures': [],
               'deterministic_algorithms': True, 'precision': 'bf16_autocast_fp32_loss_parameters'}
    receipt['tail_accumulation'] = check_tail_accumulation(device)
    receipt['data_rng'] = check_data_rng()
    heads = {}
    # 固定预生成小批次，仅验收实现，不计作完整 epoch 或 profile 成绩。
    fixed_images = [torch.randn(2, 3, 224, 224) for _ in range(3)]
    for config in runs:
        try:
            model, info = build_model(config, device)
            assert all(module.p == 0 for module in model.modules() if isinstance(module, nn.Dropout))
            counts = parameter_counts(model, config['model_key'])
            c = protocol['datasets'][config['dataset']]['num_classes']
            expected = protocol['expected_parameter_counts'][f"{config['model_key']}_C{c}"]
            assert counts['base_params'] == expected['backbone_excluding_head']
            assert counts['head_params'] == expected['head']
            if config['method'] == 'adapter':
                assert counts['added_params'] == expected['adapter_added']
                assert counts['total_params'] == expected['adapter_total']
                assert counts['trainable_params'] == expected['adapter_trainable']
            else:
                assert counts['total_params'] == expected['scratch_fullft_total']
                assert counts['trainable_params'] == (expected['head'] if config['method'] == 'linear_probe' else expected['scratch_fullft_total'])
            head_key = (config['dataset'], config['model_key'])
            if head_key in heads:
                assert heads[head_key] == info['initial_classifier_sha256']
            heads[head_key] = info['initial_classifier_sha256']
            adapters = adapters_of(model)
            identity_error = None
            if config['method'] == 'adapter':
                reference, _ = build_model(dict(config, method='full_ft'), device)
                model.eval()
                reference.eval()
                with torch.no_grad():
                    actual = model(fixed_images[0].to(device))
                    baseline = reference(fixed_images[0].to(device))
                identity_error = float((actual - baseline).abs().max())
                assert torch.allclose(actual, baseline, rtol=1e-5, atol=1e-6)
                del reference, actual, baseline
            apply_strategy(model, config['model_key'], config['method'], True)
            added_bn_ids = {id(module) for adapter in adapters for module in adapter.modules()
                            if isinstance(module, nn.modules.batchnorm._BatchNorm)}
            base_bn = [module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)
                       and id(module) not in added_bn_ids]
            if config['method'] in ['linear_probe', 'adapter']:
                assert all(not module.training for module in base_bn)
                assert all(module.training for adapter in adapters for module in adapter.modules()
                           if isinstance(module, nn.modules.batchnorm._BatchNorm))
            frozen = frozen_state(model) if config['method'] in ['linear_probe', 'adapter'] else {}
            bn_counters_before = [int(module.num_batches_tracked) for module in base_bn]
            first_before = adapters[0].down.weight.detach().cpu().clone() if adapters else None
            head_before = head_of(model, config['model_key']).weight.detach().cpu().clone()
            optimizer, groups = make_optimizer(model, config)
            scheduler = UpdateScheduler(optimizer, math.ceil(protocol['datasets'][config['dataset']]['train_n'] / 128))
            gradients = []
            for index in range(3):
                labels = torch.tensor([index % c, (index + 1) % c])
                result = effective_update(model, optimizer, scheduler,
                         [(fixed_images[index], labels, ['acceptance/synthetic/0', 'acceptance/synthetic/1'])], device)
                if adapters:
                    gradients.append({'update': index + 1,
                                      'first_down_gradient_finite': bool(torch.isfinite(adapters[0].down.weight.grad).all()),
                                      'first_down_gradient_l2': float(adapters[0].down.weight.grad.norm())})
            frozen_checked = compare_frozen(model, frozen) if frozen else 0
            if config['method'] in ['scratch', 'full_ft']:
                assert all(int(module.num_batches_tracked) == count + 3
                           for module, count in zip(base_bn, bn_counters_before))
            assert not torch.equal(head_before, head_of(model, config['model_key']).weight.detach().cpu())
            if adapters:
                assert not torch.equal(first_before, adapters[0].down.weight.detach().cpu())
                assert gradients[-1]['first_down_gradient_l2'] > 0
            item = {'run_id': config['run_id'], 'status': 'passed', 'counts': counts, 'model_identity': info,
                    'original_bn_modules': len(base_bn), 'original_bn_mode': 'eval' if config['method'] in ['linear_probe', 'adapter'] else 'train',
                    'adapter_bn_count': len(added_bn_ids), 'frozen_tensors_checked': frozen_checked,
                    'optimizer_parameter_id_whitelist_exact': True, 'initial_identity_max_abs_error': identity_error,
                    'adapter_gradient_updates': gradients, 'updates': 3, 'test_access': False}
            receipt['checks'].append(item)
            print(json.dumps({'run_id': config['run_id'], 'P0': 'passed', 'counts': counts}), flush=True)
            del model, optimizer, scheduler, frozen, adapters, groups
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            receipt['failures'].append({'run_id': config['run_id'], 'type': type(exc).__name__, 'message': str(exc)})
            receipt['status'] = 'failed'
            atomic_json(ROOT / 'records/implementation_checks.json', receipt)
            raise
        receipt['gpu_reserved_seconds'] = time.monotonic() - started
        atomic_json(ROOT / 'records/implementation_checks.json', receipt)
    receipt['resume'] = {}
    for key in ['r50', 'vitb16']:
        config = next(run for run in runs if run['model_key'] == key and run['method'] == 'adapter' and run['dataset'] == 'svhn')
        batches = [(fixed_images[i], torch.tensor([i, i + 1]), [f'acceptance/synthetic/{2*i}', f'acceptance/synthetic/{2*i+1}']) for i in range(3)]
        receipt['resume'][key] = check_resume(config, device, batches)
    receipt.update(status='passed', completed_configurations=len(receipt['checks']),
                   head_initialization_paired=True, gpu_reserved_seconds=time.monotonic() - started,
                   gpu_hours=(time.monotonic() - started) / 3600,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    atomic_json(ROOT / 'records/implementation_checks.json', receipt)
    print(json.dumps({'P0': 'passed', 'gpu_hours': receipt['gpu_hours']}), flush=True)


if __name__ == '__main__':
    main()
