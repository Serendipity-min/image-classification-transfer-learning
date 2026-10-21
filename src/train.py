"""按冻结 run ID 执行 P1；训练进程没有 test 数据入口。"""

import argparse
import copy
import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import time

import psutil
import torch

from tlcls.common import (ROOT, atomic_json, atomic_torch_save, load_json, provenance,
                          setup, sha256, sync_time)
from tlcls.data import TrainValData, epoch_order_hash, loader
from tlcls.engine import UpdateScheduler, load_resume, save_resume, train_epoch, validate
from tlcls.models import adapters_of, build_model, head_of, make_optimizer, parameter_counts


def append_epoch(path, row):
    # 仅在完整 epoch 和 checkpoint 成功后追加一行，半行不作为恢复边界。
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        file.flush()
        os.fsync(file.fileno())


def device_receipt(index):
    fields = 'index,uuid,name,pci.bus_id,memory.total,driver_version,power.limit,temperature.gpu,clocks.sm,clocks.mem'
    result = subprocess.run(['nvidia-smi', f'--id={index}', '--query-gpu=' + fields,
                             '--format=csv,noheader,nounits'], capture_output=True, text=True)
    return {'index': index, 'fields': fields.split(','), 'values': result.stdout.strip().split(', '),
            'cpu_quota_vcpu': 16, 'memory_quota_gib': 64}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--max-active-seconds', type=float, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    start = float(os.environ.get('TLCLS_PROCESS_STARTED_MONOTONIC', time.monotonic()))
    # 预留保存边界和退出开销，父进程另设总预算硬截止。
    deadline = start + max(0, args.max_active_seconds - 45)
    config = next(run for run in load_json(ROOT / 'configs/run_matrix.json')['planned_runs'] if run['run_id'] == args.run_id)
    asset = next(a for a in load_json(ROOT / 'records/assets.json')['assets']
                 if a['key'] == config['model_key'] + '_model.safetensors')
    config = dict(config, checkpoint_revision=asset['revision'], checkpoint_sha256=asset['sha256'])
    assert config['phase'] == 'P1', '本执行入口当前仅获授权 P1'
    checks = load_json(ROOT / 'records/implementation_checks.json')
    assert checks['status'] == 'passed' and checks['completed_configurations'] == 16
    prov = provenance(config['dataset'], config['model_key'])
    assert checks['provenance']['code_commit'] == prov['code_commit'], '源码已改变，须重验收'
    protocol = load_json(ROOT / 'configs/protocol.json')
    out = ROOT / 'runs' / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume and (out / 'summary.json').exists():
        raise RuntimeError('run 已有状态；请显式选择恢复或创建登记过的重试')
    torch.cuda.set_device(args.device)
    setup(config['seed'])
    device = torch.device('cuda', args.device)
    torch.cuda.reset_peak_memory_stats(device)
    summary = {'run_id': args.run_id, 'phase': 'P1', 'configuration_id': config['configuration_id'],
               'status': 'running', 'completed_epochs': 0, 'planned_epochs': 3,
               'provenance': prov, 'hardware': device_receipt(args.device),
               'test_evaluator_constructed': False, 'test_metrics': None,
               'precision': {'autocast': 'bfloat16', 'parameters': 'float32', 'loss': 'float32',
                             'grad_scaler': False, 'tf32': False, 'deterministic_algorithms': True},
               'parallel_peer_run_ids': []}
    atomic_json(out / 'summary.json', summary)
    completed_epoch, best_value, best_epoch = 0, -1.0, None
    costs = {'train_seconds': 0.0, 'validation_seconds': 0.0, 'checkpoint_seconds': 0.0,
             'fit_wall_seconds': 0.0, 'gpu_reserved_seconds': 0.0, 'recomputed_training_seconds': 0.0}
    previous_active = 0.0
    current_phase = None
    current_phase_started = None
    epoch_pending_commit = False
    current_epoch_train_seconds = 0.0
    model = optimizer = scheduler = None
    try:
        model, identity = build_model(config, device)
        optimizer, groups = make_optimizer(model, config)
        train_data = TrainValData(config['dataset'], config['model_key'], 'train', config['seed'])
        val_data = TrainValData(config['dataset'], config['model_key'], 'validation', config['seed'], raw=train_data.raw)
        train_loader, sampler = loader(train_data, config['seed'], True)
        val_loader, _ = loader(val_data, config['seed'], False)
        scheduler = UpdateScheduler(optimizer, math.ceil(len(train_data) / 128))
        resolved = dict(config, provenance=prov, model_identity=identity,
                        hardware=summary['hardware'], num_workers=4, persistent_workers=True,
                        normalization=protocol['models'][config['model_key']],
                        deterministic_algorithms=True, tf32=False,
                        sampler='numpy.PCG64_per_epoch_SHA256_v1',
                        augmentation='canonical_json_array_utf8_SHA256_low64_PCG64',
                        actual_launch_arguments=vars(args), physical_batch_size=32,
                        effective_batch_size=128, no_feature_cache=True, no_torch_compile=True)
        atomic_json(out / 'config.json', resolved)
        prov['resolved_config_sha256'] = sha256(out / 'config.json')
        atomic_json(out / 'parameter_groups.json', {'provenance': prov, 'groups': groups})
        summary.update(model_identity=identity, parameters=parameter_counts(model, config['model_key']),
                       startup_seconds=time.monotonic() - start)
        if args.resume:
            payload = load_resume(out / 'last.pt', model, optimizer, scheduler, config)
            assert payload['provenance']['protocol_sha256'] == prov['protocol_sha256']
            assert payload['provenance']['weight_sha256'] == prov['weight_sha256']
            assert payload['provenance']['split_sha256'] == prov['split_sha256']
            completed_epoch, best_value, best_epoch = payload['completed_epoch'], payload['best_value'], payload['best_epoch']
            costs.update(payload['costs'])
            timing_sidecar = out / 'active_costs.json'
            if timing_sidecar.exists():
                costs.update(load_json(timing_sidecar))
            previous_active = costs['gpu_reserved_seconds']
            summary['completed_epochs'] = completed_epoch
        for epoch in range(completed_epoch, config['epochs']):
            if time.monotonic() >= deadline:
                raise TimeoutError('gpu_budget_before_epoch')
            def heartbeat(progress):
                atomic_json(out / 'progress.json', dict(progress,
                            fit_active_seconds=previous_active + time.monotonic() - start,
                            cpu_rss_bytes=psutil.Process().memory_info().rss))
            sampler.epoch = epoch
            order_hash = epoch_order_hash(sampler)
            train_started = sync_time()
            current_phase, current_phase_started, epoch_pending_commit = 'train', train_started, True
            training = train_epoch(model, optimizer, scheduler, train_loader, sampler, config,
                                   epoch, device, deadline, heartbeat)
            train_seconds = sync_time() - train_started
            costs['train_seconds'] += train_seconds
            current_epoch_train_seconds = train_seconds
            val_started = sync_time()
            current_phase, current_phase_started = 'validation', val_started
            validation = validate(model, val_loader, device)
            val_seconds = sync_time() - val_started
            costs['validation_seconds'] += val_seconds
            completed_epoch = epoch + 1
            save_started = sync_time()
            current_phase, current_phase_started = 'checkpoint', save_started
            improved = validation['top1_percent'] > best_value
            if improved:
                best_value, best_epoch = validation['top1_percent'], completed_epoch
                atomic_torch_save(out / 'best_val.pt', {'model': model.state_dict(), 'config': config,
                                  'epoch': completed_epoch, 'validation': validation, 'provenance': prov,
                                  'includes_all_buffers': True})
            # last 始终是完整 epoch 边界；当前 segment 最终时钟另用 sidecar 精确补齐。
            costs['gpu_reserved_seconds'] = previous_active + time.monotonic() - start
            costs['fit_wall_seconds'] = costs['gpu_reserved_seconds']
            save_resume(out / 'last.pt', model, optimizer, scheduler, config, completed_epoch,
                        best_value, best_epoch, costs, prov)
            checkpoint_seconds = sync_time() - save_started
            costs['checkpoint_seconds'] += checkpoint_seconds
            costs['gpu_reserved_seconds'] = previous_active + time.monotonic() - start
            costs['fit_wall_seconds'] = costs['gpu_reserved_seconds']
            atomic_json(out / 'active_costs.json', costs)
            row = {'epoch': completed_epoch, 'provenance': prov, 'train': training, 'validation': validation,
                   'train_seconds': train_seconds, 'validation_seconds': val_seconds,
                   'checkpoint_seconds': checkpoint_seconds, 'fit_active_seconds': costs['fit_wall_seconds'],
                   'lr_by_group_next_update': [group['lr'] for group in optimizer.param_groups],
                   'cumulative_optimizer_updates': scheduler.updates, 'sampler_order_sha256': order_hash,
                   'selected_best_epoch': best_epoch, 'status': 'completed'}
            append_epoch(out / 'epochs.jsonl', row)
            epoch_pending_commit = False
            current_phase = None
            summary.update(completed_epochs=completed_epoch, best_val_top1_percent=best_value,
                           selected_epoch=best_epoch, costs=dict(costs), optimizer_updates=scheduler.updates,
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
            atomic_json(out / 'summary.json', summary)
            print(json.dumps({'run_id': args.run_id, 'epoch': completed_epoch,
                              'val_top1': validation['top1_percent'], 'train_seconds': train_seconds,
                              'validation_seconds': val_seconds, 'fit_active_seconds': costs['fit_wall_seconds']}), flush=True)
        summary['status'] = 'completed'
    except Exception as exc:
        if isinstance(exc, FloatingPointError):
            status = 'numerical_failure'
        elif isinstance(exc, torch.cuda.OutOfMemoryError):
            status = 'oom'
        elif isinstance(exc, TimeoutError):
            status = 'infrastructure_interrupted'
        else:
            status = 'invalid_protocol'
        summary.update(status=status, failure={'type': type(exc).__name__, 'reason': str(exc),
                       'epoch': completed_epoch + 1, 'optimizer_step': scheduler.updates if scheduler else None,
                       'last_finite_checkpoint': str(out / 'last.pt') if (out / 'last.pt').exists() else None})
        if status == 'numerical_failure' and model is not None:
            atomic_torch_save(out / 'failure_state.pt', {'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                'failure': summary['failure'], 'provenance': prov})
        print(json.dumps({'run_id': args.run_id, 'status': status, 'failure': summary['failure']}), flush=True)
    finally:
        finalization_started = time.monotonic()
        torch.cuda.synchronize()
        costs['gpu_reserved_seconds'] = previous_active + time.monotonic() - start
        costs['fit_wall_seconds'] = costs['gpu_reserved_seconds']
        if summary['status'] != 'completed' and epoch_pending_commit:
            # 仅给尚未入账的活动阶段补时；总适配时间包含这些分量，不再相加。
            elapsed = max(0.0, time.monotonic() - current_phase_started) if current_phase_started else 0.0
            if current_phase == 'train':
                costs['train_seconds'] += elapsed
                costs['recomputed_training_seconds'] += elapsed
            elif current_phase == 'validation':
                costs['validation_seconds'] += elapsed
                costs['recomputed_training_seconds'] += current_epoch_train_seconds
            elif current_phase == 'checkpoint':
                costs['checkpoint_seconds'] += elapsed
                costs['recomputed_training_seconds'] += current_epoch_train_seconds
        summary.update(costs=costs, gpu_hours=costs['gpu_reserved_seconds'] / 3600,
                       completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                       peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                       cpu_rss_bytes=psutil.Process().memory_info().rss)
        if model is not None and (out / 'best_val.pt').exists():
            summary['best_val_checkpoint_sha256'] = sha256(out / 'best_val.pt')
            summary['best_val_checkpoint_bytes'] = (out / 'best_val.pt').stat().st_size
        if summary['status'] == 'completed' and config['method'] in ['linear_probe', 'adapter']:
            delta_save_started = time.monotonic()
            selected = torch.load(out / 'best_val.pt', map_location='cpu', weights_only=False)
            module_ids = {id(module) for module in [head_of(model, config['model_key']), *adapters_of(model)]}
            delta = {}
            for name, module in model.named_modules():
                if id(module) in module_ids:
                    for local_name in module.state_dict():
                        full_name = name + '.' + local_name if name else local_name
                        delta[full_name] = selected['model'][full_name]
            # 按完整模块 state_dict 保存，包含 Adapter BN buffers，绑定基础权重身份。
            atomic_torch_save(out / 'task_delta.pt', {'state_dict': delta, 'provenance': prov,
                              'base_weight_sha256': prov['weight_sha256'], 'selected_epoch': best_epoch,
                              'includes_adapter_buffers': True})
            summary['task_delta_bytes'] = (out / 'task_delta.pt').stat().st_size
            summary['task_delta_sha256'] = sha256(out / 'task_delta.pt')
            costs['checkpoint_seconds'] += time.monotonic() - delta_save_started
        if (out / 'last.pt').exists():
            summary['last_checkpoint_sha256'] = sha256(out / 'last.pt')
            summary['last_checkpoint_bytes'] = (out / 'last.pt').stat().st_size
        # 将末尾哈希、任务增量存盘等实际开销纳入累计 fit，不能在恢复或报告中丢失。
        costs['gpu_reserved_seconds'] = previous_active + time.monotonic() - start
        costs['fit_wall_seconds'] = costs['gpu_reserved_seconds']
        summary['gpu_hours'] = costs['gpu_reserved_seconds'] / 3600
        summary['terminal_overhead_seconds'] = time.monotonic() - finalization_started
        atomic_json(out / 'active_costs.json', costs)
        atomic_json(out / 'summary.json', summary)
    return 0 if summary['status'] == 'completed' else 2


if __name__ == '__main__':
    raise SystemExit(main())
