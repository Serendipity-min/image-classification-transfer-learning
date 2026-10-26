"""独立 P3 候选入口：50 轮、共享预算、事务恢复；没有 test 数据入口。"""

import argparse
import datetime
import math
import os
from pathlib import Path
import signal
import subprocess
import time

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import psutil
import torch

from tlcls.common import rng_state, restore_rng, setup, sync_time
from tlcls.data import TrainValData, epoch_order_hash, loader
from tlcls.engine import UpdateScheduler, train_epoch, validate
from tlcls.models import (adapters_of, apply_strategy, build_model, head_of, make_optimizer,
                           parameter_counts)
from tlcls.p3_budget import Budget, Storage
from tlcls.p3_core import (EXIT_RESERVE_SECONDS, ROOT, STAGE, candidates, code_commit,
                           file_hash, read, require_authorization, scientific_identity, write)
from tlcls.p3_core import infrastructure_exception
from tlcls.p3_core import execution_identity, operational_compatibility, compatible_execution
from tlcls.p3_transaction import EpochTransaction, committed_summary


class StageInterruption(RuntimeError):
    pass


def reopen_future_prefixes(transaction, elapsed_seconds, attempt_id):
    path = transaction.root / 'prefixes.json'
    if not path.exists(): return
    previous = read(path)
    value = read(path)
    changed = False
    for item in value['items']:
        # 基础设施暂停不等于轨迹最终结束：只重开尚未耗尽的预算点，截止已过的切片保持锁定。
        if elapsed_seconds < item['budget_seconds'] and 'actual_run_stop_gpu_seconds' in item:
            item['status'] = 'pending_candidate' if item['checkpoint'] else 'pending'
            item['previous_interruption_stop_gpu_seconds'] = item.pop('actual_run_stop_gpu_seconds')
            changed = True
    if changed:
        write(transaction.root / 'prefix_resume_history' / (attempt_id + '.json'), previous)
        write(path, value)


class TimedLoader:
    def __init__(self, underlying, stopped):
        self.underlying, self.stopped = underlying, stopped
        self.dataset = underlying.dataset
        self.data_wait_seconds = 0.0

    def __len__(self):
        return len(self.underlying)

    def __iter__(self):
        self.data_wait_seconds = 0.0
        stream = iter(self.underlying)
        while True:
            if self.stopped():
                raise StageInterruption('attempt_stop_requested')
            started = time.monotonic()
            try:
                batch = next(stream)
            except StopIteration:
                return
            self.data_wait_seconds += time.monotonic() - started
            yield batch


def restore_checkpoint(transaction, commit, model, optimizer, scheduler, config, expected_sha):
    payload = torch.load(transaction.path(commit['last']['path']), map_location='cpu', weights_only=False)
    compatible_execution(payload['execution_commit'], payload['runtime_execution_identity'])
    assert payload['scientific_config_sha256'] == expected_sha, 'checkpoint 科学配置不一致'
    assert payload['config'] == config and payload['completed_epoch'] == commit['completed_epoch']
    peaks = [group['peak_lr'] for group in optimizer.param_groups]
    model.load_state_dict(payload['model'], strict=True)
    optimizer.load_state_dict(payload['optimizer'])
    assert [group['peak_lr'] for group in optimizer.param_groups] == peaks, '恢复后的 LR 身份不一致'
    scheduler.load_state_dict(payload['scheduler'])
    apply_strategy(model, config['model_key'], config['method'], training=True)
    # 模型、优化器、模式和数据对象构造完以后，最后恢复独立随机流。
    restore_rng(payload['rng'])
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    authorization = require_authorization()
    budget = Budget()
    attempt_id = os.environ['TLCLS_ATTEMPT_ID']
    registered = budget.snapshot()['attempts'][attempt_id]
    assert registered['run_id'] == args.run_id and registered['kind'] == 'candidate'
    assert registered['device'] == args.device and registered['status'] == 'running'
    config = next(r for r in candidates() if r['run_id'] == args.run_id)
    identity = scientific_identity(config)
    scientific_sha = identity['scientific_config_sha256']
    directory = ROOT / 'runs' / args.run_id
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / 'scientific.json'
    if config_path.exists():
        assert read(config_path) == identity, '科学配置改变，拒绝沿用 run ID 或恢复'
        assert args.resume, '已有候选必须显式恢复，不能重复初始化争取更好成绩'
    else:
        assert not args.resume
        write(config_path, identity)
    # 科学身份是初始化提交点；若其后的派生 config 写入前断电，只能按同一身份补写。
    if (directory / 'config.json').exists():
        assert read(directory / 'config.json') == config
    else:
        write(directory / 'config.json', config)
    transaction = EpochTransaction(directory, scientific_sha, config=config)
    current, rows = transaction.load()
    if current:
        assert args.resume and current['completed_epoch'] <= 50
        reopen_future_prefixes(transaction, budget.run_seconds(args.run_id), attempt_id)
    storage = Storage(authorization['confirmed_project_environment_allocation_bytes'])
    storage.require(4 * 2**30)
    stopped = False
    def stop_handler(signum, frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    torch.cuda.set_device(args.device)
    setup(config['seed'])
    device = torch.device('cuda', args.device)
    torch.cuda.reset_peak_memory_stats(device)
    summary = {'run_id': args.run_id, 'phase': 'P3', 'configuration_id': config['configuration_id'],
        'planned_epochs': 50, 'completed_epochs': current['completed_epoch'] if current else 0,
        'status': 'running', 'scientific_config_sha256': scientific_sha,
        'execution_commit': code_commit(), 'implementation': execution_identity(),
        'scientific_implementation_basis': identity['scientific_config']['implementation'],
        'operational_compatibility': operational_compatibility(),
        'attempt_ids': [a['attempt_id'] for a in budget.snapshot()['attempts'].values() if a['run_id'] == args.run_id],
        'test_evaluator_constructed': False, 'test_metrics': None,
        'scheduler_horizon_epochs': 50, 'warmup_epochs': 5,
        'physical_gpu_index': args.device, 'parallel_peer_run_ids': [],
        'gpu_identity': read(STAGE / 'entry_preflight.json')['gpu_devices'][args.device],
        'fresh_initialization': not bool(current), 'selected_hparams_sha256': config['selected_hparams_sha256']}
    if current:
        summary.update(committed_summary(current, rows, read(directory / 'current.json')))
    write(directory / 'summary.json', summary)
    model = optimizer = scheduler = None
    start = time.monotonic()
    try:
        model, model_identity = build_model(config, device)
        optimizer, groups = make_optimizer(model, config)
        train_data = TrainValData(config['dataset'], config['model_key'], 'train', config['seed'])
        validation_data = TrainValData(config['dataset'], config['model_key'], 'validation', config['seed'], raw=train_data.raw)
        train_loader, sampler = loader(train_data, config['seed'], True)
        validation_loader, _ = loader(validation_data, config['seed'], False)
        scheduler = UpdateScheduler(optimizer, math.ceil(len(train_data) / 128))
        timed_loader = TimedLoader(train_loader, lambda: stopped)
        write(directory / 'parameter_groups.json', {'scientific_config_sha256': scientific_sha, 'groups': groups})
        write(directory / 'attempts' / (attempt_id + '.json'), {
            'attempt_id': attempt_id, 'scientific_config_sha256': scientific_sha, 'launch': vars(args),
            'actual_execution_commit': code_commit(), 'num_workers': 4, 'persistent_workers': True,
            'registered_parent_record': registered, 'resumed_generation': read(directory / 'current.json') if current else None})
        if current:
            restore_checkpoint(transaction, current, model, optimizer, scheduler, config, scientific_sha)
        summary.update(model_identity=model_identity, parameters=parameter_counts(model, config['model_key']),
                       startup_this_attempt_seconds=time.monotonic() - start)
        best_epoch = current['best_epoch'] if current else None
        best_value = max((r['validation']['top1_percent'] for r in rows), default=-1.0)
        for epoch in range(current['completed_epoch'] if current else 0, 50):
            snapshot = budget.snapshot()
            peers = [a['run_id'] for a in snapshot['attempts'].values() if a['status'] != 'finalized' and
                     a['kind'] == 'candidate' and a['run_id'] != args.run_id]
            summary['parallel_peer_run_ids'] = peers
            if stopped or snapshot['stop_request']:
                raise StageInterruption('shared_stop_before_epoch')
            estimated = read(STAGE / 'admission_estimates.json')['epoch_seconds'][config['configuration_id']]
            if budget.used(snapshot) + EXIT_RESERVE_SECONDS + estimated * (len(peers) + 1) * 1.15 > budget.spending_limit:
                raise StageInterruption('budget_cannot_cover_next_complete_epoch_and_exit')
            storage_snapshot = storage.require(4 * 2**30)
            write(STAGE / 'storage_latest.json', storage_snapshot)
            sampler.epoch = epoch
            def heartbeat(value):
                write(directory / 'progress.json', dict(value, attempt_id=attempt_id,
                    scientific_config_sha256=scientific_sha, cpu_rss_current_bytes=psutil.Process().memory_info().rss,
                    cumulative_parent_gpu_seconds=budget.run_seconds(args.run_id)))
            train_started = sync_time()
            training = train_epoch(model, optimizer, scheduler, timed_loader, sampler, config, epoch,
                                   device, heartbeat=heartbeat)
            train_seconds = sync_time() - train_started
            validation_started = sync_time()
            validation = validate(model, validation_loader, device)
            validation_seconds = sync_time() - validation_started
            improved = validation['top1_percent'] > best_value
            if improved:
                best_value, best_epoch = validation['top1_percent'], epoch + 1
            completed = epoch + 1
            payload = {'schema': 'P3_transaction_v1', 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'rng': rng_state(),
                'config': config, 'scientific_config_sha256': scientific_sha,
                'scientific_config': identity['scientific_config'], 'completed_epoch': completed,
                'best_epoch': best_epoch, 'best_value': best_value, 'checkpoint_boundary': 'completed_epoch',
                'execution_commit': code_commit(), 'includes_all_buffers': True}
            payload['runtime_execution_identity'] = execution_identity()
            payload['operational_compatibility'] = operational_compatibility()
            row = {'epoch': completed, 'status': 'completed', 'attempt_id': attempt_id,
                'scientific_config_sha256': scientific_sha, 'train': training, 'validation': validation,
                'train_seconds': train_seconds, 'validation_seconds': validation_seconds,
                'data_wait_seconds': timed_loader.data_wait_seconds,
                'cumulative_optimizer_updates': scheduler.updates, 'selected_best_epoch': best_epoch,
                'sampler_order_sha256': epoch_order_hash(sampler),
                'lr_by_group_next_update': [g['lr'] for g in optimizer.param_groups],
                'parallel_peer_run_ids': peers, 'physical_gpu_index': args.device,
                'cumulative_parent_gpu_seconds_before_commit': budget.run_seconds(args.run_id)}
            row['runtime_execution_commit'] = code_commit()
            row['runtime_execution_identity_sha256'] = execution_identity()['scope_sha256']
            best_payload = {'model': model.state_dict(), 'config': config, 'epoch': completed,
                'validation': validation, 'scientific_config_sha256': scientific_sha,
                'execution_commit': code_commit(), 'includes_all_buffers': True}
            checkpoint_started = sync_time()
            current, row = transaction.commit(completed, row,
                lambda file: torch.save(payload, file),
                (lambda file: torch.save(best_payload, file)),
                best_epoch, lambda: budget.run_seconds(args.run_id), improved=improved)
            checkpoint_full_seconds = sync_time() - checkpoint_started
            write(directory / 'timings' / f'epoch_{completed:03d}.json', {
                'epoch': completed, 'commit': read(directory / 'current.json'),
                'scientific_config_sha256': scientific_sha,
                'train_seconds': train_seconds, 'validation_seconds': validation_seconds,
                'checkpoint_seconds': checkpoint_full_seconds,
                'fit_seconds': train_seconds + validation_seconds + checkpoint_full_seconds,
                'data_wait_seconds': timed_loader.data_wait_seconds,
                'parent_gpu_seconds_after_commit': budget.run_seconds(args.run_id),
                'timing_region': 'epoch train + validation + complete transaction including checkpoint hash and GC'})
            rows.append(row)
            summary.update(completed_epochs=completed, best_val_top1_percent=best_value,
                selected_epoch=best_epoch, optimizer_updates=scheduler.updates,
                commit=read(directory / 'current.json'), best_checkpoint=current['best'],
                last_checkpoint=current['last'])
            write(directory / 'summary.json', summary)
            print(f'{args.run_id} epoch={completed} validation={validation["top1_percent"]:.6f}', flush=True)
        if config['method'] in ['adapter', 'linear_probe']:
            selected = torch.load(transaction.path(current['best']['path']), map_location='cpu', weights_only=False)
            module_ids = {id(m) for m in [head_of(model, config['model_key']), *adapters_of(model)]}
            delta = {}
            for name, module in model.named_modules():
                if id(module) in module_ids:
                    for local_name in module.state_dict():
                        full_name = name + '.' + local_name if name else local_name
                        delta[full_name] = selected['model'][full_name]
            storage.require(2**30)
            artifact = transaction.artifact('delta', 50, lambda file: torch.save({
                'state_dict': delta, 'scientific_config_sha256': scientific_sha,
                'base_weight_sha256': identity['scientific_config']['weight_sha256'],
                'selected_epoch': best_epoch, 'includes_adapter_buffers': True}, file))
            summary['task_delta'] = artifact
        transaction.finalize_prefixes(lambda: budget.run_seconds(args.run_id))
        summary['prefixes'] = read(directory / 'prefixes.json')
        summary['status'] = 'completed'
    except (StageInterruption, KeyboardInterrupt) as exc:
        summary.update(status='infrastructure_interrupted', failure={'reason': str(exc), 'attempt_id': attempt_id})
    except FloatingPointError as exc:
        summary.update(status='numerical_failure', failure={'reason': str(exc),
                       'diagnostics': getattr(exc, 'diagnostics', {}), 'attempt_id': attempt_id})
        if model is not None:
            summary['failure_state'] = transaction.artifact('failure', summary['completed_epochs'] + 1,
                lambda file: torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(), 'rng': rng_state(), 'failure': summary['failure'],
                    'scientific_config_sha256': scientific_sha, 'valid_for_selection': False}, file))
    except torch.cuda.OutOfMemoryError as exc:
        summary.update(status='oom', failure={'reason': str(exc), 'attempt_id': attempt_id})
    except Exception as exc:
        infrastructure = infrastructure_exception(exc, stopped or bool(budget.snapshot()['stop_request']))
        summary.update(status='infrastructure_interrupted' if infrastructure else 'invalid_protocol',
                       failure={'type': type(exc).__name__, 'reason': str(exc),
                       'attempt_id': attempt_id})
        if not infrastructure:
            raise
    finally:
        # 数值失败也必须封定已存在的切片；不得用失败后的模型替换最后合法 checkpoint。
        if (directory / 'prefixes.json').exists():
            transaction.finalize_prefixes(lambda: budget.run_seconds(args.run_id))
        # 终值不能误标成峰值；真实采样峰值及被 SIGKILL 的成本以独立父账本为准。
        summary.update(cpu_rss_final_bytes=psutil.Process().memory_info().rss,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            parent_gpu_seconds_observed=budget.run_seconds(args.run_id),
            finalized_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(directory / 'summary.json', summary)
    return 0 if summary['status'] == 'completed' else 75 if summary['status'] == 'infrastructure_interrupted' else 2


if __name__ == '__main__':
    raise SystemExit(main())
