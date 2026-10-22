"""P2 独立调度器：48 个登记候选、两卡共享 30 GPU-hours、校准与资源保护。"""

import argparse
from collections import Counter, defaultdict
import fcntl
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import uuid

from tlcls.p2_budget import Budget, Storage
from tlcls.p2_core import (CAP_SECONDS, EXIT_RESERVE_SECONDS, ROOT, STAGE, candidates, code_commit,
                           execution_identity, file_hash, process_alive, process_ticks, read,
                           require_authorization, scientific_identity, write)
from tlcls.p2_core import operational_compatibility


def launch(run_id, device, kind, resume=False):
    attempt = uuid.uuid4().hex
    (STAGE / 'logs').mkdir(exist_ok=True)
    log = (STAGE / 'logs' / (attempt + '.log')).open('ab')
    if kind == 'gpu_acceptance':
        child = [sys.executable, 'src/p2_gpu_acceptance.py']
    else:
        child = [sys.executable, 'src/p2_train.py', '--run-id', run_id, '--device', str(device)]
        if resume:
            child.append('--resume')
    command = [sys.executable, 'src/p2_attempt.py', '--attempt-id', attempt, '--run-id', run_id,
        '--device', str(device), '--owner-pid', str(os.getpid()), '--owner-ticks', process_ticks(os.getpid()),
        '--kind', kind, '--', *child]
    guardian = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONUNBUFFERED='1'), start_new_session=True)
    return {'guardian': guardian, 'log': log, 'attempt_id': attempt, 'run_id': run_id, 'device': device}


def baseline():
    value = read(ROOT / 'records/budget_plan.json')
    return {r['configuration_id']: r for r in value['per_configuration']}


def ensure_plan(runs):
    path = STAGE / 'candidate_plan.json'
    if path.exists():
        value = read(path)
        # 原设备分配和科学候选计划不可改写成新 commit；仅允许显式运维兼容的原身份。
        assert value['execution_identity'] in [execution_identity(), operational_compatibility()['scientific_implementation_basis']]
        assert [r['config'] for r in value['runs']] == runs
        assert all(item['scientific_identity'] == scientific_identity(item['config']) for item in value['runs'])
        return value
    estimates = baseline()
    assignments, loads = {}, [0.0, 0.0]
    counts = defaultdict(Counter)
    first = ['P2-cifar100-r50-full_ft-lr2-s123', 'P2-svhn-vitb16-adapter-lr2-s123']
    def duration(r):
        b = estimates[r['configuration_id']]
        return 15 * b['epoch_seconds'] + b['per_run_fixed_and_first_epoch_excess_seconds']
    def assign(r, device):
        assignments[r['run_id']] = device
        counts[r['method']][device] += 1
        loads[device] += duration(r)
    for device, run_id in enumerate(first):
        assign(next(r for r in runs if r['run_id'] == run_id), device)
    for r in sorted(runs, key=lambda r: (-duration(r), r['run_id'])):
        if r['run_id'] in assignments:
            continue
        allowed = [d for d in [0, 1] if counts[r['method']][d] < 6]
        assign(r, min(allowed, key=lambda d: (loads[d], counts[r['method']][d], d)))
    assert all(c[0] == c[1] == 6 for c in counts.values())
    value = {'phase': 'P2', 'execution_commit': code_commit(), 'execution_identity': execution_identity(),
        'first_calibration_candidates': first, 'calibration_epochs': [2, 3],
        'runs': [{'config': r, 'physical_gpu_index': assignments[r['run_id']],
                  'scientific_identity': scientific_identity(r)} for r in runs],
        'method_device_counts': {k: dict(v) for k, v in counts.items()},
        'single_gpu_baseline_load_seconds': loads, 'P3_authorized': False}
    write(path, value)
    return value


def statuses(runs):
    result = {}
    for run in runs:
        path = ROOT / 'runs' / run['run_id'] / 'summary.json'
        result[run['run_id']] = read(path) if path.exists() else {'status': 'planned', 'completed_epochs': 0}
    return result


def calibrate(runs):
    base = baseline()
    observed, flags = [], []
    values = defaultdict(list)
    for run in runs:
        directory = ROOT / 'runs' / run['run_id']
        paths = [directory / 'timings' / f'epoch_{i:03d}.json' for i in [2, 3]]
        if not all(p.exists() for p in paths):
            continue
        timings = [read(p) for p in paths]
        t = statistics.mean(item['fit_seconds'] for item in timings)
        ref = base[run['configuration_id']]['epoch_seconds']
        entry = {'run_id': run['run_id'], 'configuration_id': run['configuration_id'],
            'epoch2_3_fit_seconds': t, 'P1_epoch2_3_baseline_seconds': ref,
            'slowdown_ratio': t / ref, 'timing_sha256': [file_hash(p) for p in paths],
            'data_wait_seconds_mean': statistics.mean(x['data_wait_seconds'] for x in timings),
            'above_10_percent': t > 1.1 * ref}
        observed.append(entry)
        values[run['configuration_id']].append(t)
        if entry['above_10_percent']:
            flags.append(entry['run_id'])
    estimate = {name: statistics.median(values[name]) if values[name] else b['epoch_seconds'] * 1.078775047654268
                for name, b in base.items()}
    write(STAGE / 'admission_estimates.json', {'epoch_seconds': estimate,
        'basis': 'median P2 complete epoch2/3 transaction timing when available; otherwise P1 times measured contention scenario'})
    value = {'observations': observed, 'above_10_percent_run_ids': flags,
             'threshold': 1.1, 'test_accuracy_used': False}
    write(STAGE / 'cost_calibration.json', value)
    return estimate, flags


def forecast(runs, state, estimates, budget):
    base = baseline()
    remaining = 0.0
    for run in runs:
        item = state[run['run_id']]
        if item['status'] in ['completed', 'numerical_failure']:
            continue
        epochs = item.get('completed_epochs', 0)
        remaining += (15 - epochs) * estimates[run['configuration_id']]
        if epochs == 0:
            remaining += base[run['configuration_id']]['per_run_fixed_and_first_epoch_excess_seconds']
    used = budget.used(budget.snapshot())
    value = {'actual_gpu_seconds_including_active': used, 'remaining_estimated_gpu_seconds': remaining,
        'projected_total_gpu_hours': (used + remaining) / 3600, 'cap_gpu_hours': 30,
        'partial_uncommitted_epoch_forecast': 'conservative full epoch remains; actual ledger never double counts',
        'forecast_exceeds_cap': used + remaining + EXIT_RESERVE_SECONDS > CAP_SECONDS}
    write(STAGE / 'forecast.json', value)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--acceptance-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    authorization = require_authorization(require_acceptance=not args.acceptance_only)
    STAGE.mkdir(exist_ok=True)
    lock = (STAGE / 'scheduler.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    budget, storage = Budget(), Storage(authorization['confirmed_project_environment_allocation_bytes'])
    snapshot = budget.snapshot()
    assert all(a['status'] == 'finalized' for a in snapshot['attempts'].values()), '存在未结算 attempt，禁止重复发任务'
    if snapshot['stop_request']:
        assert args.resume
        budget.clear_technical_stop()
    storage.require(3 * 2**30)
    write(STAGE / 'scheduler.json', {'pid': os.getpid(), 'start_ticks': process_ticks(os.getpid()),
        'execution_commit': code_commit(), 'phase': 'P2', 'acceptance_only': args.acceptance_only})
    active = {}
    interrupted = False
    def stop_handler(signum, frame):
        nonlocal interrupted
        interrupted = True
        budget.stop('operator_technical_stop')
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGINT, stop_handler)
    if args.acceptance_only:
        cpu = read(STAGE / 'cpu_acceptance.json')
        assert cpu['status'] == 'passed' and cpu['execution_commit'] == code_commit()
        assert cpu['execution_identity'] == execution_identity()
        task = launch('P2-gpu-acceptance', 0, 'gpu_acceptance')
        task['guardian'].wait()
        task['log'].close()
        assert task['guardian'].returncode == 0
        gpu = read(STAGE / 'gpu_acceptance.json')
        assert gpu['status'] == 'passed' and gpu['execution_commit'] == code_commit()
        write(STAGE / 'acceptance.json', {'status': 'passed', 'execution_commit': code_commit(),
            'execution_identity': execution_identity(), 'authorization_sha256': file_hash(STAGE / 'authorization.json'),
            'cpu_receipt_sha256': file_hash(STAGE / 'cpu_acceptance.json'),
            'gpu_receipt_sha256': file_hash(STAGE / 'gpu_acceptance.json'),
            'legacy_P0_receipt_sha256': file_hash(ROOT / 'records/implementation_checks.json'),
            'all_GPU_acceptance_charged_to_P2': True})
        receipt = read(STAGE / 'acceptance.json')
        receipt['operational_compatibility'] = operational_compatibility()
        write(STAGE / 'acceptance.json', receipt)
        return 0
    runs = candidates()
    plan = ensure_plan(runs)
    priority = plan['first_calibration_candidates']
    assignments = {x['config']['run_id']: x['physical_gpu_index'] for x in plan['runs']}
    hold = None
    try:
        while True:
            for device, task in list(active.items()):
                if task['guardian'].poll() is None:
                    continue
                task['log'].close()
                budget.reconcile_dead_guardian(task['attempt_id'])
                item = read(ROOT / 'runs' / task['run_id'] / 'summary.json') if (ROOT / 'runs' / task['run_id'] / 'summary.json').exists() else {}
                if item.get('status') == 'running' or not item:
                    # SIGKILL 不依赖子 finally；父账本已结算，保留最后事务作为恢复边界。
                    item.update(run_id=task['run_id'], phase='P2', status='infrastructure_interrupted',
                                failure={'reason': 'child_exited_without_terminal_summary', 'attempt_id': task['attempt_id']})
                    write(ROOT / 'runs' / task['run_id'] / 'summary.json', item)
                if item['status'] in ['invalid_protocol', 'oom']:
                    hold = 'technical_failure_requires_investigation:' + task['run_id']
                attempts = [a for a in budget.snapshot()['attempts'].values() if a['run_id'] == task['run_id']]
                if item['status'] == 'infrastructure_interrupted' and len(attempts) >= 2:
                    hold = 'repeated_infrastructure_interruption:' + task['run_id']
                del active[device]
            state = statuses(runs)
            estimates, slow = calibrate(runs)
            prediction = forecast(runs, state, estimates, budget)
            override = read(STAGE / 'dispatch_override.json') if (STAGE / 'dispatch_override.json').exists() else {}
            acknowledged = override.get('acknowledged_slow_run_ids', []) if override.get('investigation_completed') else []
            unreviewed = [name for name in slow if name not in acknowledged]
            if unreviewed:
                hold = 'slowdown_above_10_percent_investigate_before_dispatch'
            if prediction['forecast_exceeds_cap']:
                hold = 'projected_total_exceeds_30_gpu_hours'
            try:
                resources = storage.require(3 * 2**30)
                write(STAGE / 'storage_latest.json', resources)
                if resources['warning_54_gib']:
                    hold = 'storage_54_gib_warning_pause_dispatch'
            except RuntimeError:
                hold = 'storage_allocation_or_exit_space_insufficient'
                budget.stop(hold)
            if budget.snapshot()['stop_request'] or interrupted:
                hold = 'shared_stop_requested'
            all_terminal = all(x['status'] in ['completed', 'numerical_failure'] for x in state.values())
            progress = {'status': 'paused_dispatch' if hold else 'running', 'reason': hold,
                'completed': sum(x['status'] == 'completed' for x in state.values()),
                'numerical_failures': sum(x['status'] == 'numerical_failure' for x in state.values()),
                'active': [{k: v for k, v in task.items() if k not in ['guardian', 'log']} for task in active.values()],
                'forecast': prediction, 'slowdown_flags': slow, 'updated_unix': time.time()}
            write(STAGE / 'progress.json', progress)
            if not active and (hold or all_terminal):
                break
            max_parallel = override.get('max_parallel', 2)
            assert max_parallel in [1, 2]
            ready = all(state[name].get('completed_epochs', 0) >= 3 or state[name]['status'] == 'numerical_failure'
                        for name in priority)
            if not hold:
                for device in [0, 1]:
                    if device in active or len(active) >= max_parallel:
                        continue
                    waiting = [r for r in runs if assignments[r['run_id']] == device and
                        state[r['run_id']]['status'] in ['planned', 'infrastructure_interrupted'] and
                        r['run_id'] not in [a['run_id'] for a in active.values()]]
                    if not ready:
                        waiting = [r for r in waiting if r['run_id'] in priority]
                    # 已有完整提交的中断候选先恢复，避免继续发新任务而搁置已保存前缀。
                    waiting.sort(key=lambda r: (state[r['run_id']]['status'] != 'infrastructure_interrupted', r['run_id'] not in priority,
                        -15 * estimates[r['configuration_id']], r['run_id']))
                    if waiting:
                        r = waiting[0]
                        snapshot = budget.snapshot()
                        needed = 1.15 * estimates[r['configuration_id']] * (len(active) + 1) + EXIT_RESERVE_SECONDS
                        if budget.used(snapshot) + needed >= budget.cap:
                            hold = 'insufficient_budget_for_next_legal_epoch_boundary'
                            break
                        active[device] = launch(r['run_id'], device, 'candidate',
                            resume=(ROOT / 'runs' / r['run_id'] / 'scientific.json').exists())
            time.sleep(3)
    finally:
        if active:
            request = budget.stop('operator_technical_stop')
            # 所有守护进程共享同一个期限；不逐个等待完整 grace，避免两卡超预算。
            while active and time.monotonic() < request['drain_deadline_monotonic'] + 3:
                for device, task in list(active.items()):
                    if task['guardian'].poll() is not None:
                        task['log'].close()
                        del active[device]
                time.sleep(.1)
    write(STAGE / 'stage_status.json', {'status': 'paused_review_required' if hold else 'P2_candidates_terminal',
        'reason': hold, 'execution_commit': code_commit(), 'P3_authorized': False,
        'centralized_test_authorized': False, 'gpu_hours': budget.used(budget.snapshot()) / 3600})
    subprocess.run([sys.executable, 'src/p2_select.py'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, 'src/p2_report.py'], cwd=ROOT, check=True)
    return 2 if hold else 0


if __name__ == '__main__':
    raise SystemExit(main())
