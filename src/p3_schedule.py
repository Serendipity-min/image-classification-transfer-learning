"""P3 独立调度器：80 个正式运行、两卡共享 160 GPU-hours、校准与资源保护。"""

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

from tlcls.p3_budget import Budget, Storage
from tlcls.p3_core import (CAP_SECONDS, EXIT_RESERVE_SECONDS, ROOT, STAGE, candidates, code_commit,
                           execution_identity, file_hash, process_alive, process_ticks, read,
                           require_authorization, scientific_identity, write)
from tlcls.p3_core import operational_compatibility


def launch(run_id, device, kind, resume=False):
    # 在占用账本之前再次核对物理卡；外部新增进程不属于本项目，拒绝抢占或终止它们。
    expected_gpu = read(STAGE / 'entry_preflight.json')['gpu_devices'][device]['uuid']
    physical_gpu = subprocess.check_output(['nvidia-smi', '-i', str(device), '--query-gpu=uuid',
                                            '--format=csv,noheader'], text=True).strip()
    assert physical_gpu == expected_gpu, 'physical GPU identity changed'
    jobs = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader'], text=True).strip().splitlines()
    assert not any(line.split(',')[0].strip() == expected_gpu for line in jobs), 'target GPU occupied by another process'
    attempt = uuid.uuid4().hex
    (STAGE / 'logs').mkdir(exist_ok=True)
    log = (STAGE / 'logs' / (attempt + '.log')).open('ab')
    if kind == 'gpu_acceptance':
        child = [sys.executable, 'src/p3_gpu_acceptance.py']
    else:
        child = [sys.executable, 'src/p3_train.py', '--run-id', run_id, '--device', str(device)]
        if resume:
            child.append('--resume')
    command = [sys.executable, 'src/p3_attempt.py', '--attempt-id', attempt, '--run-id', run_id,
        '--device', str(device), '--owner-pid', str(os.getpid()), '--owner-ticks', process_ticks(os.getpid()),
        '--kind', kind, '--', *child]
    guardian = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONUNBUFFERED='1'), start_new_session=True)
    return {'guardian': guardian, 'log': log, 'attempt_id': attempt, 'run_id': run_id, 'device': device}


def baseline():
    return read(ROOT / 'reports/P3_proposal.json')['selected_timing_basis']


def ensure_plan(runs):
    return read(STAGE / 'run_manifest.json')


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
            'epoch2_3_fit_seconds': t, 'P2_selected_complete_transaction_baseline_seconds': ref,
            'slowdown_ratio': t / ref, 'timing_sha256': [file_hash(p) for p in paths],
            'data_wait_seconds_mean': statistics.mean(x['data_wait_seconds'] for x in timings),
            'above_10_percent': t > 1.1 * ref}
        observed.append(entry)
        values[run['configuration_id']].append(t)
        if entry['above_10_percent']:
            flags.append(entry['run_id'])
    estimate = {name: statistics.median(values[name]) if values[name] else b['epoch_seconds']
                for name, b in base.items()}
    write(STAGE / 'admission_estimates.json', {'epoch_seconds': estimate,
        'basis': 'median P3 complete epoch2/3 transaction timing when available; otherwise frozen P2 selected timing without multiplying contention again'})
    value = {'observations': observed, 'above_10_percent_run_ids': flags,
             'threshold': 1.1, 'test_accuracy_used': False}
    write(STAGE / 'cost_calibration.json', value)
    return estimate, flags


def forecast(runs, state, estimates, budget):
    base = baseline()
    remaining = 0.0
    for run in runs:
        item = state[run['run_id']]
        if item['status'] in ['completed', 'numerical_failure', 'oom', 'stalled', 'invalid_protocol']:
            continue
        epochs = item.get('completed_epochs', 0)
        remaining += (50 - epochs) * estimates[run['configuration_id']]
        if epochs == 0:
            remaining += base[run['configuration_id']]['startup_terminal_first_epoch_excess_seconds']
    proposal = read(ROOT / 'reports/P3_proposal.json')
    reserved = (require_authorization(False)['test_compute_reserve_gpu_seconds'] + 2.5 * 3600
                + proposal['P3_extra_prefix_save_gpu_hours_upper_scenario'] * 3600)
    used = budget.used(budget.snapshot())
    protected_total = used + 1.25 * (remaining + reserved)
    value = {'actual_gpu_seconds_including_active': used, 'remaining_estimated_gpu_seconds': remaining,
        'projected_total_gpu_hours': (used + remaining) / 3600, 'cap_gpu_hours': 160,
        'partial_uncommitted_epoch_forecast': 'conservative full epoch remains; actual ledger never double counts',
        'held_test_reserve_gpu_seconds': require_authorization(False)['test_compute_reserve_gpu_seconds'],
        'forecast_policy': 'actual_spent_plus_125_percent_of_remaining_and_reserves',
        'legacy_reinflated_forecast_gpu_hours': (used + remaining + reserved) * 1.25 / 3600,
        'forecast_with_reserves_and_margin_gpu_hours': protected_total / 3600,
        'forecast_exceeds_cap': protected_total + EXIT_RESERVE_SECONDS > CAP_SECONDS}
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
    storage.require(4 * 2**30)
    write(STAGE / 'scheduler.json', {'pid': os.getpid(), 'start_ticks': process_ticks(os.getpid()),
        'execution_commit': code_commit(), 'phase': 'P3', 'acceptance_only': args.acceptance_only})
    active = {}
    last_storage_log = 0.0
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
        for device in [0, 1]:
            task = launch('P3-gpu-acceptance-' + str(device), device, 'gpu_acceptance')
            task['guardian'].wait(timeout=300)
            task['log'].close()
            assert task['guardian'].returncode == 0
        results = [read(STAGE / ('gpu_acceptance_' + str(d) + '.json')) for d in [0, 1]]
        assert all(x['status'] == 'passed' and x['execution_commit'] == code_commit() for x in results)
        write(STAGE / 'gpu_acceptance.json', {'status': 'passed', 'devices': results,
            'execution_commit': code_commit(), 'execution_identity': execution_identity(),
            'all_GPU_acceptance_charged_to_P3': True, 'test_access': False})
        write(STAGE / 'acceptance.json', {'status': 'passed', 'six_gates_passed': True,
            'execution_commit': code_commit(), 'execution_identity': execution_identity(),
            'authorization_sha256': file_hash(STAGE / 'authorization.json'),
            'cpu_receipt_sha256': file_hash(STAGE / 'cpu_acceptance.json'),
            'gpu_receipt_sha256': file_hash(STAGE / 'gpu_acceptance.json'),
            'entry_preflight_sha256': file_hash(STAGE / 'entry_preflight.json'),
            'six_gates': ['frozen_inputs_exact_weights', '80_unique_fresh_50_epoch_five_seed_runs',
                'fault_injection_and_two_GPU_restore', 'manual_200GiB_and_160GPUhour_hard_limits',
                'timing_only_B_and_protected_immutable_prefixes', 'fail_closed_independently_signed_test_gate'],
            'P3_C_authorized': False})
        write(STAGE / 'dispatch_authorization.json', {'P3_B_authorized': True,
            'acceptance_sha256': file_hash(STAGE / 'acceptance.json'), 'centralized_test_authorized': False})
        return 0
    runs = candidates()
    plan = ensure_plan(runs)
    priority = plan['first_calibration_runs']
    assignments = {x['run_id']: x['physical_gpu_index'] for x in plan['runs']}
    hold = None
    try:
        while True:
            for device, task in list(active.items()):
                if task['guardian'].poll() is None:
                    # 进度长期无更新属于 stalled；与数值失败、OOM 和普通基础设施中断分别记账。
                    progress_path = ROOT / 'runs' / task['run_id'] / 'progress.json'
                    item = budget.snapshot()['attempts'].get(task['attempt_id'])
                    last = progress_path.stat().st_mtime if progress_path.exists() else (item['started_unix'] if item else time.time())
                    if item:
                        last = max(last, item['started_unix'])  # 恢复初始化不能沿用两天前的心跳判定 stalled。
                    limit = max(600, 5 * baseline()[next(r['configuration_id'] for r in runs if r['run_id'] == task['run_id'])]['epoch_seconds'])
                    if time.time() - last > limit:
                        task['stalled'] = True
                        hold = 'stalled_requires_investigation:' + task['run_id']
                        budget.stop(hold)
                    continue
                task['log'].close()
                budget.reconcile_dead_guardian(task['attempt_id'])
                item = read(ROOT / 'runs' / task['run_id'] / 'summary.json') if (ROOT / 'runs' / task['run_id'] / 'summary.json').exists() else {}
                if item.get('status') == 'running' or not item:
                    # SIGKILL 不依赖子 finally；父账本已结算，保留最后事务作为恢复边界。
                    item.update(run_id=task['run_id'], phase='P3', status='infrastructure_interrupted',
                                failure={'reason': 'child_exited_without_terminal_summary', 'attempt_id': task['attempt_id']})
                    write(ROOT / 'runs' / task['run_id'] / 'summary.json', item)
                if task.get('stalled'):
                    item.update(status='stalled', failure={'reason': 'training_heartbeat_timeout', 'attempt_id': task['attempt_id']})
                    write(ROOT / 'runs' / task['run_id'] / 'summary.json', item)
                if item['status'] in ['invalid_protocol', 'oom']:
                    hold = 'technical_failure_requires_investigation:' + task['run_id']
                attempts = [a for a in budget.snapshot()['attempts'].values() if a['run_id'] == task['run_id']]
                if item['status'] == 'infrastructure_interrupted':
                    hold = 'infrastructure_interruption_requires_investigation:' + task['run_id']
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
                hold = 'projected_total_with_reserves_exceeds_160_gpu_hours'
            try:
                resources = storage.require(4 * 2**30)
                write(STAGE / 'storage_latest.json', resources)
                if time.monotonic() - last_storage_log >= 60:
                    with (STAGE / 'storage_ledger.jsonl').open('ab') as log:
                        from tlcls.p3_core import canonical
                        log.write(canonical(dict(resources, unix=time.time())) + b'\n')
                    last_storage_log = time.monotonic()
            except RuntimeError:
                hold = 'storage_allocation_or_exit_space_insufficient'
                budget.stop(hold)
            if hold and active:
                budget.stop(hold)  # 发现异常即共同 drain，不让其它任务继续耗费剩余额度。
            if budget.snapshot()['stop_request'] or interrupted:
                hold = hold or budget.snapshot()['stop_request']['reason']
            all_terminal = all(x['status'] in ['completed', 'numerical_failure', 'oom', 'stalled', 'invalid_protocol'] for x in state.values())
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
                        -50 * estimates[r['configuration_id']], r['run_id']))
                    if waiting:
                        r = waiting[0]
                        snapshot = budget.snapshot()
                        needed = 1.15 * estimates[r['configuration_id']] * (len(active) + 1) + EXIT_RESERVE_SECONDS
                        if budget.used(snapshot) + needed >= budget.spending_limit:
                            hold = 'insufficient_budget_for_next_legal_epoch_boundary'
                            break
                        active[device] = launch(r['run_id'], device, 'candidate',
                            resume=(ROOT / 'runs' / r['run_id'] / 'scientific.json').exists())
            # 进程内安全监督持续执行；桌面端只按 30 分钟轮询。
            time.sleep(3)
    except BaseException as error:
        # 保留最初异常类别和 traceback，不能让 finally 的通用退出原因掩盖根因。
        import traceback
        hold = 'scheduler_exception:' + type(error).__name__
        write(STAGE / 'scheduler_failure.json', {'reason': hold, 'traceback': traceback.format_exc(),
            'execution_commit': code_commit(), 'unix': time.time(), 'automatic_restart_performed': False})
    finally:
        if active:
            request = budget.stop(hold or 'operator_technical_stop')
            # 所有守护进程共享同一个期限；不逐个等待完整 grace，避免两卡超预算。
            while active and time.monotonic() < request['drain_deadline_monotonic'] + 3:
                for device, task in list(active.items()):
                    if task['guardian'].poll() is not None:
                        task['log'].close()
                        del active[device]
                time.sleep(.1)
    write(STAGE / 'stage_status.json', {'status': 'paused_review_required' if hold else 'P3_runs_terminal',
        'reason': hold, 'execution_commit': code_commit(), 'P3_B_training_authorized': True, 'P3_C_authorized': False,
        'centralized_test_authorized': False, 'gpu_hours': budget.used(budget.snapshot()) / 3600})
    subprocess.run([sys.executable, 'src/p3_report.py'], cwd=ROOT, check=True)
    return 2 if hold else 0


if __name__ == '__main__':
    raise SystemExit(main())
