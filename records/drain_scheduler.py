"""暂停旧父调度器并让正在训练的子进程自然完成，连续核算剩余 GPU 预算。"""

import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import time


ROOT = Path('/root/user_data/tl_cls_v1')
CAP = 14400


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temporary, path)


def alive(pid):
    path = Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split(') ', 1)[1].split()[0] != 'Z'


if __name__ == '__main__':
    progress = load(ROOT / 'records/execution_progress.json')
    parent = load(ROOT / 'records/supervisor_pid.json')['pid']
    assert progress['label'].startswith('P1-') and len(progress['pids']) == 1
    child = progress['pids'][0]
    label = progress['label']
    # SIGINT 触发旧调度器的 finally，将截至暂停时刻的原始预算与监测峰值写完。
    os.kill(parent, signal.SIGINT)
    deadline = time.monotonic() + 15
    while alive(parent) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not alive(parent), '父调度器未停止，禁止启动第二个调度器'
    ledger_path = ROOT / 'records/gpu_budget.json'
    ledger = load(ledger_path)
    partial = ledger['entries'][-1]
    assert partial['label'] == label
    snapshot_wall = datetime.datetime.fromisoformat(partial['timestamp_utc']).timestamp()
    segment_start = time.monotonic() - max(0, time.time() - snapshot_wall)
    stopped_by_budget = False
    external = {}
    while alive(child):
        seconds = time.monotonic() - segment_start
        if ledger['gpu_seconds'] + seconds >= CAP - 20:
            os.killpg(child, signal.SIGTERM)
            stopped_by_budget = True
            break
        query = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used', '--format=csv,noheader,nounits'],
                                capture_output=True, text=True)
        for row in query.stdout.splitlines():
            fields = [field.strip() for field in row.split(',')]
            if len(fields) == 2 and fields[1].isdigit():
                external[fields[0]] = max(external.get(fields[0], 0), int(fields[1]))
        write(ROOT / 'records/execution_progress.json', {
            'label': label, 'phase': 'P1_draining_before_balanced_scheduling',
            'gpu_seconds_completed': ledger['gpu_seconds'], 'gpu_seconds_active_segment': seconds,
            'remaining_gpu_seconds': CAP - ledger['gpu_seconds'] - seconds,
            'pids': [child], 'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        time.sleep(0.5)
    if stopped_by_budget:
        time.sleep(5)
        if alive(child):
            os.killpg(child, signal.SIGKILL)
    seconds = time.monotonic() - segment_start
    summary = load(ROOT / 'runs' / label / 'summary.json')
    # 用连续观察的续段计时入账；保留旧父进程暂停前的已计费片段，避免丢失或重复。
    ledger['entries'].append({'label': label + '_drain_segment', 'gpu_seconds': seconds,
        'wall_seconds': seconds, 'process_count': 1, 'returncodes': [0 if summary['status'] == 'completed' else 2],
        'external_gpu_peak_memory_mib_sampled_0_5s': external,
        'reason': 'scheduler_only_balance_correction_current_training_not_interrupted',
        'measurement_source': 'continuous_clock_after_parent_finally_snapshot',
        'hard_budget_stop': stopped_by_budget,
        'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
    partial['pause_reason'] = 'orchestration_pause_child_continued_to_completion'
    partial['returncodes_final_after_drain'] = [0 if summary['status'] == 'completed' else 2]
    ledger['gpu_seconds'] += seconds
    ledger['gpu_hours'] = ledger['gpu_seconds'] / 3600
    ledger['status'] = 'ready_for_balanced_scheduler_resume' if not stopped_by_budget else 'budget_exhausted'
    write(ledger_path, ledger)
    write(ROOT / 'records/scheduler_drain.json', {'run_id': label, 'status': summary['status'],
          'parent_stopped': True, 'training_restarted': False, 'gpu_hours': ledger['gpu_hours']})
    print(json.dumps({'drained': label, 'status': summary['status'], 'gpu_hours': ledger['gpu_hours']}), flush=True)
