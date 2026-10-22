"""服务器低频定时检查：只读实验状态，30 分钟一次，不启动或重启任何训练。"""

import argparse
from collections import Counter
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path('/root/user_data/tl_cls_v1')
sys.path.insert(0, str(ROOT / 'src'))
from tlcls.p3_core import STAGE, candidates, file_hash, process_alive, process_ticks, read, write
from tlcls.p3_budget import Storage


def check():
    ledger = read(STAGE / 'gpu_ledger.json')
    authorization = read(STAGE / 'authorization.json')
    scheduler = read(STAGE / 'scheduler.json')
    alive = process_alive(scheduler['pid'], scheduler['start_ticks'])
    counts = Counter()
    for config in candidates():
        path = ROOT / 'runs' / config['run_id'] / 'summary.json'
        counts[read(path)['status'] if path.exists() else 'planned'] += 1
    now = time.monotonic()
    seconds = sum(a['charged_gpu_seconds'] if a['status'] == 'finalized' else
                  max(0, now - a['started_monotonic']) for a in ledger['attempts'].values())
    progress = read(STAGE / 'progress.json') if (STAGE / 'progress.json').exists() else {}
    storage = Storage(authorization['confirmed_project_environment_allocation_bytes']).snapshot()
    reason = None
    if authorization['centralized_test_authorized'] or authorization['P3_C_authorized']:
        reason = 'unexpected_phase_authorization'
    elif seconds > 160 * 3600:
        reason = 'budget_cap_exceeded'
    elif not alive and sum(counts[k] for k in ['completed','numerical_failure','oom','stalled','invalid_protocol']) == 80 and all(a['status']=='finalized' for a in ledger['attempts'].values()):
        reason = 'P3_finished_ready_for_evidence_recovery'
    elif not alive:
        reason = 'scheduler_exited_before_all_candidates_terminal'
    elif progress.get('reason'):
        reason = progress['reason']
    elif storage['warning_175_gib']:
        reason = 'storage_warning'
    value = {'checked_unix': time.time(), 'interval_seconds': 1800, 'counts': dict(counts),
        'scheduler_alive': alive, 'gpu_hours': seconds / 3600, 'cap_gpu_hours': 160,
        'forecast': progress.get('forecast'), 'storage': storage, 'actionable_reason': reason,
        'training_restarted_by_monitor': False, 'training_started_by_monitor': False, 'test_evaluation_started': False}
    (STAGE / 'monitor').mkdir(exist_ok=True)
    write(STAGE / 'monitor/latest.json', value)
    # 正常状态只追加机器回执；没有邮件、聊天消息或其它对外发送。
    with (STAGE / 'monitor/checks.jsonl').open('a', encoding='utf-8') as file:
        file.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    directory = STAGE / 'monitor'
    directory.mkdir(exist_ok=True)
    lock = (directory / 'monitor.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write(directory / 'registration.json', {'pid': os.getpid(), 'start_ticks': process_ticks(os.getpid()),
        'interval_seconds': 1800, 'script_sha256': file_hash(Path(__file__)),
        'execution': 'server_background_periodic_check', 'once': args.once,
        'created_unix': time.time(), 'max_runtime_hours': 240})
    deadline = time.monotonic() + 240 * 3600
    while time.monotonic() < deadline:
        try:
            value = check()
        except Exception as error:
            # 不输出异常内容；只记录类别，避免外部信息混入运维回执。
            value = {'checked_unix': time.time(), 'actionable_reason': 'monitor_error',
                     'error_type': type(error).__name__, 'training_restarted_by_monitor': False}
        if value.get('actionable_reason'):
            write(directory / 'attention.json', value)
            print(json.dumps(value, ensure_ascii=False), flush=True)
            return 0 if value['actionable_reason'] == 'P3_finished_ready_for_evidence_recovery' else 2
        if args.once:
            print(json.dumps({'status': 'normal', 'gpu_hours': value['gpu_hours'], 'counts': value['counts']}))
            return 0
        # 本进程只占 CPU 休眠，不维持前台或 SSH 会话；完成/异常后停止自身。
        time.sleep(1800)
    write(directory / 'attention.json', {'actionable_reason': '240_hour_monitor_timeout', 'checked_unix': time.time()})
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
