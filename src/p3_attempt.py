"""独立守护一个 GPU attempt；调度器死亡后仍负责清理进程组并结算真实时间。"""

import argparse
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from tlcls.p3_budget import Budget
from tlcls.p3_core import (CAP_SECONDS, EXIT_RESERVE_SECONDS, ROOT, STAGE, process_alive,
                           require_authorization)


def group_alive(group):
    try:
        os.killpg(group, 0)
        return True
    except ProcessLookupError:
        return False


def signal_group(group, sig):
    try:
        os.killpg(group, sig)
    except ProcessLookupError:
        pass


def parent_death_guard():
    # 额外防御守护进程本身被杀；正常的父调度器退出由守护循环统一清理整个训练组。
    ctypes.CDLL(None).prctl(1, signal.SIGKILL)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attempt-id', required=True)
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--device', type=int, required=True)
    parser.add_argument('--owner-pid', type=int, required=True)
    parser.add_argument('--owner-ticks', required=True)
    parser.add_argument('--kind', choices=['candidate', 'gpu_acceptance', 'cpu_fixture'], required=True)
    parser.add_argument('--fixture-ledger')
    parser.add_argument('--fixture-cap', type=float)
    parser.add_argument('--fixture-grace', type=float, default=20)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.kind == 'cpu_fixture':
        assert os.environ.get('TLCLS_CPU_TEST') == '1' and args.fixture_ledger
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
        ledger = Budget(args.fixture_ledger, args.fixture_cap)
    else:
        require_authorization(require_acceptance=args.kind == 'candidate')
        ledger = Budget()
    assert args.command and args.command[0] == '--'
    grace = args.fixture_grace if args.kind == 'cpu_fixture' else 20
    ledger.register(args.attempt_id, args.run_id, args.device, args.kind, args.owner_pid, args.owner_ticks)
    process = None
    reason, sent_term = 'completed', False
    local_stop = False
    def interrupted(signum, frame):
        nonlocal local_stop
        local_stop = True
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        environment = dict(os.environ, TLCLS_ATTEMPT_ID=args.attempt_id, PYTHONUNBUFFERED='1',
                           TLCLS_PARENT_LEDGER=str(ledger.path))
        process = subprocess.Popen(args.command[1:], cwd=ROOT, env=environment,
                                   start_new_session=True, preexec_fn=parent_death_guard)
        ledger.update(args.attempt_id, child_pid=process.pid, child_pgid=process.pid)
        import psutil
        while True:
            snapshot = ledger.snapshot()
            if not process_alive(args.owner_pid, args.owner_ticks):
                ledger.stop('owner_process_terminated', grace)
            if local_stop:
                ledger.stop('operator_technical_stop', grace)
            if ledger.used(snapshot) >= ledger.spending_limit - EXIT_RESERVE_SECONDS:
                ledger.stop('shared_gpu_budget_hard_stop', grace)
            snapshot = ledger.snapshot()
            stop = snapshot['stop_request']
            if stop and not sent_term:
                reason, sent_term = stop['reason'], True
                # 先通知训练主进程，在边界退出；同时杀 workers 会由 SIGCHLD 打断任意模型代码。
                try:
                    os.kill(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            if stop and time.monotonic() >= stop['drain_deadline_monotonic']:
                signal_group(process.pid, signal.SIGKILL)
            try:
                main_process = psutil.Process(process.pid)
                rss = main_process.memory_info().rss
                tree = rss + sum(p.memory_info().rss for p in main_process.children(recursive=True)
                                 if p.is_running())
                item = snapshot['attempts'][args.attempt_id]
                ledger.update(args.attempt_id,
                    main_process_peak_rss_bytes_sampled=max(rss, item['main_process_peak_rss_bytes_sampled']),
                    process_tree_peak_rss_bytes_sampled=max(tree, item['process_tree_peak_rss_bytes_sampled']),
                    samples=item['samples'] + 1, last_observed_monotonic=time.monotonic())
            except (psutil.Error, ProcessLookupError):
                pass
            if process.poll() is not None:
                # 先清理残余 DataLoader workers，再停止预算时钟；退出/drain 成本不丢失。
                signal_group(process.pid, signal.SIGKILL)
                break
            time.sleep(0.25)
        process.wait()
    except BaseException:
        reason = 'guardian_exception'
        if process is not None:
            signal_group(process.pid, signal.SIGKILL)
            process.wait()
        raise
    finally:
        ledger.finalize(args.attempt_id, process.returncode if process else -1, reason, time.monotonic())
    return process.returncode if process.returncode >= 0 else 128 - process.returncode


if __name__ == '__main__':
    raise SystemExit(main())
