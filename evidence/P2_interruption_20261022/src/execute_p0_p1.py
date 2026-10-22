"""首批 4 GPU-hours 的父进程预算门与阶段调度，P2/P3 始终关闭。"""

import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import psutil

from tlcls.common import ROOT, atomic_json, load_json, provenance


CAP = 4 * 3600
LEDGER = ROOT / 'records/gpu_budget.json'
LEDGER.parent.mkdir(exist_ok=True)


def read_ledger():
    if LEDGER.exists():
        return load_json(LEDGER)
    return {'protocol_id': 'TL-CLS-20261020-v1.0', 'cap_gpu_seconds': CAP,
            'provenance': provenance(), 'entries': [], 'gpu_seconds': 0.0,
            'status': 'running', 'P2_P3_authorized': False}


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu,temperature.gpu,power.draw,clocks.sm',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True)
    return result.stdout.strip()


def run_group(jobs, label, gpu_work=True):
    ledger = read_ledger()
    remaining = CAP - ledger['gpu_seconds']
    if gpu_work and remaining < 90:
        return False
    handles = []
    processes = []
    started = time.monotonic()
    for args in jobs:
        path = ROOT / 'records' / (label + f'_{len(processes)}.log')
        file = path.open('a', encoding='utf-8')
        handles.append(file)
        process = subprocess.Popen([sys.executable, *args], cwd=ROOT, stdout=file, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, PYTHONUNBUFFERED='1',
                                            TLCLS_PROCESS_STARTED_MONOTONIC=str(time.monotonic())),
                                   start_new_session=True)
        processes.append((process, time.monotonic()))
    accounted = {}
    timed_out = False
    external_peaks = {}
    cpu_rss_peaks = {}
    try:
        while True:
            now = time.monotonic()
            for process, process_started in processes:
                if process.poll() is not None and process.pid not in accounted:
                    accounted[process.pid] = now - process_started
            used = sum(accounted.get(process.pid, now - process_started) for process, process_started in processes) if gpu_work else 0
            if gpu_work and used + ledger['gpu_seconds'] >= CAP - 15:
                timed_out = True
                for process, _ in processes:
                    if process.poll() is None:
                        os.killpg(process.pid, signal.SIGTERM)
                break
            snapshot = gpu_snapshot() if gpu_work else None
            if snapshot:
                for line in snapshot.splitlines():
                    fields = [field.strip() for field in line.split(',')]
                    if len(fields) >= 3 and fields[2].isdigit():
                        external_peaks[fields[0]] = max(external_peaks.get(fields[0], 0), int(fields[2]))
            for process, _ in processes:
                try:
                    cpu_rss_peaks[str(process.pid)] = max(cpu_rss_peaks.get(str(process.pid), 0),
                                                          psutil.Process(process.pid).memory_info().rss)
                except psutil.Error:
                    pass
            atomic_json(ROOT / 'records/execution_progress.json',
                        {'label': label, 'phase': 'P0' if label.startswith('P0') else 'P1',
                         'gpu_seconds_completed': ledger['gpu_seconds'], 'gpu_seconds_active_segment': used,
                         'remaining_gpu_seconds': CAP - ledger['gpu_seconds'] - used,
                         'pids': [process.pid for process, _ in processes if process.poll() is None],
                         'gpu_snapshot': snapshot,
                         'updated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
            if all(process.poll() is not None for process, _ in processes):
                break
            time.sleep(2)
        for process, _ in processes:
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    finally:
        now = time.monotonic()
        gpu_seconds = sum(accounted.get(p.pid, now - tick) for p, tick in processes) if gpu_work else 0
        ledger['entries'].append({'label': label, 'gpu_seconds': gpu_seconds,
                                  'process_count': len(jobs), 'wall_seconds': now - started,
                                  'returncodes': [p.returncode for p, _ in processes],
                                  'hard_budget_stop': timed_out,
                                  'external_gpu_peak_memory_mib_sampled_2s': external_peaks,
                                  'cpu_main_process_peak_rss_bytes_sampled_2s': cpu_rss_peaks,
                                  'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        ledger['gpu_seconds'] += gpu_seconds
        ledger['gpu_hours'] = ledger['gpu_seconds'] / 3600
        ledger['status'] = 'budget_exhausted' if timed_out else 'running'
        atomic_json(LEDGER, ledger)
        if timed_out and label.startswith('P1-'):
            summary_path = ROOT / 'runs' / label / 'summary.json'
            if summary_path.exists():
                summary = load_json(summary_path)
                summary['status'] = 'infrastructure_interrupted'
                summary['failure'] = {'reason': 'parent_4_gpu_hour_hard_budget_stop',
                                      'checkpoint_resume_boundary': 'last_completed_saved_epoch'}
                atomic_json(summary_path, summary)
        for file in handles:
            file.close()
    return not timed_out and all(p.returncode == 0 for p, _ in processes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-p0', action='store_true')
    parser.add_argument('--reuse-data', action='store_true')
    args = parser.parse_args()
    # 防止重复启动父调度器，让每一段 GPU 成本只归属一个预算账本。
    import fcntl
    lock = (ROOT / 'records/execution.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.skip_p0:
        if not args.reuse_data:
            if not run_group([['src/prepare_data.py']], 'P0_data', gpu_work=False):
                return 2
        else:
            assert load_json(ROOT / 'records/data_integrity.json')['datasets']
            # 数据划分文件继续使用原文件与原 hash；仅复核改动后的执行代码。
            import shutil
            shutil.copyfile(ROOT / 'records/implementation_checks.json', ROOT / 'records/implementation_checks_initial.json')
        if not run_group([['src/acceptance.py', '--device', '0']], 'P0_acceptance'):
            return 2
    checks = load_json(ROOT / 'records/implementation_checks.json')
    assert checks['status'] == 'passed' and checks['provenance']['code_commit'] == provenance()['code_commit']
    runs = [r for r in load_json(ROOT / 'configs/run_matrix.json')['planned_runs'] if r['phase'] == 'P1']
    # 统计已完成配置的真实设备，在每种策略内部补齐较少使用的卡，支持续跑后的均衡。
    method_gpu_counts = {method: [0, 0] for method in ['scratch', 'full_ft', 'linear_probe', 'adapter']}
    for config in runs:
        path = ROOT / 'runs' / config['run_id'] / 'summary.json'
        if path.exists():
            summary = load_json(path)
            if summary['status'] == 'completed':
                method_gpu_counts[config['method']][summary['hardware']['index']] += 1
    for index, config in enumerate(runs):
        summary_path = ROOT / 'runs' / config['run_id'] / 'summary.json'
        if summary_path.exists() and load_json(summary_path)['status'] == 'completed':
            continue
        remaining = CAP - read_ledger()['gpu_seconds']
        if remaining < 180:
            break
        counts = method_gpu_counts[config['method']]
        device = (index % 2) if counts[0] == counts[1] else (0 if counts[0] < counts[1] else 1)
        job = ['src/train.py', '--run-id', config['run_id'], '--device', str(device),
               '--max-active-seconds', str(remaining - 45)]
        if summary_path.exists() and (summary_path.parent / 'last.pt').exists():
            job.append('--resume')
        if not run_group([job], config['run_id']):
            # 技术失败保存证据并返回诊断；不在后台循环重跑消耗预算。
            break
        method_gpu_counts[config['method']][device] += 1
    complete = all((ROOT / 'runs' / r['run_id'] / 'summary.json').exists()
                   and load_json(ROOT / 'runs' / r['run_id'] / 'summary.json')['status'] == 'completed' for r in runs)
    if complete and CAP - read_ledger()['gpu_seconds'] > 600:
        representative = ['P1-cifar100-r50-full_ft-s314159', 'P1-svhn-vitb16-adapter-s314159']
        for index, run_id in enumerate(representative):
            if not run_group([['src/benchmark.py', '--run-id', run_id, '--device', str(index), '--label', 'solo']],
                             f'P1_benchmark_solo_{index}'):
                break
        if CAP - read_ledger()['gpu_seconds'] > 300:
            barrier = time.time() + 45
            jobs = [['src/benchmark.py', '--run-id', run_id, '--device', str(index),
                     '--label', 'concurrent', '--barrier-unix', str(barrier)]
                    for index, run_id in enumerate(representative)]
            run_group(jobs, 'P1_benchmark_concurrent')
    ledger = read_ledger()
    ledger['status'] = 'P1_completed' if complete else 'P1_incomplete_review_required'
    atomic_json(LEDGER, ledger)
    subprocess.run([sys.executable, 'src/report.py'], cwd=ROOT, check=True)
    return 0 if complete else 2


if __name__ == '__main__':
    raise SystemExit(main())
