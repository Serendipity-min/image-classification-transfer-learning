"""双卡共享的 P3 成本账本；独立 attempt 守护进程持续记账并执行统一退出期限。"""

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shutil
import signal
import stat as file_stat
import time

from .p3_core import CAP_SECONDS, EXIT_RESERVE_SECONDS, ROOT, STAGE, code_commit, process_alive, process_ticks, read, write


class Budget:
    def __init__(self, path=None, cap=CAP_SECONDS):
        self.path = Path(path or STAGE / 'gpu_ledger.json')
        self.cap = cap
        # 测试预留不是训练额度；独立台账仍显示完整 160h，上限内额外保留尚未获准的测试栏位。
        reserved = read(STAGE / 'authorization.json')['test_compute_reserve_gpu_seconds'] if path is None else 0
        self.spending_limit = self.cap - reserved
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def edit(self):
        # 每次锁内重新读取，不能将各 GPU 的旧快照相互覆盖。
        with self.path.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            value = read(self.path) if self.path.exists() else {
                'schema': 1, 'phase': 'P3', 'cap_gpu_seconds': self.cap, 'attempts': {},
                'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                'stop_request': None, 'events': [], 'P3_authorized': True,
                'centralized_test_authorized': False}
            assert value['cap_gpu_seconds'] == self.cap
            assert value['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip(), '系统重启后单调时钟身份改变，须人工核算未结算占用'
            try:
                yield value
            finally:
                value['updated_monotonic'] = time.monotonic()
                value['gpu_seconds_finalized'] = sum(a.get('charged_gpu_seconds', 0) for a in value['attempts'].values()
                                                       if a['status'] == 'finalized')
                write(self.path, value)

    def snapshot(self):
        with self.edit() as value:
            return dict(value, attempts={k: dict(v) for k, v in value['attempts'].items()})

    @staticmethod
    def used(value, now=None):
        now = time.monotonic() if now is None else now
        return sum(a['charged_gpu_seconds'] if a['status'] == 'finalized' else
                   max(0, now - a['started_monotonic']) for a in value['attempts'].values())

    def run_seconds(self, run_id):
        value = self.snapshot()
        now = time.monotonic()
        return sum(a['charged_gpu_seconds'] if a['status'] == 'finalized' else
                   max(0, now - a['started_monotonic']) for a in value['attempts'].values() if a['run_id'] == run_id)

    def register(self, attempt_id, run_id, device, kind, owner_pid, owner_ticks):
        with self.edit() as value:
            assert not value['stop_request'], '阶段停止请求尚未解除'
            assert attempt_id not in value['attempts']
            assert self.used(value) + EXIT_RESERVE_SECONDS < self.spending_limit
            assert all(a['status'] == 'finalized' or a['device'] != device for a in value['attempts'].values())
            value['attempts'][attempt_id] = {'attempt_id': attempt_id, 'run_id': run_id, 'device': device,
                'kind': kind, 'owner_pid': owner_pid, 'owner_start_ticks': owner_ticks,
                'execution_commit': code_commit(),
                'guardian_pid': os.getpid(), 'guardian_start_ticks': process_ticks(os.getpid()),
                'status': 'running', 'started_monotonic': time.monotonic(),
                'started_unix': time.time(), 'child_pid': None, 'child_pgid': None,
                'main_process_peak_rss_bytes_sampled': 0, 'process_tree_peak_rss_bytes_sampled': 0,
                'samples': 0}

    def update(self, attempt_id, **fields):
        with self.edit() as value:
            value['attempts'][attempt_id].update(fields)

    def finalize(self, attempt_id, returncode, reason, ended_monotonic):
        with self.edit() as value:
            item = value['attempts'][attempt_id]
            # 幂等结算：重复监督/恢复不能再次增加同一 attempt 的时间。
            if item['status'] != 'finalized':
                item.update(status='finalized', returncode=returncode, termination_reason=reason,
                            ended_monotonic=ended_monotonic,
                            charged_gpu_seconds=max(0, ended_monotonic - item['started_monotonic']))

    def stop(self, reason, grace=20):
        with self.edit() as value:
            if value['stop_request'] is None:
                value['stop_request'] = {'reason': reason, 'requested_monotonic': time.monotonic(),
                                         'drain_deadline_monotonic': time.monotonic() + grace}
                value['events'].append(dict(value['stop_request']))
            return value['stop_request']

    def clear_technical_stop(self):
        with self.edit() as value:
            assert all(a['status'] == 'finalized' for a in value['attempts'].values())
            assert value['stop_request']['reason'] in ['owner_process_terminated', 'operator_technical_stop']
            value['events'].append({'reason': 'technical_resume_same_cap', 'monotonic': time.monotonic()})
            value['stop_request'] = None

    def reconcile_dead_guardian(self, attempt_id):
        item = self.snapshot()['attempts'][attempt_id]
        if item['status'] == 'finalized':
            return
        assert not process_alive(item['guardian_pid'], item.get('guardian_start_ticks'))
        if item.get('child_pgid'):
            try:
                os.killpg(item['child_pgid'], signal.SIGKILL)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 2
        while item.get('child_pid') and process_alive(item['child_pid']) and time.monotonic() < deadline:
            time.sleep(.05)
        assert not item.get('child_pid') or not process_alive(item['child_pid'])
        # 记录至实际确认退出的采样时刻，保守包含最后一次监督间隔，不遗漏失联片段。
        self.finalize(attempt_id, -signal.SIGKILL, 'dead_guardian_reconciled_until_exit_observed', time.monotonic())


class Storage:
    def __init__(self, allocation_bytes=200 * 2**30, environment=None):
        self.allocation = allocation_bytes
        self.environment = Path(environment or '/root/miniconda3/envs/tl_cls_v1')

    @staticmethod
    def size(path):
        seen, total = set(), 0
        directory = Path(path)
        if not directory.is_dir():
            raise FileNotFoundError('存储统计根目录不存在')
        for item in directory.rglob('*'):
            try:
                # 一个 lstat 同时取得类型和大小，避免 is_file 与 stat 之间的原子重命名竞态。
                metadata = item.lstat()
            except FileNotFoundError:
                # 临时 JSON 发布、已提交旧代次回收会正常移除目录项；只忽略 ENOENT。
                continue
            if not file_stat.S_ISREG(metadata.st_mode):
                continue
            key = (metadata.st_dev, metadata.st_ino)
            if key not in seen:
                seen.add(key)
                total += metadata.st_size
        return total

    def snapshot(self):
        project = self.size(ROOT)
        environment = self.size(self.environment) if self.environment.exists() else 0
        physical = shutil.disk_usage(ROOT).free
        used = project + environment
        return {'project_bytes': project, 'environment_non_symlink_bytes': environment,
                'combined_bytes': used, 'confirmed_allocation_bytes': self.allocation,
                'filesystem_free_bytes': physical, 'allocation_remaining_bytes': self.allocation - used,
                'warning_175_gib': used >= 175 * 2**30, 'reserve_bytes': 10 * 2**30,
                'measurement': 'logical file bytes, symlinks excluded, hardlinks deduplicated within each root',
                'allocation_source': 'explicit_user_P3_contract_resource_confirmation_2026-10-22'}

    def require(self, next_write_bytes=0):
        value = self.snapshot()
        needed = value['reserve_bytes'] + next_write_bytes
        if min(value['filesystem_free_bytes'], value['allocation_remaining_bytes']) < needed:
            raise RuntimeError('storage_exit_reserve_or_allocation_insufficient')
        return value
