"""P2 epoch 事务：不可变代次 + 单一提交指针，日志是可幂等重建的派生视图。"""

import os
from pathlib import Path
import re
import time
import uuid

from .p2_core import atomic_bytes, canonical, file_hash, fsync_directory, read, write


def committed_summary(current, rows, reference):
    """终轮提交后即使 summary 丢失，也能仅由完整提交恢复全部终结字段。"""
    assert current and rows and len(rows) == current['completed_epoch']
    return {'completed_epochs': current['completed_epoch'], 'selected_epoch': current['best_epoch'],
            'best_val_top1_percent': max(r['validation']['top1_percent'] for r in rows),
            'optimizer_updates': rows[-1]['cumulative_optimizer_updates'], 'commit': reference,
            'best_checkpoint': current['best'], 'last_checkpoint': current['last']}


class EpochTransaction:
    def __init__(self, directory, scientific_sha):
        self.root = Path(directory)
        self.scientific_sha = scientific_sha
        (self.root / 'checkpoints').mkdir(parents=True, exist_ok=True)
        (self.root / 'commits').mkdir(exist_ok=True)

    def path(self, relative):
        path = (self.root / relative).resolve()
        assert path.is_relative_to(self.root.resolve()), '事务路径越界'
        return path

    def artifact(self, kind, epoch, writer):
        name = f'checkpoints/{kind}.e{epoch:03d}.{uuid.uuid4().hex}.pt'
        target = self.path(name)
        temporary = target.with_suffix('.pt.tmp')
        with temporary.open('xb') as file:
            writer(file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, target)
        fsync_directory(target.parent)
        return {'path': name, 'sha256': file_hash(target), 'size_bytes': target.stat().st_size}

    def _manifest(self, reference):
        path = self.path(reference['path'])
        assert file_hash(path) == reference['sha256'], '提交清单 hash 不一致'
        value = read(path)
        assert value['scientific_config_sha256'] == self.scientific_sha, '科学配置不一致，拒绝恢复'
        return value

    def load(self, rebuild=True, verify_artifacts=True):
        pointer = self.root / 'current.json'
        if not pointer.exists():
            return None, []
        reference = read(pointer)
        latest = self._manifest(reference)
        rows, manifests = [], []
        current = latest
        while current:
            rows.append(current['epoch_row'])
            manifests.append(current)
            current = self._manifest(current['previous_commit']) if current['previous_commit'] else None
        rows.reverse()
        manifests.reverse()
        assert [r['epoch'] for r in rows] == list(range(1, latest['completed_epoch'] + 1))
        clocks = [m['cumulative_gpu_seconds_at_commit'] for m in manifests]
        assert all(b >= a for a, b in zip(clocks, clocks[1:])), '提交累计时间回退'
        if verify_artifacts:
            for kind in ['last', 'best']:
                item = latest[kind]
                path = self.path(item['path'])
                assert path.stat().st_size == item['size_bytes'] and file_hash(path) == item['sha256']
        if rebuild:
            # 日志在 current 发布后崩溃也可重建；重复恢复不会重复追加同一轮。
            atomic_bytes(self.root / 'epochs.jsonl', b''.join(canonical(r) + b'\n' for r in rows))
        return latest, rows

    def commit(self, epoch, row, last_writer, best_writer, best_epoch, gpu_seconds, fault=None):
        fault = fault or (lambda boundary: None)
        # 同一训练进程只使用自己的不可变代次；恢复入口已完整校验，逐轮无需重读上 GB 的旧文件。
        previous, rows = self.load(rebuild=False, verify_artifacts=False)
        assert epoch == (previous['completed_epoch'] + 1 if previous else 1)
        assert row['epoch'] == epoch
        assert best_epoch == (epoch if best_writer else previous['best_epoch']), 'best 代次与选模轮次不一致'
        started = time.monotonic()
        last = self.artifact('last', epoch, last_writer)
        fault('after_last')
        best = self.artifact('best', epoch, best_writer) if best_writer else previous['best']
        fault('after_best')
        row = dict(row, checkpoint_seconds=time.monotonic() - started)
        gpu_seconds = gpu_seconds() if callable(gpu_seconds) else gpu_seconds
        value = {'schema': 1, 'scientific_config_sha256': self.scientific_sha,
                 'completed_epoch': epoch, 'last': last, 'best': best, 'best_epoch': best_epoch,
                 'previous_commit': read(self.root / 'current.json') if previous else None,
                 'epoch_row': row, 'cumulative_gpu_seconds_at_commit': gpu_seconds}
        relative = f'commits/e{epoch:03d}.{uuid.uuid4().hex}.json'
        write(self.path(relative), value)
        reference = {'path': relative, 'sha256': file_hash(self.path(relative))}
        fault('after_manifest')
        write(self.root / 'current.json', reference)
        fault('after_pointer')
        atomic_bytes(self.root / 'epochs.jsonl', b''.join(canonical(r) + b'\n' for r in rows + [row]))
        fault('after_log')
        write(self.root / 'active_costs.json', {'commit': reference,
            'scientific_config_sha256': self.scientific_sha, 'completed_epoch': epoch,
            'cumulative_gpu_seconds': gpu_seconds})
        fault('after_sidecar')
        self.collect_superseded(value)
        return value, row

    def collect_superseded(self, current):
        keep = {current[k]['path'] for k in ['last', 'best']}
        journal = self.root / 'checkpoint_generation_history.jsonl'
        known = {}
        for manifest_path in (self.root / 'commits').glob('*.json'):
            manifest = read(manifest_path)
            for kind in ['last', 'best']:
                known[manifest[kind]['path']] = manifest[kind]
        # 只回收本事务创建且不再被当前提交引用的完整代次；不删除 P1、失败现场或其它文件。
        for path in sorted((self.root / 'checkpoints').glob('*.pt')):
            relative = str(path.relative_to(self.root)).replace('\\', '/')
            if relative in keep or not re.fullmatch(r'(last|best)\.e\d{3}\.[0-9a-f]{32}\.pt', path.name):
                continue
            descriptor = known.get(relative)
            # 已提交的不可变文件沿用生成时的 hash；未提交的孤立文件需现场计算。
            record = {'path': relative, 'sha256': descriptor['sha256'] if descriptor else file_hash(path),
                      'hash_source': 'original_immutable_generation_receipt' if descriptor else 'read_at_cleanup',
                      'size_bytes': path.stat().st_size,
                      'action': 'superseded_or_uncommitted_generation_removed_after_complete_commit'}
            with journal.open('ab') as file:
                file.write(canonical(record) + b'\n')
                file.flush()
                os.fsync(file.fileno())
            path.unlink()
        fsync_directory(self.root / 'checkpoints')

    def compatible_sidecar(self, minimum_gpu_seconds):
        path = self.root / 'active_costs.json'
        if not path.exists() or not (self.root / 'current.json').exists():
            return None
        value = read(path)
        if (value.get('commit') != read(self.root / 'current.json') or
                value.get('scientific_config_sha256') != self.scientific_sha or
                value.get('cumulative_gpu_seconds', -1) < minimum_gpu_seconds):
            return None
        return value
