"""50 轮不可变 epoch 事务；前缀只引用已完整提交且验真后的模型快照。"""
import time
import uuid
from .p2_transaction import EpochTransaction as BaseTransaction, committed_summary
from .p3_core import STAGE, atomic_bytes, canonical, file_hash, read, write


class EpochTransaction(BaseTransaction):
    def __init__(self, directory, scientific_sha, config=None, thresholds=None):
        super().__init__(directory, scientific_sha)
        self.config = config
        if thresholds is None:
            assert config is not None
            frozen = read(STAGE / 'time_budget_frozen.json')['groups']
            group = frozen[config['dataset'] + '-' + config['model_key']]
            thresholds = group['slice_seconds']
        self.thresholds = thresholds

    def prefixes(self):
        path = self.root / 'prefixes.json'
        if path.exists():
            value = read(path)
            assert value['scientific_config_sha256'] == self.scientific_sha
            assert [x['budget_seconds'] for x in value['items']] == self.thresholds
            return value
        return {'schema': 'P3_prefix_v1', 'scientific_config_sha256': self.scientific_sha,
                'items': [{'budget_fraction': f, 'budget_seconds': t, 'status': 'pending',
                           'evaluation_role': 'budget_prefix', 'checkpoint': None}
                          for f, t in zip([.25, .5, 1.0], self.thresholds)]}

    def load(self, rebuild=True, verify_artifacts=True):
        latest, rows = super().load(rebuild, verify_artifacts)
        if latest and verify_artifacts:
            # 切片锁定后独立于 best/last；损坏时拒绝恢复，绝不静默换一个模型。
            for item in self.prefixes()['items']:
                if item['checkpoint']:
                    artifact = item['checkpoint']
                    assert file_hash(self.path(artifact['path'])) == artifact['sha256']
        if latest and rebuild:
            # 指针已提交、切片清单尚未发布的崩溃，只能重放原有时钟收据，不能用恢复时间倒填。
            receipt = self.root / 'eligible' / f"epoch_{latest['completed_epoch']:03d}.json"
            if receipt.exists():
                completion = read(receipt)
                assert completion['commit'] == read(self.root / 'current.json')
                prefixes = self.prefixes()
                for item in prefixes['items']:
                    if item['status'] in ['available', 'unavailable']: continue
                    if completion['transaction_completed_gpu_seconds'] <= item['budget_seconds']:
                        item.update(checkpoint=completion['checkpoint'], completed_epoch=latest['completed_epoch'],
                                    completion=completion, status='pending_candidate')
                    else:
                        item['status'] = 'available' if item['checkpoint'] else 'unavailable'
                write(self.root / 'prefixes.json', prefixes)
        return latest, rows

    def commit(self, epoch, row, last_writer, snapshot_writer, best_epoch, gpu_seconds,
               fault=None, improved=True):
        fault = fault or (lambda _: None)
        previous, rows = self.load(rebuild=False, verify_artifacts=False)
        assert epoch == (previous['completed_epoch'] + 1 if previous else 1) and epoch <= 50
        assert row['epoch'] == epoch
        assert best_epoch == (epoch if improved else previous['best_epoch'])
        started = time.monotonic()
        last = self.artifact('last', epoch, last_writer)
        fault('after_last')
        # 每轮一个 model-only snapshot，既可作 best，也可作时间切片；不重复序列化三份模型。
        snapshot = self.artifact('snapshot', epoch, snapshot_writer)
        best = snapshot if improved else previous['best']
        fault('after_best')
        clock = gpu_seconds() if callable(gpu_seconds) else gpu_seconds
        if previous:
            assert clock >= previous['cumulative_gpu_seconds_at_commit'], '费用不得回退'
        row = dict(row, checkpoint_seconds=time.monotonic() - started)
        value = {'schema': 'P3_transaction_v1', 'scientific_config_sha256': self.scientific_sha,
                 'completed_epoch': epoch, 'last': last, 'best': best, 'snapshot': snapshot,
                 'best_epoch': best_epoch, 'previous_commit': read(self.root / 'current.json') if previous else None,
                 'epoch_row': row, 'cumulative_gpu_seconds_at_commit': clock}
        relative = f'commits/e{epoch:03d}.{uuid.uuid4().hex}.json'
        write(self.path(relative), value)
        reference = {'path': relative, 'sha256': file_hash(self.path(relative))}
        fault('after_manifest')
        write(self.root / 'current.json', reference)
        fault('after_pointer')
        atomic_bytes(self.root / 'epochs.jsonl', b''.join(canonical(r) + b'\n' for r in rows + [row]))
        fault('after_log')
        write(self.root / 'active_costs.json', {'commit': reference, 'completed_epoch': epoch,
            'scientific_config_sha256': self.scientific_sha, 'cumulative_gpu_seconds': clock})
        fault('after_sidecar')
        # 先完成清理和持久化完成标志，再采样时钟；崩溃未留下标志的轮次不可追补进早期切片。
        self.collect_superseded(value)
        completion_path = self.root / 'completions' / f'epoch_{epoch:03d}.json'
        write(completion_path, {'commit': reference, 'last': last, 'snapshot': snapshot,
                                'all_checkpoint_bytes_hashed_and_fsynced': True})
        fault('after_completion')
        finished = gpu_seconds() if callable(gpu_seconds) else gpu_seconds
        mono = time.monotonic()
        assert finished >= clock
        completion = {'commit': reference, 'epoch': epoch, 'checkpoint': snapshot,
            'transaction_completed_monotonic': mono, 'transaction_completed_gpu_seconds': finished,
            'durable_completion_marker_sha256': file_hash(completion_path),
            'clock': 'sum of own actual GPU reservation attempts; includes init/resume/save/drain; excludes queue'}
        write(self.root / 'eligible' / f'epoch_{epoch:03d}.json', completion)
        fault('after_eligibility')
        prefixes = self.prefixes()
        for item in prefixes['items']:
            if item['status'] in ['available', 'unavailable']:
                continue
            if finished <= item['budget_seconds']:
                item.update(checkpoint=snapshot, completed_epoch=epoch, completion=completion,
                            status='pending_candidate')
            else:
                item['status'] = 'available' if item['checkpoint'] else 'unavailable'
        write(self.root / 'prefixes.json', prefixes)
        fault('after_prefix')
        self.collect_superseded(value)
        return value, row

    def collect_superseded(self, current):
        import os
        import re
        keep = {current[k]['path'] for k in ['last', 'best', 'snapshot']}
        keep |= {x['checkpoint']['path'] for x in self.prefixes()['items'] if x['checkpoint']}
        for path in sorted((self.root / 'checkpoints').glob('*.pt')):
            relative = str(path.relative_to(self.root)).replace('\\', '/')
            if relative in keep or not re.fullmatch(r'(last|snapshot)\.e\d{3}\.[0-9a-f]{32}\.pt', path.name):
                continue
            # 仅回收当前 run 自己的过期模型；受保护切片始终在同一原子清单里。
            record = {'path': relative, 'sha256': file_hash(path), 'size_bytes': path.stat().st_size,
                      'action': 'unreferenced_generation_retired'}
            with (self.root / 'checkpoint_generation_history.jsonl').open('ab') as log:
                log.write(canonical(record) + b'\n'); log.flush(); os.fsync(log.fileno())
            path.unlink()
        from .p3_core import fsync_directory
        fsync_directory(self.root / 'checkpoints')

    def finalize_prefixes(self, gpu_seconds):
        prefixes = self.prefixes()
        actual_stop = gpu_seconds() if callable(gpu_seconds) else gpu_seconds
        for item in prefixes['items']:
            if item['status'] not in ['available', 'unavailable']:
                item['status'] = 'available' if item['checkpoint'] else 'unavailable'
            item['actual_run_stop_gpu_seconds'] = actual_stop
        write(self.root / 'prefixes.json', prefixes)
