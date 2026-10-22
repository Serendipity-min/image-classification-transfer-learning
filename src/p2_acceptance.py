"""新增 P2 风险的 CPU 验收：事务故障、精确选参、身份、并发账本及父进程死亡。"""

from copy import deepcopy
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

from tlcls.p2_core import (ROOT, STAGE, candidates, code_commit, execution_identity, file_hash,
                           process_alive, process_ticks, read, scientific_identity, write)
from tlcls.p2_core import infrastructure_exception
from tlcls.p2_core import operational_compatibility, scientific_implementation_basis
from tlcls.p2_transaction import EpochTransaction, committed_summary
from tlcls.p2_budget import Budget
from p2_select import choose


class Checks(unittest.TestCase):
    def test_storage_disappearing_entry_and_fail_closed_errors(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import stat
        import tlcls.p2_budget as module
        class Entry:
            def __init__(self, result): self.result = result
            def lstat(self):
                if isinstance(self.result, Exception): raise self.result
                return self.result
        metadata = SimpleNamespace(st_mode=stat.S_IFREG, st_dev=1, st_ino=7, st_size=123)
        link = SimpleNamespace(st_mode=stat.S_IFLNK, st_dev=1, st_ino=8, st_size=100)
        fake = SimpleNamespace(is_dir=lambda: True, rglob=lambda _: iter([
            Entry(FileNotFoundError()), Entry(metadata), Entry(metadata), Entry(link)]))
        with patch.object(module, 'Path', return_value=fake):
            self.assertEqual(module.Storage.size('fixture'), 123)
        for error in [PermissionError('fixture'), OSError(5, 'fixture I/O')]:
            fake.rglob = lambda _, error=error: iter([Entry(error)])
            with patch.object(module, 'Path', return_value=fake), self.assertRaises(type(error)):
                module.Storage.size('fixture')

    def test_storage_during_atomic_rename_and_retirement(self):
        import threading
        from tlcls.p2_core import atomic_bytes
        from tlcls.p2_budget import Storage
        errors = []
        with tempfile.TemporaryDirectory(prefix='p2_storage_race_') as folder:
            directory = Path(folder)
            def writer():
                try:
                    for index in range(200):
                        atomic_bytes(directory / 'ledger.json', b'x' * 1024)
                        transient = directory / ('retired_' + str(index))
                        transient.write_bytes(b'y' * 2048)
                        transient.unlink()
                except Exception as error: errors.append(error)
            thread = threading.Thread(target=writer)
            thread.start()
            for _ in range(200): self.assertGreaterEqual(Storage.size(directory), 0)
            thread.join(timeout=10)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(Storage.size(directory), 1024)

    def test_operational_compatibility_rejects_science_changes(self):
        current = execution_identity()
        for name in ['src/tlcls/engine.py', 'src/tlcls/models.py', 'src/tlcls/p2_transaction.py', 'src/p2_select.py']:
            altered = deepcopy(current)
            altered['files'][name] = '0' * 64
            with self.assertRaises(AssertionError): operational_compatibility(altered)
        self.assertEqual(operational_compatibility()['scientific_implementation_basis'], scientific_implementation_basis())

    def test_existing_science_identities_and_frozen_plan(self):
        plan = read(STAGE / 'candidate_plan.json')
        self.assertEqual(plan['execution_identity'], scientific_implementation_basis())
        existing = 0
        for item in plan['runs']:
            config = item['config']
            self.assertEqual(item['scientific_identity'], scientific_identity(config))
            path = ROOT / 'runs' / config['run_id'] / 'scientific.json'
            if path.exists():
                self.assertEqual(read(path), scientific_identity(config))
                existing += 1
        self.assertEqual(existing, 19)

    def test_frozen_assets_splits_and_actual_dependencies(self):
        import importlib.metadata
        for asset in read(ROOT / 'records/assets.json')['assets']:
            path = ROOT / asset['remote']
            self.assertEqual(path.stat().st_size, asset['size_bytes'])
            self.assertEqual(file_hash(path), asset['sha256'])
        data = read(ROOT / 'records/data_integrity.json')['datasets']
        for name, value in data.items():
            split = ROOT / 'records' / (name + '_split.json')
            self.assertEqual(file_hash(split), value['split_sha256'])
            manifest = read(split)
            self.assertEqual(file_hash(ROOT / manifest['sample_manifest_path']), manifest['sample_manifest_sha256'])
        for package, version in {'torch': '2.5.1+cu124', 'torchvision': '0.20.1+cu124',
                                 'timm': '1.0.12', 'huggingface-hub': '0.26.2', 'safetensors': '0.4.5'}.items():
            self.assertEqual(importlib.metadata.version(package), version)

    def test_infrastructure_failure_classification(self):
        self.assertTrue(infrastructure_exception(RuntimeError('DataLoader worker terminated')))
        self.assertTrue(infrastructure_exception(RuntimeError('storage_exit_reserve_or_allocation_insufficient')))
        self.assertTrue(infrastructure_exception(RuntimeError('worker interrupted'), True))
        self.assertFalse(infrastructure_exception(AssertionError('scientific mismatch'), True))
        self.assertFalse(infrastructure_exception(FloatingPointError('nonfinite'), True))
        self.assertFalse(infrastructure_exception(ValueError('invalid config')))

    def test_matrix_and_frozen_mathematics(self):
        self.assertEqual(len(candidates()), 48)
        for name in execution_identity()['files']:
            if name in ['src/tlcls/common.py', 'src/tlcls/data.py', 'src/tlcls/models.py', 'src/tlcls/engine.py']:
                old = subprocess.check_output(['git', '-C', str(ROOT), 'show',
                    '94b02df2ae94097602e40e88783f8c445006fa9a:' + name])
                self.assertEqual((ROOT / name).read_bytes(), old)

    def test_scientific_identity_changes(self):
        base = candidates()[0]
        original = scientific_identity(base)['scientific_config_sha256']
        for key, value in [('learning_rate', 0.2), ('tuning_candidate_index', 2), ('epochs', 14),
                           ('micro_batch_size', 16), ('gradient_accumulation_steps', 8),
                           ('autocast_dtype', 'float16'), ('scheduler_horizon_epochs', 15), ('warmup_epochs', 2)]:
            altered = dict(base, **{key: value})
            self.assertNotEqual(original, scientific_identity(altered)['scientific_config_sha256'])
        with tempfile.TemporaryDirectory(prefix='p2_identity_') as folder:
            transaction = EpochTransaction(folder, original)
            transaction.commit(1, {'epoch': 1}, lambda f: f.write(b'last'), lambda f: f.write(b'best'), 1, 1)
            with self.assertRaises(AssertionError):
                EpochTransaction(folder, 'mismatched-scientific-hash').load()

    def test_test_entry_closed_before_dataset_creation(self):
        completed = subprocess.run([sys.executable, str(ROOT / 'src/evaluate.py'),
            '--manifest', str(ROOT / 'not-a-real-evaluation-manifest.json'), '--device', '0'],
            cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn('centralized test remains closed', completed.stderr)

    def test_storage_enforces_confirmed_allocation_and_exit_reserve(self):
        import tlcls.p2_budget as module
        old_root = module.ROOT
        try:
            with tempfile.TemporaryDirectory(prefix='p2_storage_') as folder:
                module.ROOT = Path(folder)
                # 稀疏文件仅作 CPU 边界夹具；真实资源保障来自用户明确确认，不以此测试冒充配额。
                with (Path(folder) / 'fixture').open('wb') as file:
                    file.truncate(2 * 2**30)
                guard = module.Storage(6 * 2**30, environment=Path(folder) / 'missing')
                with self.assertRaises(RuntimeError):
                    guard.require()
        finally:
            module.ROOT = old_root

    def test_every_transaction_boundary_and_log_repair(self):
        boundaries = ['after_last', 'after_best', 'after_manifest', 'after_pointer', 'after_log', 'after_sidecar']
        for boundary in boundaries:
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory(prefix='p2_transaction_') as folder:
                tx = EpochTransaction(folder, 'science')
                tx.commit(1, {'epoch': 1}, lambda f: f.write(b'last1'), lambda f: f.write(b'best1'), 1, 10)
                def crash(position):
                    if position == boundary:
                        raise RuntimeError('injected_boundary_failure')
                with self.assertRaises(RuntimeError):
                    tx.commit(2, {'epoch': 2}, lambda f: f.write(b'last2'), lambda f: f.write(b'best2'), 2, 20, crash)
                expected = 1 if boundary in boundaries[:3] else 2
                value, rows = tx.load()
                self.assertEqual(value['completed_epoch'], expected)
                for _ in range(3):
                    self.assertEqual(tx.load()[1], rows)
                lines = (Path(folder) / 'epochs.jsonl').read_text().splitlines()
                self.assertEqual(len(lines), expected)
                self.assertEqual(tx.path(value['last']['path']).read_bytes(), f'last{expected}'.encode())
                self.assertIsNone(tx.compatible_sidecar(999))

    def test_corrupt_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix='p2_corruption_') as folder:
            tx = EpochTransaction(folder, 'science')
            value, _ = tx.commit(1, {'epoch': 1}, lambda f: f.write(b'last1'), lambda f: f.write(b'best1'), 1, 1)
            tx.path(value['last']['path']).write_bytes(b'corrupt')
            with self.assertRaises(AssertionError):
                tx.load()

    def test_selector_exact_tie_failure_and_incomplete(self):
        def item(lr, correct, status='completed'):
            return {'learning_rate': lr, 'selection_correct_sum': correct, 'selection_denominator': 30000,
                    'status': status, 'eligible': status == 'completed'}
        self.assertEqual(choose([item(.003, 99), item(.001, 99), item(.01, 98)])['selected']['learning_rate'], .001)
        self.assertEqual(choose([item(.003, 10001), item(.001, 10000), item(.01, 99)])['selected']['learning_rate'], .003)
        self.assertEqual(choose([item(.003, 99), item(.001, 0, 'numerical_failure')])['status'], 'selected')
        self.assertEqual(choose([item(.003, 99), item(.001, 0, 'infrastructure_interrupted')])['status'], 'incomplete')
        self.assertEqual(choose([item(.003, 0, 'numerical_failure')])['status'], 'blocked')

    def test_selector_reads_epochs_13_to_15_not_peak(self):
        import p2_select
        original_root, original_identity = p2_select.ROOT, p2_select.scientific_identity
        config = candidates()[0]
        identity = {'scientific_config': {'fixture': 'CPU-only'}, 'scientific_config_sha256': 'fixture'}
        try:
            with tempfile.TemporaryDirectory(prefix='p2_selector_') as folder:
                p2_select.ROOT = Path(folder)
                p2_select.scientific_identity = lambda c: identity
                root = Path(folder)
                write(root / 'records/cifar100_split.json', {'train_indices': list(range(7)), 'validation_indices': list(range(3))})
                directory = root / 'runs' / config['run_id']
                tx = EpochTransaction(directory, 'fixture')
                write(directory / 'config.json', config)
                write(directory / 'scientific.json', identity)
                for epoch in range(1, 16):
                    correct = [1, 2, 2][epoch - 13] if epoch >= 13 else 3
                    row = {'epoch': epoch, 'status': 'completed', 'scientific_config_sha256': 'fixture',
                        'train': {'samples': 7, 'loss': .5, 'optimizer_updates': 1, 'last_update': {'samples': 7}},
                        'validation': {'samples': 3, 'correct': correct, 'top1_percent': 100 * correct / 3, 'loss': .5},
                        'cumulative_optimizer_updates': epoch}
                    tx.commit(epoch, row, lambda f: f.write(b'fixture last'),
                              (lambda f: f.write(b'fixture best')) if epoch == 1 else None, 1, epoch)
                latest, committed_rows = tx.load()
                restored_summary = committed_summary(latest, committed_rows, read(directory / 'current.json'))
                write(directory / 'summary.json', dict(restored_summary, **{'phase': 'P2', 'run_id': config['run_id'],
                    'status': 'completed', 'scientific_config_sha256': 'fixture', 'test_evaluator_constructed': False,
                    'test_metrics': None, 'completed_epochs': 15}))
                result = p2_select.inspect_candidate(config)
                self.assertEqual(result['selection_correct_sum'], 5)
                self.assertEqual(result['selection_epochs'], [13, 14, 15])
                self.assertAlmostEqual(result['selection_value'], 100 * 5 / 9)
                self.assertEqual(restored_summary['completed_epochs'], 15)
                self.assertEqual(restored_summary['optimizer_updates'], 15)
                self.assertEqual(restored_summary['best_checkpoint'], latest['best'])
                self.assertEqual(restored_summary['last_checkpoint'], latest['last'])
                self.assertEqual(restored_summary['selected_epoch'], 1)
                self.assertEqual(restored_summary['best_val_top1_percent'], 100)
        finally:
            p2_select.ROOT, p2_select.scientific_identity = original_root, original_identity

    def test_parallel_ledger_adds_time_once(self):
        with tempfile.TemporaryDirectory(prefix='p2_ledger_') as folder:
            budget = Budget(Path(folder) / 'ledger.json', 100)
            for i in [0, 1]:
                budget.register(str(i), 'fixture', i, 'cpu_fixture', os.getpid(), process_ticks(os.getpid()))
            time.sleep(.2)
            ended = time.monotonic()
            for i in [0, 1]:
                budget.finalize(str(i), 0, 'fixture', ended)
            used = budget.used(budget.snapshot())
            self.assertGreaterEqual(used, .4)
            budget.finalize('0', 0, 'duplicate', ended + 100)
            self.assertEqual(used, budget.used(budget.snapshot()))

    def launch_guardian(self, ledger, owner, index, command, cap=100):
        log = open(ledger.parent / f'guardian{index}.log', 'wb')
        args = [sys.executable, str(ROOT / 'src/p2_attempt.py'), '--attempt-id', str(index),
                '--run-id', 'CPU-fixture-' + str(index), '--device', str(index), '--owner-pid', str(owner),
                '--owner-ticks', process_ticks(owner), '--kind', 'cpu_fixture', '--fixture-ledger', str(ledger),
                '--fixture-cap', str(cap), '--fixture-grace', '.5', '--', *command]
        process = subprocess.Popen(args, cwd=ROOT, env=dict(os.environ, TLCLS_CPU_TEST='1', CUDA_VISIBLE_DEVICES=''),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return process, log

    def test_parent_sigterm_and_sigkill_cleanup_and_reconcile(self):
        for sig in [signal.SIGTERM, signal.SIGKILL]:
            with self.subTest(signal=sig), tempfile.TemporaryDirectory(prefix='p2_owner_') as folder:
                root = Path(folder)
                owner = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
                budget = Budget(root / 'ledger.json', 100)
                child_file = root / 'child.json'
                program = ('import json,subprocess,sys,time; from pathlib import Path; '
                           'c=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"]); '
                           'Path(sys.argv[1]).write_text(json.dumps({"child":c.pid})); time.sleep(60)')
                guard, log = self.launch_guardian(budget.path, owner.pid, 0, [sys.executable, '-c', program, str(child_file)])
                try:
                    deadline = time.monotonic() + 8
                    while not child_file.exists() and time.monotonic() < deadline:
                        time.sleep(.05)
                    self.assertTrue(child_file.exists())
                    nested = read(child_file)['child']
                    owner.send_signal(sig)
                    owner.wait()
                    guard.wait(timeout=6)
                    value = budget.snapshot()
                    attempt = value['attempts']['0']
                    self.assertEqual(attempt['status'], 'finalized')
                    self.assertEqual(attempt['termination_reason'], 'owner_process_terminated')
                    self.assertGreater(attempt['charged_gpu_seconds'], 0)
                    self.assertFalse(process_alive(attempt['child_pid']))
                    deadline = time.monotonic() + 2
                    while process_alive(nested) and time.monotonic() < deadline:
                        time.sleep(.05)
                    self.assertFalse(process_alive(nested))
                finally:
                    if owner.poll() is None:
                        owner.kill()
                        owner.wait()
                    if guard.poll() is None:
                        guard.kill()
                        guard.wait()
                    log.close()

    def test_shared_cap_uses_one_drain_deadline(self):
        with tempfile.TemporaryDirectory(prefix='p2_cap_') as folder:
            budget = Budget(Path(folder) / 'ledger.json', 65)
            guards = [self.launch_guardian(budget.path, os.getpid(), i,
                [sys.executable, '-c', 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'], 65)
                for i in [0, 1]]
            try:
                for guard, _ in guards:
                    guard.wait(timeout=10)
                value = budget.snapshot()
                self.assertEqual(value['stop_request']['reason'], 'shared_gpu_budget_hard_stop')
                self.assertTrue(all(a['status'] == 'finalized' for a in value['attempts'].values()))
                self.assertGreaterEqual(budget.used(value), 5)
                self.assertLess(budget.used(value), 10)
                self.assertLess(budget.used(value), 65)
            finally:
                for guard, log in guards:
                    if guard.poll() is None:
                        guard.kill()
                        guard.wait()
                    log.close()

    def test_guardian_sigkill_is_reconciled_once(self):
        with tempfile.TemporaryDirectory(prefix='p2_guardian_') as folder:
            budget = Budget(Path(folder) / 'ledger.json', 100)
            guard, log = self.launch_guardian(budget.path, os.getpid(), 0,
                [sys.executable, '-c', 'import time; time.sleep(60)'])
            try:
                deadline = time.monotonic() + 6
                while time.monotonic() < deadline:
                    snapshot = budget.snapshot()
                    if snapshot['attempts'].get('0', {}).get('child_pid'):
                        break
                    time.sleep(.05)
                self.assertIsNotNone(budget.snapshot()['attempts']['0']['child_pid'])
                guard.kill()
                guard.wait()
                budget.reconcile_dead_guardian('0')
                first = budget.used(budget.snapshot())
                budget.reconcile_dead_guardian('0')
                self.assertEqual(first, budget.used(budget.snapshot()))
                self.assertFalse(process_alive(budget.snapshot()['attempts']['0']['child_pid']))
            finally:
                if guard.poll() is None:
                    guard.kill()
                    guard.wait()
                log.close()


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    value = {'status': 'passed' if result.wasSuccessful() else 'failed', 'tests_run': result.testsRun,
        'failures': len(result.failures), 'errors': len(result.errors), 'execution_commit': code_commit(),
        'execution_identity': execution_identity(), 'authorization_sha256': file_hash(STAGE / 'authorization.json'),
        'fixtures_are_cpu_only': True, 'P1_reexecution': False,
        'coverage': ['48-run dry-run', 'exact selector and ties', 'numerical versus infrastructure failure',
                     'all six epoch transaction boundaries', 'idempotent log repair', 'scientific hash mismatch',
                     'corrupt checkpoint rejection', 'two-device ledger and idempotent settlement',
                     'owner SIGTERM/SIGKILL process-group cleanup', 'shared cap and single drain deadline']}
    write(STAGE / 'cpu_acceptance.json', value)
    return 0 if result.wasSuccessful() else 2


if __name__ == '__main__':
    raise SystemExit(main())
