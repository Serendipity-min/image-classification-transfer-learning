"""P3 CPU 验收：隔离阶段、五种子矩阵、50轮事务、切片和强制关闭测试。"""
from copy import deepcopy
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from tlcls.p3_core import (ROOT, STAGE, candidates, code_commit, execution_identity, file_hash,
                           process_alive, process_ticks, read, scientific_identity, write)
from tlcls.p3_core import infrastructure_exception
from tlcls.p3_transaction import EpochTransaction, committed_summary
from tlcls.p3_budget import Budget

class Checks(unittest.TestCase):
    def test_storage_disappearing_entry_and_fail_closed_errors(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import stat
        import tlcls.p3_budget as module
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
        from tlcls.p3_core import atomic_bytes
        from tlcls.p3_budget import Storage
        errors = []
        with tempfile.TemporaryDirectory(prefix='p3_storage_race_') as folder:
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

    def test_parallel_ledger_adds_time_once(self):
        with tempfile.TemporaryDirectory(prefix='p3_ledger_') as folder:
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
        args = [sys.executable, str(ROOT / 'src/p3_attempt.py'), '--attempt-id', str(index),
                '--run-id', 'CPU-fixture-' + str(index), '--device', str(index), '--owner-pid', str(owner),
                '--owner-ticks', process_ticks(owner), '--kind', 'cpu_fixture', '--fixture-ledger', str(ledger),
                '--fixture-cap', str(cap), '--fixture-grace', '.5', '--', *command]
        process = subprocess.Popen(args, cwd=ROOT, env=dict(os.environ, TLCLS_CPU_TEST='1', CUDA_VISIBLE_DEVICES=''),
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        return process, log

    def test_parent_sigterm_and_sigkill_cleanup_and_reconcile(self):
        for sig in [signal.SIGTERM, signal.SIGKILL]:
            with self.subTest(signal=sig), tempfile.TemporaryDirectory(prefix='p3_owner_') as folder:
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
        with tempfile.TemporaryDirectory(prefix='p3_cap_') as folder:
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
        with tempfile.TemporaryDirectory(prefix='p3_guardian_') as folder:
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

    def test_80_fresh_50_epoch_five_seed_inputs(self):
        runs = candidates()
        self.assertEqual(len(runs), 80)
        self.assertEqual({r['seed'] for r in runs}, {17,42,2026,3407,65537})
        self.assertTrue(all(r['checkpoint_sha256'] and r['checkpoint_revision'] for r in runs))
        from tlcls.p3_core import PIPELINE, ARCHIVE
        for name in PIPELINE:
            blob = subprocess.check_output(['git','-C',str(ROOT),'show',ARCHIVE+':'+name])
            self.assertEqual((ROOT/name).read_bytes(), blob)

    def fixture_transaction(self, folder, thresholds=None):
        return EpochTransaction(folder, 'science', thresholds=thresholds or [25,50,100])

    def test_every_commit_boundary_and_prefix_crash(self):
        boundaries = ['after_last','after_best','after_manifest','after_pointer','after_log',
                      'after_sidecar','after_completion','after_eligibility','after_prefix']
        for boundary in boundaries:
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as folder:
                tx = self.fixture_transaction(folder)
                tx.commit(1, {'epoch':1}, lambda f:f.write(b'last1'), lambda f:f.write(b'model1'), 1, 10)
                def crash(position):
                    if position == boundary: raise RuntimeError('injected')
                with self.assertRaises(RuntimeError):
                    tx.commit(2, {'epoch':2}, lambda f:f.write(b'last2'), lambda f:f.write(b'model2'), 2, 20, crash)
                current, rows = tx.load()
                expected = 1 if boundary in boundaries[:3] else 2
                self.assertEqual(current['completed_epoch'], expected)
                self.assertEqual(len(tx.load()[1]), expected)
                self.assertEqual(tx.path(current['last']['path']).read_bytes(), ('last'+str(expected)).encode())
                self.assertIsNone(tx.compatible_sidecar(999))
                for x in tx.prefixes()['items']:
                    if x['checkpoint']:
                        self.assertTrue(tx.path(x['checkpoint']['path']).exists())
                        self.assertLessEqual(x['completion']['transaction_completed_gpu_seconds'], x['budget_seconds'])

    def test_50_epochs_best_rng_sampler_metadata_and_nonrollback(self):
        with tempfile.TemporaryDirectory() as folder:
            tx=self.fixture_transaction(folder)
            for e in range(1,51):
                row={'epoch':e,'validation':{'top1_percent':e if e<=20 else 0},
                     'cumulative_optimizer_updates':352*e,'sampler_order_sha256':str(e)}
                tx.commit(e,row,lambda f:f.write(str(e).encode()),lambda f:f.write(str(e).encode()),
                          min(e,20),e*2,improved=e<=20)
            tx.finalize_prefixes(100)
            current,rows=tx.load()
            self.assertEqual(len(rows),50)
            self.assertEqual(current['best_epoch'],20)
            self.assertEqual(tx.path(current['best']['path']).read_bytes(), b'20')
            for x in tx.prefixes()['items']:
                self.assertEqual(x['status'],'available')
                self.assertLessEqual(x['completion']['transaction_completed_gpu_seconds'],x['budget_seconds'])
            with self.assertRaises(AssertionError):
                EpochTransaction(folder,'different-science',thresholds=[25,50,100]).load()
            tx.path(current['last']['path']).write_bytes(b'corrupt')
            with self.assertRaises(AssertionError):tx.load()

    def test_full_save_crossing_threshold_cannot_backdate(self):
        with tempfile.TemporaryDirectory() as folder:
            tx=self.fixture_transaction(folder,[10,20,30])
            values=iter([9,11])
            tx.commit(1,{'epoch':1},lambda f:f.write(b'full'),lambda f:f.write(b'model'),1,lambda:next(values))
            first=tx.prefixes()['items'][0]
            self.assertEqual(first['status'],'unavailable')
            self.assertIsNone(first['checkpoint'])
            tx.finalize_prefixes(11)
            self.assertEqual(tx.prefixes()['items'][1]['actual_run_stop_gpu_seconds'],11)
            self.assertEqual(tx.prefixes()['items'][1]['status'],'available')

    def test_prefix_model_hash_immutable_under_best_last_GC(self):
        with tempfile.TemporaryDirectory() as folder:
            tx=self.fixture_transaction(folder,[10,20,30])
            for e in [1,2,3,4]:
                tx.commit(e,{'epoch':e},lambda f:f.write(('full'+str(e)).encode()),
                          lambda f:f.write(('model'+str(e)).encode()),e,e*9)
            p=tx.prefixes()['items']
            self.assertEqual([x['completed_epoch'] for x in p],[1,2,3])
            for item in p:
                self.assertEqual(file_hash(tx.path(item['checkpoint']['path'])),item['checkpoint']['sha256'])
                self.assertEqual(item['status'],'available')
            for key,value in [('seed',42),('learning_rate',1.),('epochs',15),('scheduler_horizon_epochs',15)]:
                base=candidates()[0]
                self.assertNotEqual(scientific_identity(base)['scientific_config_sha256'],
                    scientific_identity(dict(base,**{key:value}))['scientific_config_sha256'])

    def test_storage_200GiB_warning_and_safe_exit(self):
        from unittest.mock import patch
        import tlcls.p3_budget as module
        guard=module.Storage(200*2**30)
        self.assertEqual(guard.allocation,200*2**30)
        with patch.object(guard,'snapshot',return_value={'reserve_bytes':10*2**30,
            'filesystem_free_bytes':13*2**30,'allocation_remaining_bytes':20*2**30}):
            with self.assertRaises(RuntimeError):guard.require(4*2**30)

    def test_operational_resume_keeps_existing_science_and_rejects_math(self):
        from tlcls.p3_core import operational_compatibility, execution_identity
        current=execution_identity()
        for name in ['src/tlcls/engine.py','src/tlcls/p3_transaction.py','src/evaluate.py']:
            altered=deepcopy(current);altered['files'][name]='0'*64
            with self.assertRaises(AssertionError):operational_compatibility(altered)
        for config in candidates():
            path=ROOT/'runs'/config['run_id']/'scientific.json'
            if path.exists():self.assertEqual(read(path),scientific_identity(config))

    def test_resume_reopens_only_future_prefix_deadlines(self):
        from p3_train import reopen_future_prefixes
        with tempfile.TemporaryDirectory() as folder:
            tx=self.fixture_transaction(folder,[10,20,30])
            tx.commit(1,{'epoch':1},lambda f:f.write(b'full1'),lambda f:f.write(b'model1'),1,9)
            tx.finalize_prefixes(9)
            locked=deepcopy(tx.prefixes()['items'][0])
            reopen_future_prefixes(tx,11,'synthetic-resume')
            self.assertEqual(tx.prefixes()['items'][0],locked)
            self.assertEqual(tx.prefixes()['items'][1]['status'],'pending_candidate')
            tx.commit(2,{'epoch':2},lambda f:f.write(b'full2'),lambda f:f.write(b'model2'),2,18)
            self.assertEqual(tx.prefixes()['items'][0],locked)
            self.assertEqual(tx.prefixes()['items'][1]['completed_epoch'],2)
            self.assertTrue((Path(folder)/'prefix_resume_history/synthetic-resume.json').exists())

    def test_forecast_does_not_reinflate_already_spent_time(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        import p3_schedule as scheduler
        previous=read(STAGE/'resume_decision.json')['pause_forecast']
        run=candidates()[0]; name=run['configuration_id']
        seconds=previous['remaining_estimated_gpu_seconds']/50
        budget=SimpleNamespace(snapshot=lambda:{},used=lambda _:previous['actual_gpu_seconds_including_active'])
        # 使用已发生的真实暂停案例；只隔离输出，避免 CPU 验收改写正式预测记录。
        with patch.object(scheduler,'baseline',return_value={name:{'startup_terminal_first_epoch_excess_seconds':0}}), \
             patch.object(scheduler,'write'):
            result=scheduler.forecast([run],{run['run_id']:{'status':'planned','completed_epochs':0}},
                                       {name:seconds},budget)
        self.assertGreater(result['legacy_reinflated_forecast_gpu_hours'],160)
        self.assertLess(result['forecast_with_reserves_and_margin_gpu_hours'],160)
        self.assertFalse(result['forecast_exceeds_cap'])
        self.assertEqual(result['actual_gpu_seconds_including_active'],previous['actual_gpu_seconds_including_active'])
        self.assertEqual(result['cap_gpu_hours'],160)

    def test_fail_closed_missing_corrupt_version_signature_and_seal(self):
        from tlcls.p3_evaluation import require_test_authorization
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); auth=root/'records/p3/test_authorization.json'
            cases=[None,b'{broken',b'{}',b'{"schema":"wrong"}',
                b'{"schema":"P3_C_authorization_v1","centralized_test_authorized":false}',
                b'{"schema":"P3_C_authorization_v1","centralized_test_authorized":true,"separate_user_approval":true,"signature_hmac_sha256":"wrong"}']
            for data in cases:
                if data is not None:
                    auth.parent.mkdir(parents=True,exist_ok=True);auth.write_bytes(data)
                with self.assertRaises(PermissionError):require_test_authorization(root/'manifest.json',root)
        completed=subprocess.run([sys.executable,str(ROOT/'src/evaluate.py'),'--manifest',
            str(ROOT/'missing-manifest.json'),'--device','0'],cwd=ROOT,capture_output=True,text=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
        self.assertNotEqual(completed.returncode,0)
        self.assertIn('centralized test remains closed',completed.stderr)

    def test_signed_synthetic_seal_identity_and_manifest(self):
        import hashlib,hmac
        from unittest.mock import patch
        import tlcls.p3_evaluation as gate
        from tlcls.p3_core import canonical
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); runs=[{'run_id':'synthetic-'+str(i)} for i in range(80)]
            write(root/'records/p3/run_manifest.json',{'runs':runs})
            summaries={}
            for r in runs:
                p=root/'runs'/r['run_id']/'summary.json'
                write(p,{'status':'completed'}); summaries[r['run_id']]=file_hash(p)
            write(root/'selected_hparams.json',{'synthetic':True})
            manifest=root/'manifest.json'
            write(manifest,{'schema':'P3_evaluation_manifest_v1','evaluations':[]})
            seal={'schema':'P3_seal_v1','status':'sealed','evaluation_manifest_sha256':file_hash(manifest),
                'selected_hparams_sha256':file_hash(root/'selected_hparams.json'),
                'execution_identity_sha256':'fixture-code','execution_commit':'fixture-commit',
                'run_ids':[r['run_id'] for r in runs],'summary_sha256':summaries}
            seal_path=root/'records/P3_sealed.json';write(seal_path,seal)
            auth={'schema':'P3_C_authorization_v1','centralized_test_authorized':True,'separate_user_approval':True,
                'seal_sha256':file_hash(seal_path),'evaluation_manifest_sha256':file_hash(manifest),
                'selected_hparams_sha256':file_hash(root/'selected_hparams.json'),
                'execution_identity_sha256':'fixture-code'}
            secret=os.urandom(32).hex()
            auth['signature_hmac_sha256']=hmac.new(secret.encode(),canonical(auth),hashlib.sha256).hexdigest()
            write(root/'records/p3/test_authorization.json',auth)
            with patch.dict(os.environ,{'TLCLS_TEST_APPROVAL_HMAC_KEY':secret}), \
                 patch.object(gate,'execution_identity',return_value={'scope_sha256':'fixture-code'}), \
                 patch.object(gate,'code_commit',return_value='fixture-commit'):
                self.assertEqual(gate.require_test_authorization(manifest,root)['evaluations'],[])
                with patch.object(gate,'execution_identity',return_value={'scope_sha256':'other-code'}):
                    with self.assertRaises(PermissionError):gate.require_test_authorization(manifest,root)
                manifest.write_bytes(b'{}')
                with self.assertRaises(PermissionError):gate.require_test_authorization(manifest,root)

    def test_evaluation_id_roles_budget_and_cache(self):
        from tlcls.p3_evaluation import evaluation_id, cache_key
        base={'run_id':'synthetic','evaluation_role':'budget_prefix','budget_fraction':.25,
              'budget_seconds':10,'checkpoint_sha256':'0'*64,'preprocessing_identity':'frozen',
              'precision_identity':'BF16_FP32'}
        other=dict(base,budget_fraction=.5,budget_seconds=20)
        self.assertNotEqual(evaluation_id(base),evaluation_id(other))
        self.assertEqual(cache_key(base),cache_key(other))
        self.assertNotEqual(evaluation_id(base),evaluation_id(dict(base,evaluation_role='primary_best_val')))


def main():
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    write(STAGE/'cpu_acceptance.json',{'status':'passed' if result.wasSuccessful() else 'failed',
        'tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'execution_commit':code_commit(),'execution_identity':execution_identity(),
        'authorization_sha256':file_hash(STAGE/'authorization.json'),'fixtures_are_cpu_only':True,
        'official_test_access':False})
    return 0 if result.wasSuccessful() else 2

if __name__=='__main__':raise SystemExit(main())
