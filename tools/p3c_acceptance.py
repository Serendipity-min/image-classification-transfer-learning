"""只在临时目录、合成字节和 CPU 张量上验收 P3-C；不访问官方 test。"""
import argparse
import hashlib
import hmac
import itertools
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, '/root/user_data/tl_cls_v1/src')
import p3c_runtime as rt
import p3c_supervise as supervisor
import p3c_analyze as analysis

def item(identity='eval-synthetic', role='primary_best_val', fraction=None):
    return {'evaluation_id': identity, 'run_id': 'P3-synthetic-s17', 'evaluation_role': role,
        'checkpoint_sha256': '0' * 64, 'budget_fraction': fraction, 'inference_cache_key': identity}

class Acceptance(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(); self.root = Path(self.directory.name)
        self.key = os.urandom(32).hex().encode(); self.stage = self.root / 'records/p3c'
        boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        shared = {'schema': 1, 'phase': 'P3', 'cap_gpu_seconds': rt.TOTAL_CAP, 'attempts': {},
            'boot_id': boot, 'stop_request': None, 'events': [], 'P3_authorized': True,
            'centralized_test_authorized': False}
        rt.write(self.root / 'records/p3/gpu_ledger.json', shared)
        rt.write(self.stage / 'p3b_gpu_ledger.sealed.json', shared)
        rt.write(self.stage / 'evaluation_ledger.json', {'subcap_gpu_seconds': rt.CAP, 'attempt_ids': [],
            'P3_B_gpu_ledger_sha256': rt.sha(self.stage / 'p3b_gpu_ledger.sealed.json'), 'stop_request': None})
        rt.write(self.stage / 'dispatch.json', {'claims': {}, 'next_ordinal': 0})

    def tearDown(self):
        self.directory.cleanup()

    def publish(self, target=None, fault=None):
        return rt.publish_output(self.root, target or item(), 'manifest', lambda f: f.write(b'synthetic prediction'),
            {'correct': 1, 'n': 2}, 'cpu-fixture', lambda: .1, fault)

    def test_atomic_completed_output_and_restart(self):
        self.publish(); value = rt.verify_output(self.root, item(), 'manifest')
        self.assertEqual(value['metrics']['correct'], 1)
        self.assertEqual(value['evaluation_wall_seconds_in_attempt'], .1)

    def test_interruption_after_prediction_fsync_is_incomplete(self):
        def fail(point):
            if point == 'after_file_sync': raise InterruptedError()
        with self.assertRaises(InterruptedError): self.publish(fault=fail)
        self.assertIsNone(rt.verify_output(self.root, item(), 'manifest'))
        self.publish(); self.assertIsNotNone(rt.verify_output(self.root, item(), 'manifest'))

    def test_interruption_after_prediction_rename_is_incomplete(self):
        def fail(point):
            if point == 'after_prediction_publish': raise InterruptedError()
        with self.assertRaises(InterruptedError): self.publish(fault=fail)
        self.assertIsNone(rt.verify_output(self.root, item(), 'manifest'))
        self.publish(); self.assertIsNotNone(rt.verify_output(self.root, item(), 'manifest'))

    def test_interruption_after_receipt_commit_is_complete(self):
        def fail(point):
            if point == 'after_receipt_publish': raise InterruptedError()
        with self.assertRaises(InterruptedError): self.publish(fault=fail)
        self.assertIsNotNone(rt.verify_output(self.root, item(), 'manifest'))

    def test_corrupt_prediction_is_not_skipped(self):
        v = self.publish(); (self.root / v['predictions_path']).write_bytes(b'corrupt')
        with self.assertRaises(ValueError): rt.verify_output(self.root, item(), 'manifest')

    def test_wrong_checkpoint_identity_rejected(self):
        self.publish()
        with self.assertRaises(ValueError): rt.verify_output(self.root, dict(item(), checkpoint_sha256='bad'), 'manifest')
        with self.assertRaises(ValueError): supervisor.validate_checkpoint({'config': {'model': 'wrong'},
            'scientific_config_sha256': 's'}, {'model': 'right'}, 's')

    def test_wrong_manifest_rejected(self):
        self.publish()
        with self.assertRaises(ValueError): rt.verify_output(self.root, item(), 'other')

    def test_path_escape_rejected(self):
        with self.assertRaises(ValueError): rt.checked_path(self.root, '../escape')
        target = self.root / 'link'; target.symlink_to(self.root.parent)
        with self.assertRaises(ValueError): rt.checked_path(self.root, 'link/escape')

    def test_storage_reserve_refuses_write(self):
        with self.assertRaises(RuntimeError): rt.storage_guard(self.root, 14 * 2**30 - 1)
        self.assertEqual(rt.storage_guard(self.root, 14 * 2**30), 14 * 2**30)

    def test_order_and_role_barrier(self):
        a = item('z-best'); b = item('a-last', 'secondary_last'); c = item('prefix', 'budget_prefix', .25)
        order = rt.ordered([c, b, a]); self.assertEqual([x['evaluation_id'] for x in order], ['z-best', 'a-last', 'prefix'])
        self.assertEqual(rt.next_item(self.root, order, 'm', 0, 'a')['evaluation_id'], 'z-best')
        self.assertEqual(rt.next_item(self.root, order, 'm', 1, 'b'), 'WAIT')
        rt.publish_output(self.root, a, 'm', lambda f:f.write(b'x'), {}, 'a', lambda:1)
        self.assertEqual(rt.next_item(self.root, order, 'm', 1, 'b')['evaluation_id'], 'a-last')

    def test_duplicate_evaluation_id_rejected(self):
        with self.assertRaises(ValueError): rt.ordered([item(), item()])

    def test_same_gpu_duplicate_lease_rejected(self):
        from tlcls.p3_core import process_ticks
        phase = rt.Phase(self.root, self.key)
        phase.register('p3c-a', 0, os.getpid(), process_ticks(os.getpid()), 'p3c_cpu_fixture')
        with self.assertRaises(AssertionError): phase.register('p3c-b', 0, os.getpid(), process_ticks(os.getpid()), 'p3c_cpu_fixture')
        phase.finalize('p3c-a', 0, 'synthetic')

    def test_two_gpu_distinct_lease_and_idempotent_finalization(self):
        from tlcls.p3_core import process_ticks
        phase = rt.Phase(self.root, self.key)
        for d in [0, 1]: phase.register('p3c-' + str(d), d, os.getpid(), process_ticks(os.getpid()), 'p3c_cpu_fixture')
        phase.finalize('p3c-0', 0, 'synthetic'); phase.finalize('p3c-1', 0, 'synthetic')
        before = rt.sha(self.root / 'records/p3/gpu_ledger.json'); phase.finalize('p3c-0', 0, 'synthetic')
        self.assertEqual(before, rt.sha(self.root / 'records/p3/gpu_ledger.json'))
        self.assertGreater(rt.verify_chain(self.root, self.key), 0)

    def test_historical_attempt_cannot_change(self):
        shared = rt.read(self.root / 'records/p3/gpu_ledger.json')
        shared['attempts']['old'] = {'status': 'finalized', 'charged_gpu_seconds': 1}
        rt.write(self.stage / 'p3b_gpu_ledger.sealed.json', shared)
        shared['attempts']['old']['charged_gpu_seconds'] = 0
        rt.write(self.root / 'records/p3/gpu_ledger.json', shared)
        with self.assertRaises(AssertionError): rt.Phase(self.root, self.key).register('p3c-a', 0, 0, '', 'p3c_cpu_fixture')

    def test_append_chain_tampering_rejected(self):
        from tlcls.p3_core import process_ticks
        phase = rt.Phase(self.root, self.key); phase.register('p3c-a', 0, os.getpid(), process_ticks(os.getpid()), 'p3c_cpu_fixture')
        rt.verify_chain(self.root, self.key)
        with self.assertRaises(AssertionError): rt.verify_chain(self.root, b'wrong key')
        phase.finalize('p3c-a', 0, 'synthetic')

    def test_dual_budget_admission(self):
        phase = rt.Phase(self.root, self.key)
        for counts in [(rt.CAP, 0), (0, rt.TOTAL_CAP)]:
            with patch.object(phase, 'used', return_value=counts), self.assertRaises(AssertionError):
                phase.register('p3c-a', 0, 0, '', 'p3c_cpu_fixture')

    def test_guardian_stops_real_cpu_child_on_owner_loss(self):
        args = SimpleNamespace(attempt_id='p3c-a', device=0, owner_pid=99999999, owner_ticks='0', synthetic=True)
        with self.assertRaises(RuntimeError):
            supervisor.supervise_child(args, rt.Phase(self.root, self.key), [sys.executable, '-c', 'import time; time.sleep(20)'], self.root)
        value = rt.read(self.root / 'records/p3/gpu_ledger.json')['attempts']['p3c-a']
        self.assertEqual(value['status'], 'finalized'); self.assertLess(value['charged_gpu_seconds'], 5)
        self.assertEqual(value['termination_reason'], 'operator_or_owner_exit')

    def test_guardian_stops_real_cpu_child_before_subcap(self):
        from tlcls.p3_core import process_ticks
        args = SimpleNamespace(attempt_id='p3c-a', device=1, owner_pid=os.getpid(), owner_ticks=process_ticks(os.getpid()), synthetic=True)
        phase = rt.Phase(self.root, self.key)
        counts = itertools.chain([(0, 0)], itertools.repeat((rt.CAP - rt.EXIT_SECONDS + 1, 0)))
        with patch.object(phase, 'used', side_effect=counts), self.assertRaises(RuntimeError):
            supervisor.supervise_child(args, phase, [sys.executable, '-c', 'import time; time.sleep(20)'], self.root)
        self.assertEqual(rt.read(self.root / 'records/p3/gpu_ledger.json')['attempts']['p3c-a']['termination_reason'], 'dual_gpu_budget_exit_guard')

    def test_duplicate_supervisor_lock_rejected(self):
        path = self.stage / 'supervisor.lock'
        with rt.lock(path, blocking=False):
            command = [sys.executable, '-c', 'import fcntl,sys; f=open(sys.argv[1],"a");fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)', str(path)]
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_signature_cap_and_source_binding(self):
        payload = {'schema': 'P3_C_authorization_v1', 'centralized_test_authorized': True,
            'separate_user_approval': True, 'PI_independent_issuance': True, 'subcap_gpu_seconds': rt.CAP,
            'total_cap_gpu_seconds': rt.TOTAL_CAP, 'approval_id': 'synthetic',
            'execution_order_sha256': 'order', 'supervisor_source_sha256': {'synthetic': 'sha'}}
        def sign(v): return dict(v, signature_hmac_sha256=hmac.new(self.key, rt.canonical(v), hashlib.sha256).hexdigest())
        signed = sign(payload)
        rt.validate_payload(signed, self.key, {'synthetic': 'sha'}, 'order')
        for key, value in [('subcap_gpu_seconds', rt.CAP + 1), ('separate_user_approval', False)]:
            with self.assertRaises(PermissionError): rt.validate_payload(sign(dict(payload, **{key:value})), self.key, {'synthetic':'sha'}, 'order')
        with self.assertRaises(PermissionError): rt.validate_payload(signed, b'x'*32, {'synthetic':'sha'}, 'order')
        with self.assertRaises(PermissionError): rt.validate_payload(signed, self.key, {'other':'sha'}, 'order')
        with self.assertRaises(PermissionError): rt.validate_payload(signed, b'', {'synthetic':'sha'}, 'order')

    def test_original_authorization_missing_or_invalid_is_closed(self):
        from tlcls.p3_evaluation import require_test_authorization
        with self.assertRaises(PermissionError): require_test_authorization(self.root / 'manifest.json', self.root)
        rt.write(self.root / 'records/p3/test_authorization.json', {'schema':'P3_C_authorization_v1'})
        with self.assertRaises(PermissionError): require_test_authorization(self.root / 'manifest.json', self.root)

    def test_direct_PI_authorization_requires_explicit_waiver(self):
        payload={'schema':'P3_C_authorization_v1','centralized_test_authorized':True,'separate_user_approval':True,
            'PI_independent_issuance':False,'approval_mode':'direct_user_PI_authorization',
            'independent_signing_waived_by_PI':True,'subcap_gpu_seconds':rt.CAP,'total_cap_gpu_seconds':rt.TOTAL_CAP,
            'approval_id':'synthetic-direct','execution_order_sha256':'order','supervisor_source_sha256':{'synthetic':'sha'}}
        def sign(v):return dict(v,signature_hmac_sha256=hmac.new(self.key,rt.canonical(v),hashlib.sha256).hexdigest())
        rt.validate_payload(sign(payload),self.key,{'synthetic':'sha'},'order')
        for wrong in [dict(payload,independent_signing_waived_by_PI=False),dict(payload,approval_mode='unspecified')]:
            with self.assertRaises(PermissionError):rt.validate_payload(sign(wrong),self.key,{'synthetic':'sha'},'order')

    def test_metrics_on_known_cpu_logits(self):
        import torch
        logits = torch.tensor([[3.,0.],[0.,3.],[0.,3.],[3.,0.]])
        metrics = supervisor.classification_metrics(logits, torch.tensor([0,1,0,0]))
        self.assertEqual(metrics['correct'], 3); self.assertEqual(metrics['n'], 4)
        self.assertEqual(metrics['top1_percent'], 75); self.assertEqual(metrics['confusion_matrix'], [[2,1],[0,1]])
        self.assertAlmostEqual(metrics['balanced_accuracy'], (2/3+1)/2)
        self.assertFalse(torch.cuda.is_initialized())

    def test_forecast_does_not_use_accuracy(self):
        target = item(); estimates = {target['evaluation_id']: 10}
        value = rt.forecast(self.root, [target], estimates, 1)
        self.assertEqual(value['forecast_gpu_seconds'], 71)
        self.assertTrue(rt.forecast(self.root, [target], estimates, rt.CAP)['forecast_exceeds_subcap'])

    def test_worker_requires_live_matching_guardian_lease(self):
        value = {'status':'running','device':0,'guardian_pid':123,'kind':'p3c_evaluation'}
        supervisor.validate_worker_lease(value,0,False,123)
        for wrong in [dict(value,status='finalized'),dict(value,device=1),dict(value,guardian_pid=1),dict(value,kind='candidate')]:
            with self.assertRaises(AssertionError): supervisor.validate_worker_lease(wrong,0,False,123)

    def test_independent_metrics_match_tensor_metrics(self):
        import numpy as np
        import torch
        logits=np.array([[3.,0.],[0.,3.],[0.,3.],[3.,0.]],dtype=np.float32);labels=np.array([0,1,0,0],dtype=np.int64)
        actual=analysis.independent_metrics(logits,labels);expected=supervisor.classification_metrics(torch.from_numpy(logits),torch.from_numpy(labels))
        self.assertEqual(actual['confusion_matrix'],expected['confusion_matrix'])
        self.assertAlmostEqual(actual['nll'],expected['nll'],places=6)

    def test_real_compressed_npz_atomic_round_trip(self):
        import numpy as np
        data=np.array([[1.,2.],[3.,4.]],dtype=np.float32)
        def writer(file):np.savez_compressed(file,logits=data,labels=np.array([1,0]),sample_id=np.array(['s0','s1']))
        value=rt.publish_output(self.root,item(),'manifest',writer,{},'cpu-fixture',lambda:.1)
        verified=rt.verify_output(self.root,item(),'manifest')
        with np.load(self.root/verified['predictions_path'],allow_pickle=False) as actual:
            self.assertTrue(np.array_equal(actual['logits'],data))

    def test_paired_statistics_fixed_denominator_and_noninferiority(self):
        records=[]
        for seed,delta in zip([17,42,2026,3407,65537],[-1,-1.1,-.9,-1.2,-.8]):
            for config,score in [('cifar100-vitb16-full_ft',90),('cifar100-vitb16-linear_probe',90+delta),
                ('svhn-r50-full_ft',95),('svhn-r50-adapter',95-delta)]:
                records.append({'configuration_id':config,'seed':seed,'top1_percent':score})
        results=analysis.paired_statistics(records)
        self.assertAlmostEqual(results[0]['mean_difference_pp'],-1)
        self.assertFalse(results[0]['noninferiority_CI95_lower_rule'])
        self.assertTrue(results[1]['superiority_Holm_alpha_0_05'])
        self.assertLessEqual(results[0]['paired_t_two_sided_p'],results[0]['Holm_adjusted_p'])
        incomplete=analysis.paired_statistics(records[1:])
        self.assertEqual(incomplete[0]['n_complete_pairs'],4)
        self.assertIsNone(incomplete[0]['paired_t_two_sided_p'])
        self.assertIsNone(incomplete[1]['Holm_adjusted_p'])

def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); args = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CPU验收必须屏蔽CUDA'
    started = time.time()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Acceptance))
    rt.write(args.output, {'schema': 'P3C_CPU_fault_injection_v1', 'status':'passed' if result.wasSuccessful() else 'failed',
        'tests_run': result.testsRun, 'failures':len(result.failures), 'errors':len(result.errors),
        'CPU_wall_seconds': time.time()-started, 'GPU_seconds':0, 'official_test_accessed':False,
        'temporary_synthetic_fixtures_only':True, 'source_sha256':{p.name:rt.sha(p) for p in
            [Path(__file__), Path(rt.__file__), Path(supervisor.__file__),Path(analysis.__file__)]}})
    raise SystemExit(0 if result.wasSuccessful() else 1)

if __name__ == '__main__':
    main()
