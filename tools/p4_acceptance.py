"""P4 CPU 入口验收：真实数学函数、配对重采样及事务故障边界。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from p4_core import ROOT,STAGE,require_scope,jobs,configuration,select,read,write,value_hash,lock,budget,file_hash
from p4_statistics import calibration,holm,paired_image_bootstrap,primary_noninferiority
from tlcls.p2_transaction import EpochTransaction


class Acceptance(unittest.TestCase):
    def test_registered_matrix_and_LR(self):
        _,f,_=require_scope();rows=jobs()
        self.assertEqual(len(rows),84);self.assertEqual(len({r['run_id'] for r in rows}),84)
        self.assertEqual(sum(r['stage']=='P4C-formal' for r in rows),40)
        self.assertTrue(all(r['test_during_training'] is False for r in rows))
        for r in rows:
            self.assertEqual(r['head_lr_multiplier'],10 if r['method']=='full_ft' else 1)
            self.assertEqual((r['scheduler_horizon_epochs'],r['warmup_epochs']),(50,5))
        self.assertEqual(f['design_protocol']['C']['primary_question']['margin_pp'],-.5)

    def test_smaller_LR_exact_tie_and_failed_exclusion(self):
        rows=[{'run_id':str(i),'status':status,'completed_epochs':15,'selection_correct_sum':score,'learning_rate':lr}
              for i,(status,score,lr) in enumerate([('completed',500,.003),('completed',500,.001),('numerical_failure',1000,.01)])]
        self.assertEqual(select(rows)['learning_rate'],.001)

    def test_actual_tail_update_normalization(self):
        import torch
        from tlcls.engine import effective_update
        from tlcls.models import make_optimizer
        torch.manual_seed(9);a=torch.nn.Linear(3,2);b=torch.nn.Linear(3,2);b.load_state_dict(a.state_dict())
        oa=torch.optim.SGD(a.parameters(),lr=.01);ob=torch.optim.SGD(b.parameters(),lr=.01)
        class Scheduler:
            updates=0
            def step(self):self.updates+=1
        x=torch.randn(124,3);y=torch.arange(124)%2
        batches=[(x[i:i+32],y[i:i+32],[str(j) for j in range(i,min(124,i+32))]) for i in range(0,124,32)]
        actual=effective_update(a,oa,Scheduler(),batches,torch.device('cpu'),bf16=False)
        ob.zero_grad();torch.nn.functional.cross_entropy(b(x),y).backward();torch.nn.utils.clip_grad_norm_(b.parameters(),1.,foreach=True);ob.step()
        self.assertEqual(actual['samples'],124)
        for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,atol=1e-7,rtol=1e-6)

    def test_scheduler_fifty_epoch_horizon(self):
        import torch
        from tlcls.engine import UpdateScheduler
        p=torch.nn.Parameter(torch.ones(1));o=torch.optim.AdamW([{'params':[p],'peak_lr':.001}]);s=UpdateScheduler(o,8)
        self.assertEqual((s.warmup,s.horizon),(40,400));self.assertAlmostEqual(s.factor(),.01)
        s.updates=399;self.assertAlmostEqual(s.factor(),.01)

    def test_ECE_Brier_and_bins(self):
        x=np.array([[1000.,1000.],[-1000.,1000.]]);y=np.array([0,1])
        for bins in [10,15,30]:
            r=calibration(x,y,bins);self.assertEqual(sum(b['n'] for b in r['reliability']),2)
            self.assertAlmostEqual(r['brier_no_class_division'],.25);self.assertAlmostEqual(r['ece'],.25)
            self.assertAlmostEqual(r['nll'],np.log(2)/2)

    def test_shared_images_not_seed_pooling(self):
        d=np.tile(np.array([1,1,-1,-1]),(5,1));labels=np.array([0,0,1,1])
        r=paired_image_bootstrap(d,labels,np.random.default_rng(1),n=200)
        self.assertEqual(r['CI95_percentile_pp'],[0.,0.])
        crossed=paired_image_bootstrap(d,labels,np.random.default_rng(2),n=200,seed_rng=np.random.default_rng(3))
        self.assertEqual(crossed['CI95_percentile_pp'],[0.,0.]);self.assertTrue(crossed['seed_pairs_resampled'])

    def test_Holm_family_and_missing_pairs(self):
        self.assertEqual(holm([.01,.04,.02]),[.03,.04,.04])
        self.assertIsNone(primary_noninferiority([0,0])['noninferiority_supported'])
        self.assertFalse(primary_noninferiority([-.5]*5)['noninferiority_supported'])
        self.assertTrue(primary_noninferiority([0]*5)['noninferiority_supported'])

    def test_no_test_constructor_for_training(self):
        from p4_data_models import FlowerData
        with self.assertRaises(AssertionError):FlowerData('r50','test',123)

    def test_conditional_test_gate_closed_before_seal(self):
        from p4_infer import flowers_test_gate
        with self.assertRaises(FileNotFoundError):flowers_test_gate({})

    def test_epoch_commit_faults_and_resume(self):
        for point in ['after_last','after_best','after_manifest','after_pointer','after_log','after_sidecar']:
            with tempfile.TemporaryDirectory() as tmp:
                t=EpochTransaction(Path(tmp),'fixture');row={'epoch':1}
                def fault(p):
                    if p==point:raise InterruptedError('injected')
                with self.assertRaises(InterruptedError):t.commit(1,row,lambda f:f.write(b'last'),lambda f:f.write(b'best'),1,1.,fault)
                state,rows=t.load()
                self.assertEqual(len(rows),1 if point in ['after_pointer','after_log','after_sidecar'] else 0)
                if state:self.assertEqual(state['completed_epoch'],1)

    def test_duplicate_slot_and_finalization(self):
        from tlcls.p3_budget import Budget
        with tempfile.TemporaryDirectory() as tmp:
            b=Budget(Path(tmp)/'fixture.json',1000);ticks=__import__('p4_core').process_ticks(os.getpid())
            b.register('a','fixture',0,'cpu_fixture',os.getpid(),ticks)
            with self.assertRaises(AssertionError):b.register('b','fixture2',0,'cpu_fixture',os.getpid(),ticks)
            b.register('c','fixture2',1,'cpu_fixture',os.getpid(),ticks)
            now=__import__('time').monotonic();b.finalize('a',0,'completed',now);first=b.snapshot()['attempts']['a']
            b.finalize('a',0,'completed',now+100);self.assertEqual(first,b.snapshot()['attempts']['a'])
            b.finalize('c',0,'completed',now)

    def test_immutable_old_evidence(self):
        _,_,identity=require_scope();self.assertIn('records/p3/gpu_ledger.json',identity['historical_read_only_files'])
        self.assertNotEqual(budget().path,ROOT/'records/p3/gpu_ledger.json')

    def test_prediction_transaction_faults(self):
        from p4_atomic import publish,completed
        for point in ['file_synced','data_published','receipt_published']:
            with tempfile.TemporaryDirectory(dir=STAGE) as tmp:
                directory=Path(tmp)
                def fault(p):
                    if p==point:raise InterruptedError('injected')
                with self.assertRaises(InterruptedError):publish(directory,'fixture',lambda f:np.savez_compressed(f,x=np.arange(3)),{'synthetic':True},fault)
                self.assertEqual(completed(directory,'fixture') is not None,point=='receipt_published')


if __name__=='__main__':
    import time
    start=time.monotonic();suite=unittest.defaultTestLoader.loadTestsFromTestCase(Acceptance);result=unittest.TextTestRunner(verbosity=2).run(suite)
    write(STAGE/'CPU_acceptance.json',{'status':'passed' if result.wasSuccessful() else 'failed','tests_run':result.testsRun,
        'errors':len(result.errors),'failures':len(result.failures),'CPU_wall_seconds':time.monotonic()-start,'GPU_seconds':0,
        'official_test_prediction_accessed':False,'source_identity_sha256':value_hash(read(STAGE/'execution_identity.json'))})
    raise SystemExit(0 if result.wasSuccessful() else 1)
