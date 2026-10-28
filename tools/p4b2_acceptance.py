"""B2 CPU针对性验收；无CUDA上下文、无测试集构造。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import tempfile
from pathlib import Path
import time
import unittest
from p4b2 import ROOT,STAGE,require_scope,manifest,read,write,value_hash,file_hash,readonly_ledger


class Acceptance(unittest.TestCase):
    def test_exact_two_config_deltas(self):
        require_scope();reference=read(STAGE/'reference_scientific.json')['configuration']
        for job in manifest()['runs']:
            self.assertEqual({k for k in reference if reference[k]!=job[k]},{'run_id','stage','learning_rate','new_test_scope'})
            self.assertEqual((job['epochs'],job['seed'],job['scheduler_horizon_epochs'],job['warmup_epochs']),(15,123,50,5))

    def test_reference_and_rebuilt_head_hash(self):
        import torch
        from tlcls.common import derive_seed,tensor_hash
        expected=manifest()['head_initial_sha256']
        self.assertEqual(read(STAGE/'reference_summary.json')['model_identity']['initial_classifier_sha256'],expected)
        head=torch.nn.Linear(2048,102)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(derive_seed('head',123,'flowers102','r50'));torch.nn.init.trunc_normal_(head.weight,std=.02);torch.nn.init.zeros_(head.bias)
        self.assertEqual(tensor_hash(head.state_dict().items()),expected)

    def test_real_parameter_groups_head10x(self):
        import torch
        from tlcls.models import make_optimizer
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__();self.body=torch.nn.Linear(3,3);self.fc=torch.nn.Linear(3,102)
        for job in manifest()['runs']:
            model=Tiny();optimizer,groups=make_optimizer(model,job)
            for row in groups:self.assertEqual(row['peak_lr'],job['learning_rate']*(10 if row['name'].startswith('fc.') else 1))
            self.assertEqual(optimizer.defaults['betas'],(.9,.999));self.assertEqual(optimizer.defaults['eps'],1e-8)

    def test_real_dataset_order_and_class_range(self):
        from p4_data_models import FlowerData
        from tlcls.data import EpochSampler,epoch_order_hash
        train=FlowerData('r50','train',123);validation=FlowerData('r50','validation',123)
        self.assertEqual((len(train),len(validation)),(1020,1020));self.assertEqual(set(train.raw._labels),set(range(102)))
        rows=[__import__('json').loads(s) for s in (ROOT/'runs/P4C-tuning-flowers102-r50-full_ft-lr3-s123/epochs.jsonl').read_text().splitlines()]
        sampler=EpochSampler(train,123)
        for epoch in range(3):sampler.epoch=epoch;self.assertEqual(epoch_order_hash(sampler),rows[epoch]['sampler_order_sha256'])
        self.assertEqual(tuple(train[(0,0)][0].shape),(3,224,224))

    def test_proxy_denies_official_test(self):
        from p4b2 import model_proxy
        proxy=model_proxy()
        with self.assertRaises(PermissionError):proxy.FlowerData('r50','test',123)
        with self.assertRaises(PermissionError):proxy.FlowerData('r50','train',123,allow_test=True)

    def test_scheduler_and_tail(self):
        import torch
        from tlcls.engine import UpdateScheduler
        p=torch.nn.Parameter(torch.ones(1));o=torch.optim.AdamW([{'params':[p],'peak_lr':.001}]);s=UpdateScheduler(o,8)
        self.assertEqual((s.warmup,s.horizon),(40,400));self.assertEqual(1020%128,124)
        for _ in range(120):s.step()
        self.assertEqual(s.updates,120);self.assertGreater(s.factor(),.01)

    def test_epoch_transaction_resume(self):
        from tlcls.p2_transaction import EpochTransaction
        with tempfile.TemporaryDirectory(dir=STAGE) as tmp:
            transaction=EpochTransaction(Path(tmp),'fixture')
            transaction.commit(1,{'epoch':1},lambda f:f.write(b'last'),lambda f:f.write(b'best'),1,1.)
            recovered,rows=EpochTransaction(Path(tmp),'fixture').load();self.assertEqual(recovered['completed_epoch'],1);self.assertEqual(len(rows),1)
            with self.assertRaises(AssertionError):EpochTransaction(Path(tmp),'wrong_identity').load()

    def test_failed_candidates_not_ranked(self):
        from p4_core import select
        records=[{'status':'numerical_failure','completed_epochs':14,'selection_correct_sum':None,'learning_rate':.003},
                 {'status':'completed','completed_epochs':15,'selection_correct_sum':2502,'learning_rate':.001}]
        self.assertEqual(select(records)['learning_rate'],.001)

    def test_terminal_ledger_read_is_byte_immutable(self):
        with tempfile.TemporaryDirectory(dir=STAGE) as tmp:
            p=Path(tmp)/'ledger.json';write(p,{'attempts':{},'updated_monotonic':1})
            digest=file_hash(p)
            for _ in range(3):readonly_ledger(p)
            self.assertEqual(file_hash(p),digest)

    def test_nonfinite_failure_diagnostics_preserved(self):
        from p4b2 import diagnostic_json
        result=diagnostic_json({'status':'numerical_failure','diagnostics':{'loss':float('nan'),'grad':float('inf')}})
        self.assertEqual(result['status'],'numerical_failure');self.assertEqual(result['diagnostics'],{'loss':'NaN','grad':'Infinity'})


if __name__=='__main__':
    start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Acceptance))
    write(STAGE/'CPU_acceptance.json',{'status':'passed' if result.wasSuccessful() else 'failed','tests_run':result.testsRun,
        'errors':len(result.errors),'failures':len(result.failures),'CPU_wall_seconds':time.monotonic()-start,'GPU_seconds':0,
        'official_test_constructed':False,'execution_identity_sha256':value_hash(read(STAGE/'execution_identity.json'))})
    raise SystemExit(0 if result.wasSuccessful() else 1)
