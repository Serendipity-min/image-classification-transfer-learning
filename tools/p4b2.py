"""B2隔离入口：复用冻结P4训练函数，仅注入独立登记、预算和train/val权限。"""
import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import types
import uuid

ROOT=Path(os.environ.get('TLCLS_ROOT','/root/user_data/tl_cls_v1')).resolve()
sys.path.insert(0,str(ROOT/'src'))
from tlcls.p2_core import read,write,file_hash,value_hash,process_alive,process_ticks
from tlcls.p3_budget import Budget

STAGE=ROOT/'records/p4b2'
CAP=1800
WARNING=540
PYTHON='/root/miniconda3/envs/tl_cls_v1/bin/python'


def readonly_ledger(path=None):return read(path or STAGE/'gpu_ledger.json')
def budget():return Budget(STAGE/'gpu_ledger.json',CAP)
def manifest():return read(STAGE/'run_manifest.json')


def diagnostic_json(value):
    if isinstance(value,float) and not math.isfinite(value):return 'NaN' if math.isnan(value) else 'Infinity' if value>0 else '-Infinity'
    if isinstance(value,dict):return {k:diagnostic_json(v) for k,v in value.items()}
    if isinstance(value,list):return [diagnostic_json(v) for v in value]
    return value


def require_scope():
    a=read(STAGE/'authorization.json');m=manifest();identity=read(STAGE/'execution_identity.json')
    assert a['execution_authorized'] and a['total_cap_GPU_hours']==.5 and not a['test_authorized'] and not a['paper_phase_authorized']
    assert file_hash(STAGE/'run_manifest.json')==identity['protocol_frozen_sha256']
    assert len(m['runs'])==2 and {r['learning_rate'] for r in m['runs']}=={.001,.003}
    reference=read(STAGE/'reference_scientific.json')['configuration']
    for job in m['runs']:
        assert {k for k in reference if reference[k]!=job[k]}=={'run_id','stage','learning_rate','new_test_scope'}
        assert job['dataset']=='flowers102' and job['model_key']=='r50' and job['method']=='full_ft'
        assert job['seed']==123 and job['epochs']==15 and job['head_lr_multiplier']==10 and not job['test_during_training']
        assert job['new_test_scope']=='FORBIDDEN_FOR_P4B2'
    for name,digest in identity['files'].items():assert file_hash(ROOT/name)==digest,'B2 implementation identity mismatch'
    for name,digest in identity['readonly_baseline'].items():assert file_hash(ROOT/name)==digest,'Historical evidence changed'
    return a,m,identity


def storage():
    import shutil
    free=shutil.disk_usage(ROOT).free
    assert free>=14*2**30,'physical exit and atomic-write reserve'
    value={'physical_free_bytes':free,'exit_and_atomic_reserve_bytes':14*2**30,'B2_planned_increment_bytes':4*2**30,
           'no_historical_checkpoint_deletion':True};write(STAGE/'storage_latest.json',value);return value


def model_proxy():
    import p4_data_models as original
    proxy=types.ModuleType('p4_data_models')
    def trainval(key,split,seed,allow_test=False):
        if split not in ['train','validation'] or allow_test:raise PermissionError('B2 rejects official test construction')
        return original.FlowerData(key,split,seed,allow_test=False)
    def build(config,device):
        model,info=original.build_flowers(config,device)
        if info['initial_classifier_sha256']!=manifest()['head_initial_sha256']:
            budget().stop('B2_initialization_identity_mismatch',5)
            raise AssertionError('classifier initialization mismatch')
        assert info['target_classes']==102 and model.fc.in_features==2048 and model.fc.out_features==102
        return model,info
    proxy.build_flowers=build;proxy.FlowerData=trainval
    return proxy


def worker(args):
    require_scope()
    import p4_train as entry
    proxy=model_proxy()
    # 只在新进程路由旧入口的外部依赖；旧模块文件和模型函数均不改写。
    entry.STAGE=STAGE;entry.require_scope=require_scope;entry.jobs=lambda:manifest()['runs']
    entry.configuration=lambda job:dict(job);entry.budget=budget;entry.storage=storage
    entry.write=lambda path,value:write(path,diagnostic_json(value))
    sys.modules['p4_data_models']=proxy
    sys.argv=[str(ROOT/'tools/p4_train.py'),'--run-id',args.run_id,'--device',str(args.device)]
    if args.resume:sys.argv.append('--resume')
    return entry.main()


def signal_group(pid,sig):
    try:os.killpg(pid,sig)
    except ProcessLookupError:pass


def parent_guard():ctypes.CDLL(None).prctl(1,signal.SIGKILL)


def guardian(args):
    _,_,identity=require_scope();b=budget();attempt='p4b2-'+uuid.uuid4().hex
    b.register(attempt,args.run_id,args.device,'P4B2_validation_only',args.owner,args.owner_ticks)
    process=None;reason='entry_not_started'
    try:
        inventory=read(STAGE/'preflight.json')['GPU_inventory'];gpu=inventory[str(args.device)]
        now=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines()
        actual={line.split(',')[0].strip():line.split(',')[1].strip() for line in now};assert actual[str(args.device)]==gpu['uuid']
        b.update(attempt,gpu_uuid=gpu['uuid'],B2_execution_identity_sha256=value_hash(identity),B2_source_commit=identity['P4_source_commit'])
        env=dict(os.environ,TLCLS_ATTEMPT_ID=attempt,PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        env.pop('TLCLS_P4_ISOLATED_PROFILE',None)
        command=[PYTHON,'tools/p4b2.py','worker','--run-id',args.run_id,'--device',str(args.device)]
        if args.resume:command.append('--resume')
        write(STAGE/'attempt_commands'/(attempt+'.json'),{'command':command,'attempt_id':attempt,'gpu_uuid':gpu['uuid'],'execution_identity_sha256':value_hash(identity)})
        process=subprocess.Popen(command,cwd=ROOT,env=env,start_new_session=True,preexec_fn=parent_guard)
        reason='completed'
        b.update(attempt,child_pid=process.pid,child_pgid=process.pid)
        import psutil
        terminating=None
        while process.poll() is None:
            state=b.snapshot();used=b.used(state);stamp=time.monotonic()
            if not process_alive(args.owner,args.owner_ticks):b.stop('B2_owner_lost',5)
            if used>=CAP-30:b.stop('B2_hard_cap_exit_reserve',5)
            if used>=WARNING and not (STAGE/'budget_warning.json').exists():
                done=0;durations=[]
                for job in manifest()['runs']:
                    path=ROOT/'runs'/job['run_id']/'summary.json'
                    if path.exists():done+=read(path)['completed_epochs']
                    for p in (ROOT/'runs'/job['run_id']/'timings').glob('*.json'):durations.append(read(p)['fit_seconds'])
                import statistics
                remaining=(30-done)*(statistics.median(durations) if durations else 10)
                forecast=used+remaining*1.25+30
                write(STAGE/'budget_warning.json',{'used_GPU_seconds':used,'forecast_total_GPU_seconds':forecast,'warning_line':WARNING,'hard_cap':CAP})
                if forecast>CAP:b.stop('B2_forecast_exceeds_cap',5)
            stop=readonly_ledger()['stop_request']
            if stop and terminating is None:
                reason=stop['reason'];terminating=stamp
                try:os.kill(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            if terminating is not None and stamp-terminating>=5:signal_group(process.pid,signal.SIGKILL)
            if int(stamp)%10==0:
                try:storage()
                except AssertionError:b.stop('B2_storage_exit_reserve',5)
            try:
                p=psutil.Process(process.pid);rss=p.memory_info().rss;tree=rss+sum(c.memory_info().rss for c in p.children(recursive=True) if c.is_running())
                previous=state['attempts'][attempt]
                b.update(attempt,main_process_peak_rss_bytes_sampled=max(rss,previous['main_process_peak_rss_bytes_sampled']),
                         process_tree_peak_rss_bytes_sampled=max(tree,previous['process_tree_peak_rss_bytes_sampled']),samples=previous['samples']+1)
            except psutil.Error:pass
            time.sleep(.5)
        process.wait();signal_group(process.pid,signal.SIGKILL)
    finally:
        if process is not None and process.poll() is None:signal_group(process.pid,signal.SIGKILL);process.wait();reason='B2_guardian_exception'
        b.finalize(attempt,process.returncode if process else -1,reason,time.monotonic())
    return process.returncode if process else 1


def launch():
    _,_,identity=require_scope();cpu=read(STAGE/'CPU_acceptance.json')
    assert cpu['status']=='passed' and cpu['execution_identity_sha256']==value_hash(identity)
    assert not readonly_ledger()['attempts'] and not (STAGE/'launch.json').exists()
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    import fcntl
    with (STAGE/'launcher.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        active=[]
        for device,job in enumerate(manifest()['runs']):
            command=[PYTHON,'tools/p4b2.py','guardian','--run-id',job['run_id'],'--device',str(device),'--owner',str(os.getpid()),'--owner-ticks',process_ticks(os.getpid())]
            path=STAGE/'logs'/f'guardian_{device}.log';path.parent.mkdir(parents=True,exist_ok=True)
            with path.open('ab') as log:p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active.append(p)
        write(STAGE/'launch.json',{'pid':os.getpid(),'start_ticks':process_ticks(os.getpid()),'source_commit':identity['P4_source_commit'],
              'started_unix':time.time(),'scope':'two15epoch validation-only candidates; paper deferred'})
        codes=[p.wait() for p in active]
        write(STAGE/'status.json',{'status':'terminal_ready_for_independent_audit' if all(c in [0,2] for c in codes) else 'terminal_requires_failure_review',
              'returncodes':codes,'GPU_seconds':Budget.used(readonly_ledger()),'updated_unix':time.time()})
        return 0 if all(c in [0,2] for c in codes) else 1


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['worker','guardian','launch']);parser.add_argument('--run-id');parser.add_argument('--device',type=int)
    parser.add_argument('--owner',type=int);parser.add_argument('--owner-ticks');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    return launch() if args.mode=='launch' else guardian(args) if args.mode=='guardian' else worker(args)


if __name__=='__main__':raise SystemExit(main())
