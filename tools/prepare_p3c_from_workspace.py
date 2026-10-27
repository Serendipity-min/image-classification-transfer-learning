"""部署独立运维工具并做 CPU 入口验收；不生成真实授权或访问测试集。"""
import json
from pathlib import Path
from remote_session import connect
from transport import ROOT, command, upload, pull

SCRIPT = r'''
import pathlib,json,sys,time,subprocess,os,shutil,hashlib,statistics
r=pathlib.Path('/root/user_data/tl_cls_v1');sys.path.insert(0,str(r/'src'));sys.path.insert(0,str(r/'tools'))
from p3c_runtime import read,write,sha,ordered,atomic,CAP,TOTAL_CAP,storage_guard
from tlcls.p3_core import code_commit
stage=r/'records/p3c';stage.mkdir(exist_ok=True)
assert code_commit()=='700a1adcba85e03896dfafa303959f8d035008aa'
seal=read(r/'records/P3_sealed.json');manifest_path=r/'records/p3/evaluation_manifest.draft.json'
assert sha(r/'records/P3_sealed.json')=='977d8c69bef9f6a783b4f5188102e5d25f1e55497b3ed1e79791265a7d02c8c6'
assert sha(manifest_path)=='0dcfff493c6dcde0c92aa357f2e64434eaa7ab9d9e960bf45453c14893e5e663'
assert sha(r/'selected_hparams.json')=='66662cbb09bf8973530ab257b49a92617fce27a9129b0b6bbadb924d4fe7e74f'
shared=r/'records/p3/gpu_ledger.json';assert sha(shared)==seal['gpu_ledger_sha256']
assert all(a['status']=='finalized' for a in read(shared)['attempts'].values())
assert not (r/'records/p3/test_authorization.json').exists(),'已有授权时不覆盖准备状态'
inventory=read(r/'records/p3/checkpoint_inventory.json')['items']
assert len(inventory)==508
for cp in inventory:
 assert (r/cp['path']).is_file() and (r/cp['path']).stat().st_size==cp['size_bytes']
space=storage_guard(r);prior=read(r/'records/p3/server_closeout_verification.json')
assert prior['status']=='passed' and prior['seal_sha256']==sha(r/'records/P3_sealed.json')
baseline=stage/'p3b_gpu_ledger.sealed.json'
if baseline.exists():assert sha(baseline)==seal['gpu_ledger_sha256']
else:atomic(baseline,shared.read_bytes())
items=ordered(read(manifest_path)['evaluations']);assert len(items)==400
estimates={}
configs={x['run_id']:x for x in read(r/'records/p3/run_manifest.json')['runs']}
for item in items:
 c=configs[item['run_id']];rows=[json.loads(x) for x in (r/'runs'/item['run_id']/'epochs.jsonl').read_text().splitlines()]
 median=statistics.median(x['validation_seconds'] for x in rows if x['epoch']>=2)
 estimates[item['evaluation_id']]=median*(2 if c['dataset']=='cifar100' else 26032/7326)*1.25+5
write(stage/'execution_order.json',{'schema':'P3C_order_v1','evaluation_manifest_sha256':sha(manifest_path),
 'sort':'80 primary_best_val then 80 secondary_last then 240 budget_prefix; stable run_id/fraction; completion barrier between roles',
 'evaluation_ids':[x['evaluation_id'] for x in items],'estimated_seconds':estimates,
 'estimation':'own P3 validation epochs2-50 median scaled by official test sample counts, future margin1.25 plus5s load/write; no test data read'})
ledger=stage/'evaluation_ledger.json'
if ledger.exists():assert read(ledger)['attempt_ids']==[]
else:write(ledger,{'schema':'P3C_ledger_v1','subcap_gpu_seconds':CAP,'total_cap_gpu_seconds':TOTAL_CAP,
 'P3_B_gpu_ledger_sha256':seal['gpu_ledger_sha256'],'P3_B_actual_gpu_seconds':448598.1942236051,
 'attempt_ids':[],'stop_request':None,'chain_tip_sha256':None,'centralized_test_authorized':False})
write(stage/'dispatch.json',{'claims':{},'next_ordinal':0})
write(stage/'entry_preflight.json',{'schema':'P3C_entry_preflight_v1','status':'prepared_NOT_AUTHORIZED',
 'checked_unix':time.time(),'execution_commit':code_commit(),'seal_sha256':sha(r/'records/P3_sealed.json'),
 'evaluation_manifest_sha256':sha(manifest_path),'selected_hparams_sha256':sha(r/'selected_hparams.json'),
 'checkpoint_inventory_entries_present_with_sizes':508,'evaluation_sources':400,
 'checkpoint_bytes_rehashed':False,'model_hash_evidence':'reuse accepted sealed terminal audit and server_closeout_verification per explicit user request; original runtime authorization remains unchanged',
 'prior_server_closeout_verification_sha256':sha(r/'records/p3/server_closeout_verification.json'),
 'P3_B_gpu_ledger_sha256':seal['gpu_ledger_sha256'],'baseline_ledger_archived_readonly_by_policy':True,
 'execution_order_sha256':sha(stage/'execution_order.json'),'filesystem_free_bytes':space,
 'storage_exit_guard_bytes':14*2**30,'new_GPU_seconds':0,'subcap_gpu_seconds':CAP,
 'total_cap_gpu_seconds':TOTAL_CAP,'P3_C_authorized':False,'official_test_accessed':False,
 'missing_external_requirements':['independent PI signature','PI-controlled runtime HMAC key injection'],
 'GPU_acceptance':'NOT_RUN until independently signed scope available'})
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUTF8='1')
with (stage/'fault_injection.log').open('w') as log:
 result=subprocess.run([sys.executable,str(r/'tools/p3c_acceptance.py'),'--output',str(stage/'fault_injection.json')],cwd=r,env=env,stdout=log,stderr=log)
assert result.returncode==0,'CPU_fault_injection_failed'
denial=subprocess.run([sys.executable,str(r/'tools/p3c_supervise.py')],cwd=r,env=env,capture_output=True,text=True)
assert denial.returncode!=0 and 'P3C_stopped:' in denial.stderr
assert sha(shared)==seal['gpu_ledger_sha256'],'CPU preparation changed production GPU ledger'
preflight=read(stage/'entry_preflight.json');preflight['actual_unsigned_launch_denied']=True
preflight['production_gpu_ledger_unchanged_after_acceptance']=True;preflight['CPU_acceptance']=read(stage/'fault_injection.json')['status']
write(stage/'entry_preflight.json',preflight)
write(stage/'status.json',{'status':'WAITING_INDEPENDENT_PI_SIGNATURE','CPU_acceptance':read(stage/'fault_injection.json')['status'],
 'GPU_acceptance':'NOT_RUN','official_test_accessed':False,'P3_C_authorized':False,'new_GPU_seconds':0})
print(json.dumps({'CPU_acceptance':read(stage/'fault_injection.json'),'checkpoint_files_present':508,
 'new_GPU_seconds':0,'official_test_accessed':False,'execution_commit':code_commit()}))
'''

def main():
    client = connect()
    try:
        history = r'''import pathlib,uuid,shutil
r=pathlib.Path('/root/user_data/tl_cls_v1');s=r/'records/p3c'
if (s/'fault_injection.json').exists():
 h=s/'preparation_history'/uuid.uuid4().hex;h.mkdir(parents=True)
 for p in [s/'fault_injection.json',s/'fault_injection.log',s/'entry_preflight.json']+list((r/'tools').glob('p3c_*.py')):
  if p.exists():shutil.copyfile(p,h/p.name)
'''
        assert command(client, ['/root/miniconda3/envs/tl_cls_v1/bin/python', '-c', history])[0] == 0
        client.close()
        for name in ['p3c_runtime.py', 'p3c_supervise.py', 'p3c_acceptance.py', 'p3c_analyze.py']:
            client = connect()
            upload(client, ROOT / 'p3_source/tools' / name, 'tools/' + name)
            client.close()
        client = connect()
        code, output, _ = command(client, ['/root/miniconda3/envs/tl_cls_v1/bin/python', '-c', SCRIPT], timeout=180)
        print(output)
        client.close()
        for name in ['entry_preflight.json','fault_injection.json','fault_injection.log','status.json',
                     'execution_order.json','evaluation_ledger.json']:
            client = connect()
            pull(client, 'records/p3c/' + name)
            client.close()
        assert code == 0, 'P3C preparation failed; inspect CPU receipt'
    finally:
        client.close()

if __name__ == '__main__':
    try:main()
    except Exception as error:
        print('P3C_prepare_failed:'+type(error).__name__)
        raise SystemExit(1)
