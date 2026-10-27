"""按本轮 PI 直接授权签发完整性凭证并后台启动；密钥不离开服务器运行时。"""
import json
from remote_session import connect
from transport import command, pull

SCRIPT = r'''
import pathlib,sys,os,time,uuid,secrets,hashlib,hmac,subprocess,shutil
r=pathlib.Path('/root/user_data/tl_cls_v1');sys.path.insert(0,str(r/'src'));sys.path.insert(0,str(r/'tools'))
from p3c_runtime import read,write,sha,canonical,storage_guard
from p3c_supervise import scope
from tlcls.p3_core import code_commit,process_ticks,process_alive
s=r/'records/p3c';ledger=read(r/'records/p3/gpu_ledger.json')
assert code_commit()=='700a1adcba85e03896dfafa303959f8d035008aa'
assert not read(s/'evaluation_ledger.json')['attempt_ids']
assert all(a['status']=='finalized' for a in ledger['attempts'].values())
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
storage_guard(r)
launch=s/'direct_PI_launch.json'
if launch.exists():
 previous=read(launch);assert not process_alive(previous['pid'],previous['start_ticks'])
cpu=read(s/'fault_injection.json');current={pathlib.Path(n).name:h for n,h in scope().items()}
if cpu['source_sha256']!=current:
 history=s/'preparation_history'/('direct_PI_'+uuid.uuid4().hex);history.mkdir(parents=True)
 for n in ['fault_injection.json','fault_injection.log']:shutil.copyfile(s/n,history/n)
 with (s/'fault_injection.log').open('w') as log:
  p=subprocess.run([sys.executable,str(r/'tools/p3c_acceptance.py'),'--output',str(s/'fault_injection.json')],
   cwd=r,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUTF8='1'),stdout=log,stderr=log)
 assert p.returncode==0
cpu=read(s/'fault_injection.json');assert cpu['status']=='passed' and cpu['tests_run']>=28 and cpu['source_sha256']==current
decision={'schema':'P3C_PI_direct_approval_v1','approved_unix':time.time(),
 'user_statement':'我就是PI，明确授权执行P3-C的400项集中测试。本轮取消独立签署要求，密钥配置由你安全处理；3 GPU-hours子上限和160 GPU-hours总上限不变，后台执行，完成后回收分析并推送。',
 'independent_signing_waived_by_PI':True,'P3_C_authorized':True,'evaluation_count':400,
 'subcap_gpu_seconds':10800,'total_cap_gpu_seconds':576000,'new_training_authorized':False}
write(s/'PI_direct_approval.json',decision)
old=r/'records/p3/test_authorization.json'
if old.exists():
 history=s/'authorization_history';history.mkdir(exist_ok=True)
 target=history/(sha(old)+'.json')
 if not target.exists():shutil.copyfile(old,target)
payload=read(s/'authorization_request.json')['payload']
payload.update(approval_id='P3C-direct-'+uuid.uuid4().hex,approved_unix=time.time(),
 PI_independent_issuance=False,approval_mode='direct_user_PI_authorization',independent_signing_waived_by_PI=True,
 PI_direct_approval_sha256=sha(s/'PI_direct_approval.json'),supervisor_source_sha256=scope(),
 CPU_fault_injection_sha256=sha(s/'fault_injection.json'),execution_order_sha256=sha(s/'execution_order.json'))
key=secrets.token_hex(32)
credential_dir=pathlib.Path('/run/tlcls-p3c');credential_dir.mkdir(mode=0o700,exist_ok=True);credential_dir.chmod(0o700)
key_file=credential_dir/(payload['approval_id']+'.key')
fd=os.open(key_file,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as f:f.write(key);f.flush();os.fsync(f.fileno())
payload['signature_hmac_sha256']=hmac.new(key.encode(),canonical(payload),hashlib.sha256).hexdigest()
write(old,payload)
write(s/'runtime_credential_reference.json',{'credential_file':str(key_file),'storage':'host runtime tmpfs; outside project',
 'mode':'0600','never_copy_or_publish':True})
phase=read(s/'evaluation_ledger.json');phase.update(centralized_test_authorized=True,approval_id=payload['approval_id'],
 authorization_sha256=sha(old));write(s/'evaluation_ledger.json',phase)
env=dict(os.environ,TLCLS_TEST_APPROVAL_HMAC_KEY=key,PYTHONUNBUFFERED='1',PYTHONUTF8='1')
log=s/('authorized_launch_'+str(int(time.time()))+'.log')
with log.open('ab') as output:
 p=subprocess.Popen([sys.executable,str(r/'tools/p3c_run_authorized.py'),'--background'],cwd=r,env=env,
  stdout=output,stderr=output,stdin=subprocess.DEVNULL,start_new_session=True)
receipt={'schema':'P3C_direct_PI_launch_v1','pid':p.pid,'start_ticks':process_ticks(p.pid),'started_unix':time.time(),
 'approval_id':payload['approval_id'],'approval_mode':payload['approval_mode'],'independent_signing_waived_by_PI':True,
 'authorization_sha256':sha(old),'supervisor_source_sha256':scope(),'CPU_tests':cpu['tests_run'],
 'log_path':str(log.relative_to(r)),'key_handling':'server runtime tmpfs, mode0600, never copied to client or Git',
 'subcap_gpu_hours':3,'total_cap_gpu_hours':160,'sequence':'two synthetic GPU checks then frozen400 and independent analysis',
 'execution_commit':code_commit()}
write(launch,receipt);print(__import__('json').dumps(receipt))
'''

def main():
    client=connect()
    try:
        code,output,_=command(client,['/root/miniconda3/envs/tl_cls_v1/bin/python','-c',SCRIPT],timeout=120)
        assert code==0,'P3C direct launch failed before confirmed receipt'
        print(output)
    finally:client.close()
    for name in ['records/p3c/direct_PI_launch.json','records/p3c/PI_direct_approval.json',
                 'records/p3c/fault_injection.json','records/p3c/fault_injection.log','records/p3/test_authorization.json']:
        client=connect()
        try:pull(client,name)
        finally:client.close()

if __name__=='__main__':
    try:main()
    except Exception as error:
        print('P3C_direct_launch_error:'+type(error).__name__)
        raise SystemExit(1)
