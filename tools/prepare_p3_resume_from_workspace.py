"""保存暂停现场并绑定已授权的运维修订；不启动训练或改写历史运行身份。"""
from remote_session import connect
from transport import command

SCRIPT = r'''
import datetime,json,pathlib,subprocess,sys,shutil
r=pathlib.Path('/root/user_data/tl_cls_v1');sys.path.insert(0,str(r/'src'));s=r/'records/p3'
from tlcls.p3_core import read,write,file_hash,code_commit,execution_identity,operational_compatibility,candidates,scientific_identity
from tlcls.p3_budget import Budget,Storage
ledger=read(s/'gpu_ledger.json');assert all(a['status']=='finalized' for a in ledger['attempts'].values())
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
old_auth=read(s/'authorization.json');old_forecast=read(s/'forecast.json')
archive=s/'resume_history/20261026_before_resume';archive.mkdir(parents=True,exist_ok=True)
names=['authorization.json','entry_preflight.json','acceptance.json','cpu_acceptance.json','gpu_acceptance.json',
       'gpu_acceptance_0.json','gpu_acceptance_1.json','dispatch_authorization.json','gpu_ledger.json',
       'forecast.json','progress.json','stage_status.json','terminal_audit.json','checkpoint_inventory.json',
       'prefix_manifest.json','evaluation_manifest.draft.json']
for name in names:
 p=s/name
 if p.exists():
  target=archive/name
  if target.exists():assert target.read_bytes()==p.read_bytes(),'preserve original pause evidence'
  else:shutil.copyfile(p,target)
completed={};resumes=[]
for config in candidates():
 d=r/'runs'/config['run_id'];p=d/'summary.json'
 if not p.exists():continue
 v=read(p);assert read(d/'scientific.json')==scientific_identity(config),'science identity changed'
 if v['status']=='completed':
  completed[config['run_id']]={n:file_hash(d/n) for n in ['summary.json','epochs.jsonl','current.json','scientific.json']}
 elif v['status']=='infrastructure_interrupted':
  target=archive/config['run_id'];target.mkdir(exist_ok=True)
  for n in ['summary.json','epochs.jsonl','current.json','scientific.json','prefixes.json']:
   if not (target/n).exists():shutil.copyfile(d/n,target/n)
  ref=read(d/'current.json');commit=read(d/ref['path'])
  assert commit['completed_epoch']==v['completed_epochs']
  resumes.append({'run_id':config['run_id'],'completed_epochs':v['completed_epochs'],'commit':ref,'last':commit['last']})
assert len(completed)==54 and sorted(v['completed_epochs'] for v in resumes)==[3,37]
for n,sha in old_auth['frozen_inputs'].items():assert file_hash(r/n)==sha,n
decision={'schema':'P3_resume_decision_v1','date':'2026-10-26','resume_authorized':True,
 'user_statement':'请结合服务器硬件实际资源，如果可以从终端继续，就继续实验，最终完整结果在回收分析。关于硬件资源设置上限，暂时没有没必要，服务器资源充足',
 'execution_commit':code_commit(),'hard_cap_gpu_seconds':576000,'artificial_storage_ceiling_enforced':False,
 'forecast_policy':'actual_spent_plus_125_percent_of_remaining_and_reserves','pause_forecast':old_forecast,
 'prior_gpu_seconds_not_reset':ledger['gpu_seconds_finalized'],'preserved_completed_runs':completed,
 'resume_boundaries':resumes,'operational_compatibility':operational_compatibility(),
 'pause_text_export':{'id':'ed0627ff14f9435f8c55f5fa9d534ae0','sha256':'3c5a086fe0891df5848e8ae718b8ba0e69bb1973a13e6312d13ff3b93205b412'},
 'centralized_test_authorized':False}
write(s/'resume_decision.json',decision)
auth=dict(old_auth,execution_commit=code_commit(),execution_identity=execution_identity(),
 operational_compatibility=operational_compatibility(),resource_policy='physical free space; historical 200GiB minimum is not a ceiling',
 resume_authority='explicit_user_instruction_2026-10-26')
write(s/'authorization.json',auth)
preflight=read(s/'entry_preflight.json');preflight.update(execution_commit=code_commit(),execution_identity=execution_identity(),
 storage=Storage(200*2**30).require(4*2**30),resume_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 artificial_storage_ceiling_enforced=False,retained_original_preflight=str(archive.relative_to(r)/'entry_preflight.json'))
write(s/'entry_preflight.json',preflight)
write(s/'dispatch_authorization.json',{'P3_B_authorized':False,'reason':'resume acceptance pending',
 'centralized_test_authorized':False})
if ledger['stop_request']:Budget().clear_technical_stop()
print(json.dumps({'prepared':True,'execution_commit':code_commit(),'completed_preserved':len(completed),
 'resume_epochs':[x['completed_epochs'] for x in resumes],'GPU_hours_preserved':ledger['gpu_seconds_finalized']/3600,
 'formal_dispatch_authorized':False}))
'''


if __name__ == '__main__':
    client = connect()
    try:
        code, output, error = command(client, ['/root/miniconda3/envs/tl_cls_v1/bin/python', '-c', SCRIPT], timeout=60)
        assert code == 0, error
        print(output.strip())
    finally: client.close()
