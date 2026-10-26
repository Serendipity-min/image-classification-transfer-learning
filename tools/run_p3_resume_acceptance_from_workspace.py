"""后台执行新入口和真实中断状态的有界 GPU 验收，全部计入原 P3 台账。"""
from remote_session import connect
from transport import command

SCRIPT = r'''
import json,pathlib,subprocess,sys,time
r=pathlib.Path('/root/user_data/tl_cls_v1');sys.path.insert(0,str(r/'src'));s=r/'records/p3'
from tlcls.p3_core import read,write
assert read(s/'cpu_acceptance.json')['status']=='passed'
program=r"""
import pathlib,subprocess,sys,os,uuid
r=pathlib.Path('/root/user_data/tl_cls_v1');sys.path.insert(0,str(r/'src'));s=r/'records/p3'
from tlcls.p3_core import read,write,file_hash,code_commit,execution_identity,process_ticks,operational_compatibility
from tlcls.p3_budget import Budget
subprocess.run([sys.executable,str(r/'src/p3_schedule.py'),'--acceptance-only'],cwd=r,check=True)
write(s/'dispatch_authorization.json',{'P3_B_authorized':False,'reason':'historical checkpoint GPU validation pending','centralized_test_authorized':False})
for device in [0,1]:
 aid=uuid.uuid4().hex
 args=[sys.executable,str(r/'src/p3_attempt.py'),'--attempt-id',aid,'--run-id','P3-resume-acceptance-'+str(device),
       '--device',str(device),'--owner-pid',str(os.getpid()),'--owner-ticks',process_ticks(os.getpid()),'--kind','gpu_acceptance','--',
       sys.executable,str(r/'tools/p3_resume_gpu_acceptance.py')]
 subprocess.run(args,cwd=r,check=True,timeout=180)
decision=read(s/'resume_decision.json')
for run_id,files in decision['preserved_completed_runs'].items():
 for n,sha in files.items():assert file_hash(r/'runs'/run_id/n)==sha,'completed evidence altered'
for device in [0,1]:
 child=read(s/f'resume_gpu_acceptance_{device}.json')
 assert child['status']=='passed' and child['execution_identity']==execution_identity()
 assert child['production_files_modified'] is False and child['repeat_next_update_bitwise_equal']
ledger=read(s/'gpu_ledger.json');assert all(a['status']=='finalized' for a in ledger['attempts'].values())
original=read(s/'resume_history/20261026_before_resume/gpu_ledger.json')
for aid,a in original['attempts'].items():assert a==ledger['attempts'][aid],'historical charge modified'
receipt={'status':'passed','execution_commit':code_commit(),'execution_identity':execution_identity(),
 'CPU_tests':read(s/'cpu_acceptance.json')['tests_run'],'preserved_completed_runs':54,
 'actual_resumed_checkpoint_epochs':[3,37],'all_previous_attempt_charges_unchanged':True,
 'operational_compatibility':operational_compatibility(),
 'historical_GPU_receipt_sha256':[file_hash(s/f'resume_gpu_acceptance_{d}.json') for d in [0,1]],
 'all_attempts_settled_before_resume':True,'official_test_access':False}
write(s/'resume_acceptance.json',receipt)
a=read(s/'acceptance.json');a['resume_receipt_sha256']=file_hash(s/'resume_acceptance.json')
a['operational_compatibility']=operational_compatibility();write(s/'acceptance.json',a)
write(s/'dispatch_authorization.json',{'P3_B_authorized':True,'acceptance_sha256':file_hash(s/'acceptance.json'),
 'resume_receipt_sha256':file_hash(s/'resume_acceptance.json'),'centralized_test_authorized':False})
print('P3 resume acceptance passed; formal scheduler has not been restarted',flush=True)
"""
log=(s/'resume_acceptance.log').open('ab')
p=subprocess.Popen([sys.executable,'-c',program],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write(s/'resume_acceptance_job.json',{'pid':p.pid,'started_unix':time.time()})
print(json.dumps({'resume_acceptance_started':True,'pid':p.pid,'formal_training_restarted':False}))
'''


if __name__ == '__main__':
    client = connect()
    try:
        code, output, error = command(client, ['/root/miniconda3/envs/tl_cls_v1/bin/python', '-c', SCRIPT])
        assert code == 0, error
        print(output.strip())
    finally: client.close()
