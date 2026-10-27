"""已验收修订的单次恢复入口；先保全54个完成run，再恢复调度与30分钟服务器检查。"""
import json
import subprocess
import argparse
from remote_session import connect
from transport import ROOT, command

SCRIPT = r'''
import json,pathlib,subprocess,sys,time,shutil,signal
r=pathlib.Path('/root/user_data/tl_cls_v1');sys.path.insert(0,str(r/'src'));s=r/'records/p3'
from tlcls.p3_core import read,write,require_authorization,process_alive,process_ticks,file_hash,code_commit
from tlcls.p3_budget import Storage
require_authorization();receipt=read(s/'resume_acceptance.json')
assert receipt['status']=='passed' and receipt['execution_commit']==code_commit()
old=read(s/'scheduler.json');assert not process_alive(old['pid'],old['start_ticks'])
ledger=read(s/'gpu_ledger.json');assert all(a['status']=='finalized' for a in ledger['attempts'].values())
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert sys.argv[2] in ['resume_decision.json','slowdown_resume_decision.json']
d=read(s/sys.argv[2])
label='slowdown_resume' if sys.argv[2].startswith('slowdown') else 'resume'
for run_id,files in d['preserved_completed_runs'].items():
 for n,sha in files.items():assert file_hash(r/'runs'/run_id/n)==sha
for item in d['resume_boundaries']:
 assert read(r/'runs'/item['run_id']/'current.json')==item['commit']
Storage().require(4*2**30)
started=time.time();log=(s/'resume_scheduler.log').open('ab')
p=subprocess.Popen([sys.executable,str(r/'src/p3_schedule.py'),'--resume'],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
registration={'scheduler_pid':p.pid,'start_ticks':process_ticks(p.pid),'started_unix':started,'execution_commit':code_commit(),
 'published_resume_receipt_commit':sys.argv[1],'completed_runs_unchanged':len(d['preserved_completed_runs']),'centralized_test_authorized':False}
write(s/(label+'_launch.json'),registration)
write(s/'stage_status.json',{'status':'running_resumed','execution_commit':code_commit(),'reason':None,
 'resumed_unix':started,'previous_pause_preserved_in':d.get('pause_history_path','records/p3/resume_history/20261026_before_resume'),
 'P3_C_authorized':False,'centralized_test_authorized':False})
# 只等待本次调度器发布启动回执，最多15秒；不会在前台持续监测训练。
deadline=time.monotonic()+15
while time.monotonic()<deadline:
 progress=read(s/'progress.json')
 parent=read(s/'scheduler.json')
 if parent['pid']==p.pid and progress['updated_unix']>=started and not progress['reason']:break
 time.sleep(.2)
else:raise RuntimeError('resume scheduler startup receipt not yet ready; inspect existing process, do not duplicate launch')
monitor=s/'monitor';history=monitor/'history'/str(int(started));history.mkdir(parents=True,exist_ok=True)
if (monitor/'registration.json').exists():
 previous=read(monitor/'registration.json')
 if process_alive(previous['pid'],previous['start_ticks']):os_signal=signal.SIGTERM; __import__('os').kill(previous['pid'],os_signal)
for name in ['registration.json','latest.json','attention.json']:
 source=monitor/name
 if source.exists():
  target=history/name
  if not target.exists():shutil.copyfile(source,target)
  if name=='attention.json':source.unlink()
mlog=(monitor/'monitor.log').open('ab')
mp=subprocess.Popen([sys.executable,str(r/'tools/p3_monitor_server.py')],cwd=r,stdout=mlog,stderr=subprocess.STDOUT,start_new_session=True)
write(s/(label+'_monitor_launch.json'),{'pid':mp.pid,'start_ticks':process_ticks(mp.pid),'interval_seconds':1800,
 'script_sha256':file_hash(r/'tools/p3_monitor_server.py'),'started_unix':time.time(),'training_restarts_permitted':False})
print(json.dumps({'resumed':True,'scheduler_pid':p.pid,'monitor_pid':mp.pid,'interval_minutes':30,
 'execution_commit':code_commit(),'completed_preserved':len(d['preserved_completed_runs']),
 'resume_after_epochs':[x['completed_epochs'] for x in d['resume_boundaries']],
 'forecast':progress['forecast'],'official_test_authorized':False}))
'''


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--decision',choices=['resume_decision.json','slowdown_resume_decision.json'],default='resume_decision.json')
    args=parser.parse_args()
    source = ROOT / 'p3_source'
    published = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    remote = subprocess.check_output(['git', '-c', 'core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=15',
        'ls-remote', 'origin', 'refs/heads/codex/p3-execution'], cwd=source, text=True).split()[0]
    assert remote == published, '先推送恢复回执，再启动'
    client = connect()
    try:
        code, output, error = command(client, ['/root/miniconda3/envs/tl_cls_v1/bin/python', '-c', SCRIPT, published,args.decision])
        assert code == 0, error
        print(output.strip())
    finally: client.close()
