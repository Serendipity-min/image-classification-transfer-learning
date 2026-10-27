"""P3-C 单次只读检查；不读取密钥或在运行中回显测试分数。"""
import json
from remote_session import connect
from transport import ROOT,command

SCRIPT=r'''
import pathlib,json,time,sys,shutil
r=pathlib.Path('/root/user_data/tl_cls_v1');sys.path.insert(0,str(r/'src'))
from tlcls.p3_core import read,process_alive
s=r/'records/p3c';ledger=read(r/'records/p3/gpu_ledger.json');phase=read(s/'evaluation_ledger.json')
launch=read(s/'direct_PI_launch.json');status=read(s/'status.json');now=time.monotonic()
used=lambda a:a['charged_gpu_seconds'] if a['status']=='finalized' else max(0,now-a['started_monotonic'])
attempts=list(ledger['attempts'].values());own=[a for a in attempts if a['kind'].startswith('p3c_')]
counts={}
for p in (s/'results').glob('*/completed.json'):
 v=read(p);role=v['evaluation_role'];counts[role]=counts.get(role,0)+1
alive=process_alive(launch['pid'],launch['start_ticks']);settled=all(a['status']=='finalized' for a in own)
audit=read(s/'independent_delivery_audit.json') if (s/'independent_delivery_audit.json').exists() else None
reason=None
if phase.get('stop_request'):reason=phase['stop_request']['reason']
elif not alive and settled:
 reason='completed_ready_for_recovery' if sum(counts.values())==400 and audit and audit['status']=='passed' else 'background_exited_before_delivery_complete'
v={'checked_unix':time.time(),'wrapper_alive':alive,'status':status,'completed':sum(counts.values()),
 'completed_by_role':counts,'all_P3C_attempts_settled':settled,'P3C_GPU_hours_including_active':sum(map(used,own))/3600,
 'total_P3_GPU_hours_including_active':sum(map(used,attempts))/3600,'P3C_cap_GPU_hours':3,'total_cap_GPU_hours':160,
 'active_attempts':[{'attempt_id':a['attempt_id'],'device':a['device'],'kind':a['kind']} for a in own if a['status']!='finalized'],
 'filesystem_free_bytes':shutil.disk_usage(r).free,'forecast':read(s/'forecast.json') if (s/'forecast.json').exists() else None,
 'gpu_acceptance':read(s/'gpu_acceptance.json')['status'] if (s/'gpu_acceptance.json').exists() else 'NOT_RUN',
 'independent_audit_status':audit['status'] if audit else None,'actionable_reason':reason}
if reason and reason!='completed_ready_for_recovery':
 logs=sorted(s.glob('authorized_launch_*.log'),key=lambda p:p.stat().st_mtime)
 if logs:v['log_tail']=logs[-1].read_text(errors='replace')[-3000:]
print(json.dumps(v))
'''

def main():
    client=connect()
    try:
        code,out,_=command(client,['/root/miniconda3/envs/tl_cls_v1/bin/python','-c',SCRIPT])
        assert code==0
        value=json.loads(out);p=ROOT/'records/p3c/monitor/latest.json';p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
        print(json.dumps(value,ensure_ascii=False))
    finally:client.close()

if __name__=='__main__':
    try:main()
    except Exception as error:
        print('P3C_poll_failed:'+type(error).__name__)
        raise SystemExit(1)
