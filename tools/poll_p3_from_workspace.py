"""P3 一次性受控状态检查：不启动训练、不恢复任务、不读取官方测试集。"""
import json
import sys
from remote_session import connect
from transport import ROOT, command
SCRIPT = """import json,sys,time
from pathlib import Path
r=Path('/root/user_data/tl_cls_v1')
sys.path.insert(0,str(r/'src'))
from tlcls.p3_core import process_alive
s=r/'records/p3'
def read(p):return json.loads(p.read_text()) if p.exists() else None
ledger=read(s/'gpu_ledger.json') or {'attempts':{}}
now=time.monotonic()
used=sum(a.get('charged_gpu_seconds',0) if a['status']=='finalized' else max(0,now-a['started_monotonic']) for a in ledger['attempts'].values())
active=[]
for a in ledger['attempts'].values():
 if a['status']=='finalized':continue
 d=r/'runs'/a['run_id']
 active.append({'run_id':a['run_id'],'attempt_id':a['attempt_id'],'device':a['device'],'kind':a['kind'],
  'progress':read(d/'progress.json'),'summary':{k:v for k,v in (read(d/'summary.json') or {}).items() if k in ['status','completed_epochs','peak_allocated_bytes','peak_reserved_bytes']}})
statuses={}
for p in (r/'runs').glob('P3-*/summary.json'):
 v=read(p);statuses[v['run_id']]=v['status']
parent=read(s/'scheduler.json') or {}
result={'scheduler_alive':process_alive(parent['pid'],parent['start_ticks']) if parent else False,
'gpu_hours_including_active':used/3600,'cap_gpu_hours':160,'acceptance':(read(s/'acceptance.json') or {}).get('status'),
'cpu_acceptance':(read(s/'cpu_acceptance.json') or {}).get('status'),'gpu_acceptance':(read(s/'gpu_acceptance.json') or {}).get('status'),
'completed':sum(x=='completed' for x in statuses.values()),'numerical_failures':[k for k,v in statuses.items() if v=='numerical_failure'],
'other_failures':{k:v for k,v in statuses.items() if v not in ['completed','running','numerical_failure']},
'active':active,'stage_progress':read(s/'progress.json'),'stage_status':read(s/'stage_status.json'),'monitor':read(s/'monitor/latest.json'),
'storage_latest':read(s/'storage_latest.json'),'sealed':(r/'records/P3_sealed.json').exists()}
from collections import Counter
result['run_status_counts']=dict(Counter(statuses.values()))
result['ledger_settled']=all(a['status']=='finalized' for a in ledger['attempts'].values())
if not result['scheduler_alive'] and not result['acceptance']:
 logs=sorted((s/'logs').glob('*.log'),key=lambda p:p.stat().st_mtime)
 if logs:result['latest_log_tail']=logs[-1].read_text(errors='replace')[-3500:]
print(json.dumps(result))
"""

def main():
    client=connect()
    try:
        code,output,error=command(client,['/root/miniconda3/envs/tl_cls_v1/bin/python','-c',SCRIPT])
        assert code==0
        value=json.loads(output)
        path=ROOT/'records/p3/monitor/latest.json';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
        print(json.dumps(value,ensure_ascii=False))
    finally:client.close()

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('P3_poll_failed:'+type(exc).__name__,file=sys.stderr)
        raise SystemExit(1)
