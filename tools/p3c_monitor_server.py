import sys,time,json,fcntl,os
from pathlib import Path
sys.path.insert(0,"/root/user_data/tl_cls_v1/src")
from tlcls.p3_core import write,process_ticks

def check():

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
    return v

def main():
    folder=Path('/root/user_data/tl_cls_v1/records/p3c/monitor');folder.mkdir(exist_ok=True)
    handle=(folder/'monitor.lock').open('a');fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(folder/'registration.json',{'pid':os.getpid(),'start_ticks':process_ticks(os.getpid()),'interval_seconds':1800})
    deadline=time.monotonic()+24*3600
    while time.monotonic()<deadline:
        try:value=check()
        except Exception as error:value={'actionable_reason':'monitor_read_failed','error_type':type(error).__name__}
        write(folder/'latest.json',value)
        with (folder/'checks.jsonl').open('a') as f:f.write(json.dumps(value,ensure_ascii=False)+'\n')
        if value.get('actionable_reason'):
            write(folder/'attention.json',value);return
        time.sleep(1800)
if __name__=='__main__':main()
