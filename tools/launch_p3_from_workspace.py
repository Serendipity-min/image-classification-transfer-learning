"""平台保护及真实验收全部通过后，脱离 SSH 启动原路径正式训练与低频服务器检查。"""
import json
import subprocess
from remote_session import connect
from transport import ROOT, command


SCRIPT = r'''
import datetime,json,pathlib,subprocess,sys,time
r=pathlib.Path('/root/user_data/tl_cls_v1');sys.path.insert(0,str(r/'src'));s=r/'records/p3'
from tlcls.p3_core import read,write,require_authorization,process_alive,process_ticks,file_hash,code_commit
from tlcls.p3_budget import Storage
auth=require_authorization()
protection=read(s/'branch_protection.json');assert protection['github_platform_protected']
assert protection['execution_commit']==code_commit()
assert not any((r/'runs').glob('P3-*/scientific.json')),'formal phase already initialized; do not implicitly resume'
old=read(s/'scheduler.json');assert not process_alive(old['pid'],old['start_ticks'])
ledger=read(s/'gpu_ledger.json');assert all(a['status']=='finalized' for a in ledger['attempts'].values())
resources=Storage(auth['confirmed_project_environment_allocation_bytes']).require(4*2**30)
assert not resources['warning_175_gib']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
write(s/'launch_preflight.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'execution_commit':code_commit(),'storage':resources,'branch_protection_sha256':file_hash(s/'branch_protection.json'),
    'acceptance_sha256':file_hash(s/'acceptance.json'),'published_entry_commit':sys.argv[1],
    'P3_B_authorized':True,'P3_C_authorized':False,'centralized_test_authorized':False})
log=(s/'formal_scheduler.log').open('ab')
p=subprocess.Popen([sys.executable,str(r/'src/p3_schedule.py')],cwd=r,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
write(s/'launch_registration.json',{'scheduler_pid':p.pid,'start_ticks':process_ticks(p.pid),
    'execution_commit':code_commit(),'started_unix':time.time(),'project_root':str(r),
    'formal_runs':80,'epoch_horizon':50,'centralized_test_authorized':False})
print(json.dumps({'started':True,'scheduler_pid':p.pid,'execution_commit':code_commit(),
                  'published_entry_commit':sys.argv[1],'project_root':str(r)}))
'''


def main():
    source = ROOT / 'p3_source'
    published = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    remote_head = subprocess.check_output(['git', '-c', 'core.sshCommand=ssh -o BatchMode=yes -o ConnectTimeout=15',
        'ls-remote', 'origin', 'refs/heads/codex/p3-execution'], cwd=source, text=True).split()[0]
    assert published == remote_head, '启动前必须先推送可审查入口与真实验收'
    protection = json.loads(subprocess.check_output(['gh', 'api',
        'repos/Serendipity-min/image-classification-transfer-learning/branches/codex%2Fp3-execution/protection'], text=True))
    assert protection['enforce_admins']['enabled']
    assert not protection['allow_force_pushes']['enabled'] and not protection['allow_deletions']['enabled']
    client = connect()
    try:
        code, output, error = command(client, ['/root/miniconda3/envs/tl_cls_v1/bin/python', '-c', SCRIPT, published])
        assert code == 0, error
        print(output.strip())
    finally: client.close()


if __name__ == '__main__': main()
