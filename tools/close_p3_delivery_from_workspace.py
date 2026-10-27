"""交付推送后暂停本项目检查，保留封存与监控证据。"""
import argparse
import json
from pathlib import Path
import re
import subprocess
import time
import tomllib
from remote_session import connect
from transport import ROOT, command, pull

SCRIPT = r'''
import json,pathlib,sys,time,os,signal
r=pathlib.Path('/root/user_data/tl_cls_v1');sys.path.insert(0,str(r/'src'))
from tlcls.p3_core import read,write,file_hash,process_alive
s=r/'records/p3';seal=read(r/'records/P3_sealed.json');ledger=read(s/'gpu_ledger.json')
assert seal['run_status_counts']=={'completed':80} and all(a['status']=='finalized' for a in ledger['attempts'].values())
assert seal['gpu_ledger_sha256']==file_hash(s/'gpu_ledger.json')
assert not seal['P3_C_authorized'] and not seal['centralized_test_authorized']
auth=read(s/'authorization.json')
assert not auth['P3_C_authorized'] and not auth['centralized_test_authorized']
assert not list((r/'runs').glob('P3-*/*_predictions.npz'))
m=read(s/'monitor/registration.json');alive=process_alive(m['pid'],m['start_ticks'])
if alive:
 cmd=pathlib.Path('/proc')/str(m['pid'])/'cmdline'
 assert b'p3_monitor_server.py' in cmd.read_bytes()
 os.kill(m['pid'],signal.SIGTERM)
 for _ in range(20):
  if not process_alive(m['pid'],m['start_ticks']):break
  time.sleep(.1)
assert not process_alive(m['pid'],m['start_ticks'])
v={'schema':'P3_delivery_closeout_v1','status':'delivered_waiting_PI_review','checked_unix':time.time(),
 'delivered_commit':sys.argv[1],'runs_completed':80,'all_attempts_settled':True,
 'server_monitor_alive':False,'server_monitor_was_already_stopped':not alive,
 'seal_sha256':file_hash(r/'records/P3_sealed.json'),'gpu_ledger_unchanged':True,
 'official_test_prediction_files':0,'authorization_still_closed':True,
 'P3_C_authorized':False,'centralized_test_authorized':False,'checkpoint_downloaded':False}
write(s/'delivery_closeout.json',v);print(json.dumps(v))
'''

def main():
    p = argparse.ArgumentParser();p.add_argument('--delivered-commit', required=True);args = p.parse_args()
    repo = ROOT / 'p3_source'
    remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/codex/p3-execution'], cwd=repo, text=True).split()[0]
    assert remote == args.delivered_commit
    client = connect()
    try:
        code, output, _ = command(client, ['/root/miniconda3/envs/tl_cls_v1/bin/python', '-c', SCRIPT, args.delivered_commit])
        assert code == 0
        pull(client, 'records/p3/delivery_closeout.json')
        value = json.loads(output)
    finally:
        client.close()
    path = Path.home() / '.codex/automations/tlcls-p3-30min/automation.toml'
    data = path.read_text(encoding='utf8');before = tomllib.loads(data)
    assert before['id'] == 'tlcls-p3-30min'
    data = re.sub(r'^status = ".*"$', 'status = "PAUSED"', data, flags=re.M)
    data = re.sub(r'^updated_at = \d+$', 'updated_at = ' + str(int(time.time()*1000)), data, flags=re.M)
    path.write_text(data, encoding='utf8', newline='\n')
    after = tomllib.loads(path.read_text(encoding='utf8'))
    assert after['status'] == 'PAUSED' and after['prompt'] == before['prompt']
    value['desktop_automation_status'] = 'PAUSED'
    value['desktop_automation_readback_verified'] = True
    value['chat_archived'] = False
    (repo / 'records/p3/delivery_closeout.json').write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf8', newline='\n')
    print(json.dumps(value))

if __name__ == '__main__':
    main()
