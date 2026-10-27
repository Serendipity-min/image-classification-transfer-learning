"""只读复核封存、历史 checkpoint 和进程；不读取测试数据或变更台账。"""
import json
from pathlib import Path
from remote_session import connect
from transport import command, pull

SCRIPT = r'''
import hashlib,json,pathlib,sys,time,subprocess
r=pathlib.Path('/root/user_data/tl_cls_v1');sys.path.insert(0,str(r/'src'))
from tlcls.p3_core import file_hash,read,write,execution_identity,code_commit,process_alive
s=r/'records/p3';seal=read(r/'records/P3_sealed.json');before=file_hash(s/'gpu_ledger.json')
assert seal['gpu_ledger_sha256']==before
ledger=read(s/'gpu_ledger.json');assert all(a['status']=='finalized' for a in ledger['attempts'].values())
parent=read(s/'scheduler.json');assert not process_alive(parent['pid'],parent['start_ticks'])
for name,key in [('evaluation_manifest.draft.json','evaluation_manifest_sha256'),('run_manifest.json','run_manifest_sha256'),('checkpoint_inventory.json','checkpoint_inventory_sha256')]:
 assert file_hash(s/name)==seal[key]
assert not seal['P3_C_authorized'] and not seal['centralized_test_authorized']
p1=read(r/'records/checkpoint_inventory.json')['files'];p2=read(r/'records/p2/checkpoint_inventory.json')['P2_files']
for item in p1+p2:
 p=r/item['path'];assert p.stat().st_size==item['size_bytes'] and file_hash(p)==item['sha256'],item['path']
assert file_hash(r/'records/p2/gpu_ledger.json')=='528ca97a91bf3092b2b9782be4b23f74d1f507a2ff23e1c1cbb9228dfa24d18c'
inventory=read(s/'checkpoint_inventory.json')
for item in inventory['items']:
 assert (r/item['path']).stat().st_size==item['size_bytes']
for name,sha in execution_identity()['files'].items():
 assert hashlib.sha256(subprocess.check_output(['git','-C',str(r),'show',code_commit()+':'+name])).hexdigest()==sha
assert before==file_hash(s/'gpu_ledger.json')
v={'schema':'P3_server_closeout_verification_v1','status':'passed','checked_unix':time.time(),
 'execution_commit':code_commit(),'execution_identity':execution_identity(),
 'seal_sha256':file_hash(r/'records/P3_sealed.json'),'gpu_ledger_sha256':before,
 'all_attempts_settled':True,'scheduler_alive':False,'P1_checkpoint_hashes_verified':len(p1),
 'P2_checkpoint_hashes_verified':len(p2),'P2_ledger_unchanged':True,
 'P3_checkpoint_sizes_verified':len(inventory['items']),
 'P3_checkpoint_hash_evidence':'independent p3_report terminal audit bound by seal; no checkpoint download or new GPU work',
 'P3_C_authorized':False,'centralized_test_authorized':False}
write(s/'server_closeout_verification.json',v);print(json.dumps(v))
'''

def main():
    client = connect()
    try:
        code, output, _ = command(client, ['/root/miniconda3/envs/tl_cls_v1/bin/python', '-c', SCRIPT], timeout=420)
        assert code == 0, 'server closeout verification failed'
        pull(client, 'records/p3/server_closeout_verification.json')
        print(output)
    finally:
        client.close()

if __name__ == '__main__':
    main()
