"""回收 P3-C 小型文本快照；密钥、原始预测和模型均不进入归档。"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import uuid
from remote_session import connect
from transport import ROOT,command

SCRIPT=r'''
import pathlib,json,sys,tarfile,time,hashlib
r=pathlib.Path('/root/user_data/tl_cls_v1');sys.path.insert(0,str(r/'src'))
from tlcls.p3_core import read,write,process_alive
s=r/'records/p3c';launch_only=sys.argv[2]=='launch'
ledger=read(r/'records/p3/gpu_ledger.json');own=[a for a in ledger['attempts'].values() if a['kind'].startswith('p3c_')]
if not launch_only:
 assert all(a['status']=='finalized' for a in own)
 launch=read(s/'direct_PI_launch.json');assert not process_alive(launch['pid'],launch['start_ticks'])
paths=set()
if launch_only:
 for name in ['direct_PI_launch.json','PI_direct_approval.json','fault_injection.json','fault_injection.log',
  'execution_order.json','independent_authorization_receipt.json','gpu_acceptance.json']:
  if (s/name).exists():paths.add(s/name)
 for name in ['status.json','evaluation_ledger.json']:
  if (s/name).exists():paths.add(s/name)
 paths.add(r/'records/p3/gpu_ledger.json')
else:
 for p in s.rglob('*'):
  relative=p.relative_to(s)
  if any(x in relative.parts for x in ['preparation_history','authorization_history','monitor']):continue
  if p.is_file() and p.suffix in ['.json','.jsonl','.log'] and p.name!='runtime_credential_reference.json':paths.add(p)
 for p in (r/'reports').glob('P3_C*'):
  if p.is_file() and p.suffix in ['.json','.csv','.md']:paths.add(p)
 paths.add(r/'records/p3/gpu_ledger.json')
export=r/'records/p3c_exports'/sys.argv[1];snapshot=export/'snapshot';snapshot.mkdir(parents=True)
files={};total=0
for p in sorted(paths):
 data=p.read_bytes();total+=len(data);assert total<100_000_000
 name=p.relative_to(r).as_posix();target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
 files[name]={'sha256':hashlib.sha256(data).hexdigest(),'size_bytes':len(data)}
write(snapshot/'snapshot_manifest.json',{'schema':'P3C_text_snapshot_v1','files':files,'text_bytes':total,
 'launch_only_NOT_FINAL':launch_only,'secret_or_predictions_or_checkpoint_downloaded':False,'created_unix':time.time()})
archive=export/'text_evidence.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for p in sorted(snapshot.rglob('*')):
  if p.is_file():tar.add(p,arcname=p.relative_to(snapshot).as_posix())
print(json.dumps({'archive':str(archive),'size_bytes':archive.stat().st_size,'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
 'files':len(files),'text_bytes':total,'launch_only_NOT_FINAL':launch_only}))
'''

def main():
    p=argparse.ArgumentParser();p.add_argument('--launch-only',action='store_true');args=p.parse_args()
    identity=uuid.uuid4().hex;client=connect()
    try:
        code,out,_=command(client,['/root/miniconda3/envs/tl_cls_v1/bin/python','-c',SCRIPT,identity,
            'launch' if args.launch_only else 'final'],timeout=180)
        assert code==0
        receipt=json.loads(out);assert receipt['size_bytes']<100_000_000
        folder=ROOT/'p3c_delivery'/identity;folder.mkdir(parents=True)
        archive=folder/'text_evidence.tar.gz'
        with client.open_sftp() as sftp:sftp.get(receipt['archive'],str(archive))
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==receipt['sha256']
        snapshot=folder/'snapshot'
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                target=(snapshot/member.name).resolve();assert target.is_relative_to(snapshot.resolve()) and member.isfile()
                assert target.suffix not in ['.key','.npz','.pt','.pth','.safetensors']
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(member) as source:target.write_bytes(source.read())
        manifest=json.loads((snapshot/'snapshot_manifest.json').read_text(encoding='utf8'))
        for name,item in manifest['files'].items():
            data=(snapshot/name).read_bytes();assert len(data)==item['size_bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
        receipt.update(local_snapshot=str(snapshot),verified=True,raw_predictions_downloaded=False,key_downloaded=False)
        (ROOT/'records/p3c/latest_collection.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
        print(json.dumps(receipt,ensure_ascii=False))
    finally:client.close()

if __name__=='__main__':
    try:main()
    except Exception as error:
        print('P3C_collection_failed:'+type(error).__name__)
        raise SystemExit(1)
