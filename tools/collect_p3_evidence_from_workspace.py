"""回收 P3 小型文本证据的不可变快照；二进制 checkpoint 始终留服务器。"""
import hashlib
import io
import json
from pathlib import Path
import tarfile
import uuid
from remote_session import connect
from transport import ROOT, command

SCRIPT = r'''
import hashlib,json,pathlib,tarfile,time,sys
r=pathlib.Path('/root/user_data/tl_cls_v1');sys.path.insert(0,str(r/'src'))
from tlcls.p3_core import read,write,execution_identity,code_commit
export=r/'records/p3_exports'/sys.argv[1]; snap=export/'snapshot';snap.mkdir(parents=True)
paths=set()
for p in (r/'records/p3').rglob('*'):
 if p.is_file() and p.suffix in ['.json','.jsonl','.log']: paths.add(p)
for p in (r/'runs').glob('P3-*/**/*'):
 if p.is_file() and p.suffix in ['.json','.jsonl','.log']: paths.add(p)
for n in execution_identity()['files']: paths.add(r/n)
for n in ['selected_hparams.json','configs/protocol.json','configs/run_matrix.json','records/assets.json',
          'records/dependencies.lock','records/cifar100_split.json','records/svhn_split.json',
          'reports/P3_proposal.json','records/P3_sealed.json','reports/P3_阶段交接报告.md','reports/P3_run_results.csv']:
 p=r/n
 if p.exists():paths.add(p)
files={};total=0
for p in sorted(paths):
 data=p.read_bytes();total+=len(data)
 assert total<100_000_000,'text evidence exceeds bounded recovery size; keep on server'
 relative=p.relative_to(r).as_posix(); target=snap/relative;target.parent.mkdir(parents=True,exist_ok=True)
 target.write_bytes(data); files[relative]={'sha256':hashlib.sha256(data).hexdigest(),'size_bytes':len(data)}
manifest={'schema':'P3_text_snapshot_v1','created_unix':time.time(),'actual_execution_commit':code_commit(),
          'files':files,'checkpoint_downloaded':False,'text_bytes':total,
          'training_may_be_active':not (r/'records/P3_sealed.json').exists()}
write(snap/'snapshot_manifest.json',manifest)
archive=export/'evidence.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for p in sorted(snap.rglob('*')):
  if p.is_file():tar.add(p,arcname=p.relative_to(snap).as_posix())
print(json.dumps({'archive':str(archive),'size_bytes':archive.stat().st_size,
 'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files':len(files),'text_bytes':total}))
'''


def main():
    identity = uuid.uuid4().hex
    client = connect()
    try:
        code, output, error = command(client, ['/root/miniconda3/envs/tl_cls_v1/bin/python', '-c', SCRIPT, identity], timeout=60)
        assert code == 0, error
        receipt = json.loads(output)
        assert receipt['size_bytes'] < 100_000_000
        folder = ROOT / 'p3_delivery' / identity
        folder.mkdir(parents=True)
        archive = folder / 'evidence.tar.gz'
        with client.open_sftp() as sftp: sftp.get(receipt['archive'], str(archive))
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == receipt['sha256']
        destination = folder / 'snapshot'; destination.mkdir()
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                path = (destination / member.name).resolve()
                assert path.is_relative_to(destination.resolve()) and member.isfile()
                assert not member.name.endswith(('.pt', '.pth', '.safetensors'))
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source: path.write_bytes(source.read())
        manifest = json.loads((destination / 'snapshot_manifest.json').read_text(encoding='utf8'))
        for name, item in manifest['files'].items():
            data = (destination / name).read_bytes()
            assert len(data) == item['size_bytes'] and hashlib.sha256(data).hexdigest() == item['sha256']
        receipt.update(local_snapshot=str(destination), verified=True, checkpoint_downloaded=False)
        (ROOT / 'records/p3').mkdir(exist_ok=True)
        (ROOT / 'records/p3/latest_collection.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
        print(json.dumps(receipt, ensure_ascii=False))
    finally: client.close()


if __name__ == '__main__': main()
