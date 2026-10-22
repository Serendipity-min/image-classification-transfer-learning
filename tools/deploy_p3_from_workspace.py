"""部署 P3 源码；仅推进 Git 身份和源码，保留服务器 P1/P2 原始运行文件。"""
import json
from pathlib import Path
import subprocess
from remote_session import connect
from transport import ROOT, REMOTE, command, upload

SOURCE = ROOT / 'p3_source'


def main():
    assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=SOURCE)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    bundle = ROOT / 'records/p3' / ('source_' + commit[:12] + '.bundle')
    bundle.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['git', 'bundle', 'create', str(bundle), 'codex/p3-execution',
                    '^d66193bc3265954094e0badebfcd3279deb69e5a'], cwd=SOURCE, check=True)
    client = connect()
    try:
        relative = 'records/p3/' + bundle.name
        upload(client, bundle, relative)
        script = r'''
import hashlib,json,pathlib,subprocess,sys
r=pathlib.Path('/root/user_data/tl_cls_v1'); target=sys.argv[1]; bundle=sys.argv[2]
def git(*args): return subprocess.check_output(['git','-C',str(r),*args],text=True).strip()
before=git('rev-parse','HEAD')
old_ledger=r/'records/p2/gpu_ledger.json'; old_sha=hashlib.sha256(old_ledger.read_bytes()).hexdigest()
git('fetch',bundle,'codex/p3-execution')
assert git('rev-parse','FETCH_HEAD')==target
subprocess.run(['git','-C',str(r),'merge-base','--is-ancestor',before,target],check=True)
jobs=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not jobs,'deployment requires no active GPU jobs'
# read-tree 只更新索引，不覆盖任何现场回执、ledger 或 checkpoint；源码按目标 Git 字节发布。
git('read-tree',target)
# NUL 分隔避免中文路径被 Git core.quotepath 转义成不可执行的文件名。
names=git('ls-tree','-rz','--name-only',target,'src','tools','configs','docs','ops','README.md','.gitignore','.gitattributes').split('\0')
names=[name for name in names if name]
for name in names:
    blob=subprocess.check_output(['git','-C',str(r),'show',target+':'+name])
    p=r/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(blob)
git('update-ref',git('symbolic-ref','HEAD'),target,before)
assert hashlib.sha256(old_ledger.read_bytes()).hexdigest()==old_sha
print(json.dumps({'execution_commit':target,'previous_commit':before,'P2_ledger_preserved_sha256':old_sha,
                  'runtime_evidence_overwritten':False,'source_files_published':len(names)}))
'''
        code, output, error = command(client, ['/root/miniconda3/envs/tl_cls_v1/bin/python', '-c',
            script, commit, str(REMOTE / relative)], timeout=60)
        assert code == 0, error
        print(output.strip())
    finally: client.close()


if __name__ == '__main__': main()
