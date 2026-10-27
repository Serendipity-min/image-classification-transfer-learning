"""PI 注入密钥后后台执行；不签发授权、不把密钥保存到文件或命令行。"""
import argparse
import getpass
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/root/user_data/tl_cls_v1')

def main():
    p = argparse.ArgumentParser(); p.add_argument('--background', action='store_true'); args = p.parse_args()
    if not args.background:
        if not sys.stdin.isatty():
            raise SystemExit('PI-controlled interactive terminal required')
        assert (ROOT / 'records/p3/test_authorization.json').is_file(), 'independently signed authorization required'
        key = getpass.getpass('PI HMAC key (hidden; not saved): ')
        assert len(key.encode()) >= 32
        environment = dict(os.environ, TLCLS_TEST_APPROVAL_HMAC_KEY=key, PYTHONUNBUFFERED='1')
        log = ROOT / 'records/p3c' / ('authorized_launch_' + str(int(time.time())) + '.log')
        with log.open('ab') as output:
            process = subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--background'], cwd=ROOT,
                env=environment,stdout=output,stderr=output,stdin=subprocess.DEVNULL,start_new_session=True)
        print('P3-C supervised background PID:',process.pid)
        print('Log:',log)
        return
    assert len(os.environ.get('TLCLS_TEST_APPROVAL_HMAC_KEY','').encode()) >= 32
    command = [sys.executable,str(ROOT/'tools/p3c_supervise.py')]
    subprocess.run(command+['--synthetic'],cwd=ROOT,check=True)
    subprocess.run(command,cwd=ROOT,check=True)

if __name__ == '__main__':
    try:main()
    except Exception as error:
        print('P3C_authorized_launch_stopped:'+type(error).__name__,file=sys.stderr)
        raise SystemExit(1)
