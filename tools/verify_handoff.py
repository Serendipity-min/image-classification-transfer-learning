"""在固定 P0/P1 Git 快照内核对交付字节；后续阶段源码变更不回填旧收据。"""

import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
TRAINING_COMMIT = '94b02df2ae94097602e40e88783f8c445006fa9a'
SNAPSHOT = 'b485177a393b8532e07d1504a64629ad54043d2b'


def blob(relative, ref=SNAPSHOT):
    return subprocess.check_output(['git', 'show', f'{ref}:{relative}'], cwd=ROOT)


def main():
    manifest = json.loads(blob('reports/delivery_manifest.json'))
    for entry in manifest['files']:
        data = blob(entry['path'])
        assert len(data) == entry['size_bytes'], entry['path']
        assert hashlib.sha256(data).hexdigest() == entry['sha256'], entry['path']
    # 当前阶段可以新增源码；P1 始终核对审核快照，不能要求当前工作树永久维持 P1 状态。
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', TRAINING_COMMIT,
                                     '--', 'src', 'configs'], cwd=ROOT).decode().splitlines()
    for name in names:
        expected = subprocess.check_output(['git', 'show', f'{TRAINING_COMMIT}:{name}'], cwd=ROOT)
        assert blob(name) == expected, name
    status = json.loads(blob('records/pi_review_status.json'))
    assert status['status'] == 'awaiting_pi_review' and status['pi_decision'] is None
    assert all(status[p]['authorized'] is False and status[p]['status'] == 'not_started' for p in ['P2', 'P3'])
    count = 0
    for relative in ['README.md', 'docs/PI_审核与阶段分析.md', 'docs/PI_审核决定模板.md']:
        path = Path(relative)
        for target in re.findall(r'\]\(([^)]+)\)', blob(relative).decode('utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            normalized = (ROOT / path.parent / unquote(target.split('#')[0])).resolve().relative_to(ROOT).as_posix()
            blob(normalized)
            count += 1
    print(json.dumps({'status': 'passed', 'original_delivery_files_verified': len(manifest['files']),
                      'original_training_source_and_config_files_verified': len(names),
                      'review_document_relative_links_verified': count,
                      'snapshot_commit': SNAPSHOT, 'P1_receipts_rewritten': False}))


if __name__ == '__main__':
    main()
