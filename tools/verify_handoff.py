"""验证 GitHub 交接副本的原始字节、训练源码身份、相对链接和阶段状态。"""

import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
TRAINING_COMMIT = '94b02df2ae94097602e40e88783f8c445006fa9a'


def main():
    manifest = json.loads((ROOT / 'reports/delivery_manifest.json').read_text(encoding='utf-8'))
    for entry in manifest['files']:
        path = (ROOT / entry['path']).resolve()
        assert path.is_relative_to(ROOT) and path.is_file(), entry['path']
        assert path.stat().st_size == entry['size_bytes'], entry['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], entry['path']
    # 将训练源码与其真实历史 commit 的 blob 比对，不能用新的归档 HEAD 替代科学身份。
    names = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', TRAINING_COMMIT,
                                     '--', 'src', 'configs'], cwd=ROOT).decode().splitlines()
    for name in names:
        expected = subprocess.check_output(['git', 'show', f'{TRAINING_COMMIT}:{name}'], cwd=ROOT)
        assert (ROOT / name).read_bytes() == expected, name
    status = json.loads((ROOT / 'records/pi_review_status.json').read_text(encoding='utf-8'))
    assert status['status'] == 'awaiting_pi_review' and status['pi_decision'] is None
    assert all(status[p]['authorized'] is False and status[p]['status'] == 'not_started' for p in ['P2', 'P3'])
    count = 0
    for relative in ['README.md', 'docs/PI_审核与阶段分析.md', 'docs/PI_审核决定模板.md']:
        path = ROOT / relative
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            assert (path.parent / unquote(target.split('#')[0])).exists(), (relative, target)
            count += 1
    print(json.dumps({'status': 'passed', 'original_delivery_files_verified': len(manifest['files']),
                      'original_training_source_and_config_files_verified': len(names),
                      'review_document_relative_links_verified': count,
                      'P2_P3_authorized': False}))


if __name__ == '__main__':
    main()
