"""核对 P2 运行代码与真实 CPU/GPU 验收绑定；不启动任何实验。"""

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt = load('records/p2/acceptance.json')
    authorization = load('records/p2/authorization.json')
    assert receipt['status'] == 'passed' and authorization['P2_authorized']
    assert not authorization['P3_authorized'] and not authorization['centralized_test_authorized']
    assert authorization['hard_cap_gpu_seconds'] == 30 * 3600
    assert receipt['authorization_sha256'] == digest(ROOT / 'records/p2/authorization.json')
    for filename, key in [('cpu_acceptance.json', 'cpu_receipt_sha256'),
                          ('gpu_acceptance.json', 'gpu_receipt_sha256')]:
        assert receipt[key] == digest(ROOT / 'records/p2' / filename)
        child = load('records/p2/' + filename)
        assert child['status'] == 'passed' and child['execution_commit'] == receipt['execution_commit']
        assert child['execution_identity'] == receipt['execution_identity']
    for relative, expected in receipt['execution_identity']['files'].items():
        assert digest(ROOT / relative) == expected, relative
        archived = subprocess.check_output(['git', 'show', receipt['execution_commit'] + ':' + relative], cwd=ROOT)
        assert hashlib.sha256(archived).hexdigest() == expected, relative
    print(json.dumps({'status': 'passed', 'execution_commit': receipt['execution_commit'],
        'verified_execution_files': len(receipt['execution_identity']['files']),
        'cpu_tests_passed': load('records/p2/cpu_acceptance.json')['tests_run'], 'P3_authorized': False}))


if __name__ == '__main__':
    main()
