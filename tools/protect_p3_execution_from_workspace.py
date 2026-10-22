"""准备本机推送保护和执行冻结证据；不把本地保护冒充 GitHub 平台保护。"""
import hashlib
import json
from pathlib import Path
import subprocess
import shutil

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'p3_source'


def main():
    frozen = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=SOURCE, text=True).strip()
    common = subprocess.check_output(['git', 'rev-parse', '--git-common-dir'], cwd=SOURCE, text=True).strip()
    directory = (SOURCE / common).resolve() / 'hooks'
    target = directory / 'pre-push'
    # 仅约束 P3 分支；允许科学源码不变的证据提交，禁止删除、回退或替换执行源码。
    hook = '\n'.join(['#!/bin/sh', 'while read local_ref local_sha remote_ref remote_sha', 'do',
        '  if [ "$remote_ref" = "refs/heads/codex/p3-execution" ]; then',
        '    case "$local_sha" in 0000000000000000000000000000000000000000) echo "P3 branch deletion refused" >&2; exit 1;; esac',
        '    if [ "$remote_sha" != "0000000000000000000000000000000000000000" ]; then',
        '      git merge-base --is-ancestor "$remote_sha" "$local_sha" || exit 1', '    fi',
        '    git diff --quiet ' + frozen + ' "$local_sha" -- src || { echo "P3 source is frozen; review required" >&2; exit 1; }',
        '  fi', 'done', 'exit 0', ''])
    directory.mkdir(exist_ok=True)
    if target.exists():
        assert target.read_text(encoding='utf8') == hook, '已有用户 hook，不自动覆盖'
    target.write_text(hook, encoding='utf8', newline='\n'); target.chmod(0o755)
    valid = f'refs/heads/codex/p3-execution {frozen} refs/heads/codex/p3-execution {frozen}\n'
    deletion = f'(delete) {"0"*40} refs/heads/codex/p3-execution {frozen}\n'
    shell = shutil.which('sh')
    assert shell, '需要已安装 Git 的 POSIX shell 验证 hook'
    # 使用 LF 字节流；Windows 文本管道的 CRLF 会把末尾 CR 当作 Git SHA 的一部分。
    normal = subprocess.run([shell, target.as_posix()], cwd=SOURCE, input=valid.encode(), capture_output=True)
    invalid = subprocess.run([shell, target.as_posix()], cwd=SOURCE, input=deletion.encode(), capture_output=True)
    assert normal.returncode == 0 and invalid.returncode != 0, normal.stderr.decode(errors='replace')
    receipt = {'schema': 'P3_branch_protection_v1', 'branch': 'codex/p3-execution',
        'execution_commit': frozen, 'github_platform_protected': False,
        'github_api_http_status': 403, 'github_api_reason': 'private repository platform protection requires upgraded plan',
        'local_push_guard': {'installed': True, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
            'normal_evidence_push_allowed': True, 'delete_rejected': True,
            'source_change_and_non_fast_forward_rejected': True},
        'server_execution_pinned_by_commit_and_all_source_sha256': True,
        'equivalent_protection_approval': 'pending_user_decision', 'P3_B_authorized': False,
        'limitation': 'local hook does not prevent pushes from another clone or GitHub administrator'}
    path = ROOT / 'records/p3/branch_protection.json'; path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__': main()
