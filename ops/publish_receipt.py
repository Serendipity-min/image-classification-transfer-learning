"""将派生回执送回同一服务器项目；沿用暂存、哈希核验和精确清理规则。"""

import json
from pathlib import Path

from remote_session import connect
from transport import ROOT, upload


if __name__ == '__main__':
    files = [ROOT / 'reports/P0_P1_验收回执.md', ROOT / 'reports/P1_acceptance_table.csv',
             ROOT / 'reports/figure_source_preflight.json', ROOT / 'records/acceptance_audit.json']
    files += sorted(p for p in (ROOT / 'reports/figures').rglob('*') if p.is_file())
    operators = ['audit_acceptance.py', 'plot_profiles.py', 'write_acceptance_receipt.py',
                 'package_evidence.py', 'publish_receipt.py']
    targets = [(p, p.relative_to(ROOT).as_posix()) for p in files]
    # 操作附录置于 records，保持已验收的 src/configs Git 身份不变。
    targets += [(ROOT / 'ops' / name, 'records/operators/' + name) for name in operators]
    client = connect()
    try:
        receipts = [upload(client, source, destination) for source, destination in targets]
        publication = {'status': 'all_uploaded_sha256_verified_staging_cleaned', 'files': receipts,
                       'checkpoint_transfers': 0, 'core_source_changed': False}
        path = ROOT / 'records/receipt_publication.json'
        path.write_text(json.dumps(publication, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        upload(client, path, 'records/receipt_publication.json')
        print(json.dumps({'published_files': len(receipts) + 1,
                          'published_bytes': sum(r['size_bytes'] for r in receipts),
                          'checkpoint_transfers': 0}))
    finally:
        client.close()
