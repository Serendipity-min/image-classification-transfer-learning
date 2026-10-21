"""打包有限验收证据；按显式清单收集，排除 checkpoint、下载暂存与连接运行时。"""

import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    evidence = json.loads((ROOT / 'records/evidence_manifest.json').read_text(encoding='utf-8'))
    selected = {entry['path'] for entry in evidence['files']}
    selected.update(['records/evidence_manifest.json', 'records/source.bundle',
                     'records/acceptance_audit.json', 'records/transfers.jsonl',
                     'records/receipt_publication.json'])
    selected.update(str(p.relative_to(ROOT)).replace('\\', '/') for p in (ROOT / 'reports').rglob('*')
                    if p.is_file() and p.suffix not in ['.zip', '.sha256']
                    and p.name != 'delivery_manifest.json')
    selected.update(['ops/audit_acceptance.py', 'ops/finalize_server_receipt.py',
                     'ops/plot_profiles.py', 'ops/write_acceptance_receipt.py', 'ops/package_evidence.py',
                     'ops/publish_receipt.py'])
    selected.update(str(p.relative_to(ROOT)).replace('\\', '/') for p in
                    (ROOT / '迁移学习实验执行包_v1').rglob('*') if p.is_file())
    records = []
    for relative in sorted(selected):
        path = (ROOT / relative).resolve()
        assert path.is_relative_to(ROOT.resolve()) and path.is_file()
        # 原始证据仅限文本和明确的源代码 bundle；禁止递归收进模型或环境。
        assert path.suffix not in ['.pt', '.pth', '.safetensors', '.whl', '.asset', '.pyc']
        records.append({'path': relative, 'size_bytes': path.stat().st_size, 'sha256': digest(path)})
    manifest = {'protocol_id': 'TL-CLS-20261020-v1.0', 'files': records,
                'checkpoint_policy': '40 verified checkpoint binaries remain on server',
                'excluded': ['model binaries', 'dataset archives', 'download staging',
                             'connection runtime', 'SSH configuration', 'chat history']}
    manifest_path = ROOT / 'reports/delivery_manifest.json'
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    target = ROOT / 'reports/P0_P1_验收证据包.zip'
    with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for entry in records:
            archive.write(ROOT / entry['path'], entry['path'])
        archive.write(manifest_path, 'reports/delivery_manifest.json')
    # 回读 ZIP 的每一项与打包前 hash 比对，确保可交付文件本身也通过完整性检查。
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
        for entry in records:
            assert hashlib.sha256(archive.read(entry['path'])).hexdigest() == entry['sha256']
    checksum = digest(target)
    target.with_suffix('.zip.sha256').write_text(checksum + '  ' + target.name + '\n', encoding='utf-8')
    print(json.dumps({'archive': str(target), 'size_bytes': target.stat().st_size,
                      'evidence_files': len(records), 'sha256': checksum}, ensure_ascii=False))


if __name__ == '__main__':
    main()
