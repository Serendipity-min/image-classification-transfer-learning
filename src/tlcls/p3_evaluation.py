"""集中测试严格入口：无独立签名授权、完整封存或冻结清单，立即拒绝。"""
import hashlib
import hmac
import os
from pathlib import Path
from .p3_core import ROOT, STAGE, canonical, file_hash, read, value_hash


def evaluation_id(item):
    return 'eval-' + value_hash({k: item.get(k) for k in
        ['run_id', 'evaluation_role', 'budget_fraction', 'budget_seconds', 'checkpoint_sha256']})


def cache_key(item):
    return value_hash({k: item[k] for k in
        ['run_id', 'checkpoint_sha256', 'preprocessing_identity', 'precision_identity']})


def require_test_authorization(manifest_path, root=None):
    root = Path(root or ROOT).resolve()
    try:
        # 此密钥必须由单独批准后的外部签名者注入；训练进程不拥有自动签发测试授权的权限。
        auth = read(root / 'records/p3/test_authorization.json')
        assert auth['schema'] == 'P3_C_authorization_v1'
        assert auth['centralized_test_authorized'] is True and auth['separate_user_approval'] is True
        secret = os.environ.get('TLCLS_TEST_APPROVAL_HMAC_KEY', '').encode('utf-8')
        assert len(secret) >= 32
        signature = auth['signature_hmac_sha256']
        unsigned = {k: v for k, v in auth.items() if k != 'signature_hmac_sha256'}
        assert hmac.compare_digest(signature, hmac.new(secret, canonical(unsigned), hashlib.sha256).hexdigest())
        path = Path(manifest_path).resolve()
        assert path.is_relative_to(root)
        seal_path = root / 'records/P3_sealed.json'
        seal = read(seal_path)
        assert seal['schema'] == 'P3_seal_v1' and seal['status'] == 'sealed'
        assert auth['seal_sha256'] == file_hash(seal_path)
        assert auth['evaluation_manifest_sha256'] == seal['evaluation_manifest_sha256'] == file_hash(path)
        assert auth['selected_hparams_sha256'] == seal['selected_hparams_sha256'] == file_hash(root / 'selected_hparams.json')
        assert auth['execution_identity_sha256'] == seal['execution_identity_sha256']
        runs = read(root / 'records/p3/run_manifest.json')['runs']
        formal_ids = {r['run_id'] for r in runs}
        assert len(runs) == len(formal_ids) == 80 and set(seal['run_ids']) == formal_ids
        for run_id in formal_ids:
            summary_path = root / 'runs' / run_id / 'summary.json'
            assert seal['summary_sha256'][run_id] == file_hash(summary_path)
            assert read(summary_path)['status'] in ['completed', 'numerical_failure', 'oom', 'stalled', 'invalid_protocol']
        manifest = read(path)
        assert manifest['schema'] == 'P3_evaluation_manifest_v1'
        ids = [evaluation_id(x) for x in manifest['evaluations']]
        assert len(ids) == len(set(ids))
        for item, identity in zip(manifest['evaluations'], ids):
            assert item['run_id'] in formal_ids and item['evaluation_id'] == identity
            assert item['inference_cache_key'] == cache_key(item)
            checkpoint = (root / item['checkpoint']).resolve()
            assert checkpoint.is_relative_to(root) and file_hash(checkpoint) == item['checkpoint_sha256']
        return manifest
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as exc:
        # 不返回半解析 manifest，且不把外部签名材料写入异常或日志。
        raise PermissionError('centralized test remains closed: independent authorization/seal/manifest required') from None
