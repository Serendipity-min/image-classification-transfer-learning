"""P3 独立身份、授权与固定矩阵；P2 归档和预算始终只读。"""
from collections import defaultdict
import subprocess
from .p2_core import (ROOT, PIPELINE, atomic_bytes, canonical, code_commit, file_hash,
                      fsync_directory, infrastructure_exception, monotonic, process_alive,
                      process_ticks, read, value_hash, write)

STAGE = ROOT / 'records/p3'
CAP_SECONDS = 160 * 3600
EXIT_RESERVE_SECONDS = 60
SEEDS = [17, 42, 2026, 3407, 65537]
ARCHIVE = '987fb5a0a5f7505086e2a7bbb64d421cf7f336ad'


def execution_identity():
    # 把所有间接导入的事务与持久化原语也纳入身份，不能仅比较训练入口。
    names = PIPELINE + ['src/evaluate.py', 'src/tlcls/p2_core.py',
                        'src/tlcls/p2_transaction.py']
    names += [str(p.relative_to(ROOT)).replace('\\', '/') for pattern in
              ['src/p3_*.py', 'src/tlcls/p3_*.py'] for p in ROOT.glob(pattern)]
    files = {n: file_hash(ROOT / n) for n in sorted(set(names))}
    return {'files': files, 'scope_sha256': value_hash(files),
            'mathematical_pipeline_sha256': value_hash({n: files[n] for n in PIPELINE})}


def operational_compatibility():
    # P3 从新身份启动；没有默许恢复旧 P2 或跨代码提交的兼容映射。
    return {'phase': 'P3', 'cross_commit_resume_allowed': False,
            'actual_execution_identity': execution_identity()}


def candidates():
    manifest = read(STAGE / 'run_manifest.json')
    assert manifest['schema'] == 'P3_runs_v1'
    selected_sha = file_hash(ROOT / 'selected_hparams.json')
    selected = read(ROOT / 'selected_hparams.json')['configurations']
    assets = {a['key']: a for a in read(ROOT / 'records/assets.json')['assets']}
    runs, groups = manifest['runs'], defaultdict(list)
    original = {r['run_id']: r for r in read(ROOT / 'configs/run_matrix.json')['planned_runs'] if r['phase'] == 'P3'}
    assert len(runs) == len({r['run_id'] for r in runs}) == 80
    assert {r['run_id'] for r in runs} == set(original)
    for r in runs:
        assert r['phase'] == 'P3' and r['formal_run'] and r['epochs'] == 50
        assert r['seed'] in SEEDS and r['gpu_count'] == 1
        assert r['scheduler_horizon_epochs'] == 50 and r['warmup_epochs'] == 5
        assert (r['micro_batch_size'], r['gradient_accumulation_steps'], r['effective_batch_size']) == (32, 4, 128)
        assert r['autocast_dtype'] == 'bfloat16' and r['test_during_training'] is False
        assert r['train_fraction'] == 1 and r['split_seed'] == 20261020
        assert r['head_lr_multiplier'] == (10 if r['method'] == 'full_ft' else 1)
        assert r['selected_hparams_sha256'] == selected_sha
        assert r['learning_rate'] == selected[r['configuration_id']]['learning_rate']
        assert r['resume_pilot_checkpoint'] is False
        assert r['initialization'] == 'fresh_original_backbone_or_random_scratch_plus_matched_head'
        assert r['checkpoint_revision'] == assets[r['model_key'] + '_model.safetensors']['revision']
        assert r['checkpoint_sha256'] == assets[r['model_key'] + '_model.safetensors']['sha256']
        assert r['physical_gpu_index'] in [0, 1]
        # 原矩阵中允许解析的字段有明确范围，其余科学字段必须完全相等。
        for k, v in original[r['run_id']].items():
            if k not in {'status', 'learning_rate', 'checkpoint_revision', 'checkpoint_sha256',
                         'budget_required_at', 'provenance_required_at', 'gpu_hour_budget'}:
                assert r[k] == v, k
        groups[r['configuration_id']].append(r['seed'])
    assert len(groups) == 16 and all(sorted(v) == sorted(SEEDS) for v in groups.values())
    return runs


def scientific_identity(config):
    frozen = ['configs/protocol.json', 'configs/run_matrix.json', 'records/dependencies.lock',
              'selected_hparams.json', 'records/assets.json', 'records/p3/time_budget_frozen.json',
              'records/' + config['dataset'] + '_split.json']
    identity = {'configuration': {k: v for k, v in config.items() if k != 'physical_gpu_index'},
                'inputs': {n: file_hash(ROOT / n) for n in frozen},
                'implementation': execution_identity(), 'weight_revision': config['checkpoint_revision'],
                'weight_sha256': config['checkpoint_sha256']}
    return {'scientific_config': identity, 'scientific_config_sha256': value_hash(identity)}


def require_authorization(require_acceptance=True):
    auth = read(STAGE / 'authorization.json')
    assert auth['schema'] == 'P3_authorization_v1' and auth['P3_A_authorized'] is True
    assert auth['contract_id'] == 'TL-CLS-P3-GATE-20261022-v1.0'
    assert auth['centralized_test_authorized'] is False and auth['P3_C_authorized'] is False
    assert auth['hard_cap_gpu_seconds'] == CAP_SECONDS
    assert auth['confirmed_project_environment_allocation_bytes'] >= 200 * 2**30
    assert auth['allocation_evidence']['kind'] == 'explicit_user_contract_resource_confirmation'
    assert auth['execution_commit'] == code_commit() and auth['execution_identity'] == execution_identity()
    # 绑定树内实际字节与 Git 对象，防止未提交补丁沿用旧验收收据。
    for name, expected in auth['execution_identity']['files'].items():
        import hashlib
        blob = subprocess.check_output(['git', '-C', str(ROOT), 'show', code_commit() + ':' + name])
        assert hashlib.sha256(blob).hexdigest() == expected
    for n, expected in auth['frozen_inputs'].items():
        assert file_hash(ROOT / n) == expected, '冻结输入改变: ' + n
    if require_acceptance:
        receipt = read(STAGE / 'acceptance.json')
        assert receipt['status'] == 'passed' and receipt['six_gates_passed'] is True
        assert receipt['execution_commit'] == code_commit() and receipt['execution_identity'] == execution_identity()
        assert receipt['authorization_sha256'] == file_hash(STAGE / 'authorization.json')
        for key, filename in [('cpu_receipt_sha256', 'cpu_acceptance.json'),
                              ('gpu_receipt_sha256', 'gpu_acceptance.json'),
                              ('entry_preflight_sha256', 'entry_preflight.json')]:
            assert receipt[key] == file_hash(STAGE / filename)
        assert read(STAGE / 'dispatch_authorization.json')['acceptance_sha256'] == file_hash(STAGE / 'acceptance.json')
        assert read(STAGE / 'dispatch_authorization.json')['P3_B_authorized'] is True
    return auth
