"""离线核验 P3 启动收据与真实执行 Git 字节；不加载模型或测试数据。"""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf8'))


def sha(name):
    return hashlib.sha256((ROOT / name).read_bytes()).hexdigest()


def main():
    stage = 'records/p3/'
    receipt = read(stage + 'acceptance.json')
    auth = read(stage + 'authorization.json')
    assert receipt['status'] == 'passed' and receipt['six_gates_passed']
    assert receipt['authorization_sha256'] == sha(stage + 'authorization.json')
    assert auth['hard_cap_gpu_seconds'] == 160 * 3600
    assert auth['confirmed_project_environment_allocation_bytes'] >= 200 * 2**30
    assert not auth['P3_C_authorized'] and not auth['centralized_test_authorized']
    for filename, key in [('cpu_acceptance.json', 'cpu_receipt_sha256'),
                          ('gpu_acceptance.json', 'gpu_receipt_sha256'),
                          ('entry_preflight.json', 'entry_preflight_sha256')]:
        assert receipt[key] == sha(stage + filename)
        child = read(stage + filename)
        assert child['status'] == 'passed' and child['execution_commit'] == receipt['execution_commit']
        assert child['execution_identity'] == receipt['execution_identity']
    assert read(stage + 'cpu_acceptance.json')['tests_run'] >= 17
    for name, expected in receipt['execution_identity']['files'].items():
        assert sha(name) == expected
        blob = subprocess.check_output(['git', 'show', receipt['execution_commit'] + ':' + name], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == expected
    for name, expected in auth['frozen_inputs'].items():
        assert sha(name) == expected, name
    protection = read(stage + 'branch_protection.json')
    assert protection['github_platform_protected'] and protection['github_api_http_status'] == 200
    assert not protection['platform_protection']['allow_force_pushes']
    assert not protection['platform_protection']['allow_deletions']
    assert protection['platform_protection']['enforce_admins']
    selected = read('selected_hparams.json')['configurations']
    runs = read(stage + 'run_manifest.json')['runs']
    assets = {a['key']: a for a in read('records/assets.json')['assets']}
    assert len(runs) == len({r['run_id'] for r in runs}) == 80
    groups = {}
    for r in runs:
        assert r['epochs'] == r['scheduler_horizon_epochs'] == 50 and r['warmup_epochs'] == 5
        assert r['learning_rate'] == selected[r['configuration_id']]['learning_rate']
        assert r['checkpoint_revision'] == assets[r['model_key'] + '_model.safetensors']['revision']
        assert r['checkpoint_sha256'] == assets[r['model_key'] + '_model.safetensors']['sha256']
        assert r['resume_pilot_checkpoint'] is False and not r['test_during_training']
        assert r['selected_hparams_sha256'] == sha('selected_hparams.json')
        groups.setdefault(r['configuration_id'], []).append(r['seed'])
    assert len(groups) == 16 and all(sorted(v) == [17, 42, 2026, 3407, 65537] for v in groups.values())
    frozen = read(stage + 'time_budget_frozen.json')
    proposal = read('reports/P3_proposal.json')
    assert frozen['source_sha256'] == sha('reports/P3_proposal.json')
    for key, group in frozen['groups'].items():
        assert group['B_seconds'] == proposal['common_time_slices'][key]['B_seconds_proposal']
        assert group['slice_seconds'] == [group['B_seconds'] * f for f in [.25, .5, 1.]]
    ledger = read(stage + 'gpu_ledger.json')
    finalized = [a for a in ledger['attempts'].values() if a['status'] == 'finalized']
    assert ledger['cap_gpu_seconds'] == 576000
    assert all(a['charged_gpu_seconds'] >= 0 for a in finalized)
    assert sum(a['charged_gpu_seconds'] for a in finalized) <= 576000
    for device in read(stage + 'gpu_acceptance.json')['devices']:
        a = ledger['attempts'][device['parent_attempt_id']]
        assert a['status'] == 'finalized' and a['returncode'] == 0 and a['kind'] == 'gpu_acceptance'
        assert device['continued_model_bitwise_equal'] and device['continued_optimizer_bitwise_equal']
        assert device['eval_logit_max_abs_error'] == 0 and device['all_inputs_synthetic']
    result = {'status': 'passed', 'execution_commit': receipt['execution_commit'], 'runs': 80,
        'CPU_tests': read(stage + 'cpu_acceptance.json')['tests_run'], 'GPU_devices': 2, 'frozen_time_groups': 4,
        'github_platform_protected': True, 'centralized_test_authorized': False}
    print(json.dumps(result))


if __name__ == '__main__': main()
