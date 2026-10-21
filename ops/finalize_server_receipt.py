"""P1 停止后执行的 CPU 证据封存；不改变训练代码、配置或 checkpoint。"""

import datetime
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess


ROOT = Path('/root/user_data/tl_cls_v1')


def read(relative):
    return json.loads((ROOT / relative).read_text())


def write(relative, value):
    path = ROOT / relative
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def disk_bytes(path):
    # 不跟随环境链接，避免把复用的基础 torch 环境算入项目新增磁盘。
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file() and not p.is_symlink())


def main():
    report = read('reports/P0_P1_report.json')
    ledger = read('records/gpu_budget.json')
    assert report['P1_completed_runs'] == 16 and ledger['status'] == 'P1_completed'
    assert ledger['gpu_seconds'] <= ledger['cap_gpu_seconds']
    assert all(item['status'] == 'completed' for item in report['P1'])
    # 当前阶段不允许混入 P2/P3；原始短程运行证据保留其真实 code commit。
    summaries = list((ROOT / 'runs').glob('*/summary.json'))
    assert len(summaries) == 16 and all(json.loads(p.read_text())['phase'] == 'P1' for p in summaries)
    inventory = []
    for item in report['P1']:
        summary = item['summary']
        for filename, prefix in [('best_val.pt', 'best_val_checkpoint'), ('last.pt', 'last_checkpoint'),
                                 ('task_delta.pt', 'task_delta')]:
            if prefix + '_bytes' not in summary:
                continue
            path = ROOT / 'runs' / item['run_id'] / filename
            actual = digest(path)
            assert actual == summary[prefix + '_sha256'] and path.stat().st_size == summary[prefix + '_bytes']
            inventory.append({'path': str(path.relative_to(ROOT)), 'size_bytes': path.stat().st_size,
                              'sha256': actual, 'verified_against_training_receipt': True})
    write('records/checkpoint_inventory.json', {'files': inventory, 'status': 'verified',
          'policy': 'checkpoint binaries remain on the server; no local checkpoint transfer'})
    revision = read('records/scheduling_revision.json')
    changed = subprocess.check_output(['git', 'diff', '--name-only', revision['old_code_commit'],
                                      revision['new_code_commit']], cwd=ROOT, text=True).splitlines()
    assert changed == ['src/execute_p0_p1.py']
    write('records/final_source_verification.json', {'changed_files_between_training_commits': changed,
          'training_model_data_implementation_unchanged': True,
          'old_commit': revision['old_code_commit'], 'new_commit': revision['new_code_commit'],
          'git_status_porcelain': subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)})
    folders = {name: disk_bytes(ROOT / name) for name in ['data', 'weights', 'runs', 'records',
                                                       'src', 'configs', 'offline_wheels', 'reports']}
    storage = {'project_root': str(ROOT), 'project_file_bytes': disk_bytes(ROOT),
               'by_folder_file_bytes': folders,
               'project_environment_non_symlink_bytes': disk_bytes(Path('/root/miniconda3/envs/tl_cls_v1')),
               'filesystem_free_bytes': shutil.disk_usage(ROOT).free,
               'tenant_confirmed_quota_bytes': None, 'planning_cap_gib': 100,
               'measurement': 'logical file sizes; shared filesystem free space is not tenant quota'}
    write('records/storage_receipt.json', storage)
    budget = read('records/budget_plan.json')
    # 预算是阶段评审提案，不能把余量、预留或推算误写为已发生计算。
    evaluation_hours = 0.0
    for item in report['P1']:
        validation = statistics.mean(e['validation_seconds'] for e in item['epochs'] if e['epoch'] in [2, 3])
        dataset = report['data']['datasets'][item['dataset']]
        evaluation_hours += 5 * 2 * validation * dataset['official_test_n'] / dataset['validation_n'] / 3600
    training = budget['P2_gpu_hours'] + budget['P3_gpu_hours']
    staging = read('records/transfer_constraints.json')
    addendum = {'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'billing_context': read('records/billing_context.json'), 'transfer_constraints': staging,
        'original_report_sha256': digest(ROOT / 'reports/P0_P1_report.json'),
        'original_budget_sha256': digest(ROOT / 'records/budget_plan.json'),
        'checkpoint_count_verified': len(inventory), 'storage': storage,
        'budget_proposal': {
            'P2_P3_training_gpu_hours': training,
            'centralized_best_plus_last_test_estimate_gpu_hours': evaluation_hours,
            'evaluation_estimation_basis': 'P1 epoch2/3 validation time scaled by official test/validation sample counts, 80 best + 80 last; no test model evaluation executed',
            'bounded_diagnostics_reserve_gpu_hours': 2.0,
            'inference_benchmark_reserve_gpu_hours': 0.5,
            'training_evaluation_diagnostics_inference_with_25_percent_margin_gpu_hours':
                1.25 * (training + evaluation_hours + 2.5),
            'P3_budget_prefix_weights_and_evaluations': 'not frozen; separate cost/storage review after P2',
            'monetary_cost': 'not applicable: school server without pricing',
            'authorization_status': 'proposal only; P2/P3 not started'}}
    write('reports/acceptance_addendum.json', addendum)
    print(json.dumps({'checkpoint_files_verified': len(inventory),
                      'project_gib': storage['project_file_bytes'] / 2**30,
                      'training_gpu_hours_proposal': training,
                      'billing_applicable': False}))


if __name__ == '__main__':
    main()
