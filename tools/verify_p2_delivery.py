"""离线交付复核：重新计算事务、选参和成本；不导入 torch，不运行训练/测试。"""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()

    def read(relative):
        return json.loads((root / relative).read_text(encoding='utf-8'))

    manifest = read('records/p2/evidence_manifest.json')
    if not args.allow_incomplete:
        assert manifest['candidates_terminal'] and manifest['all_configurations_selected']
    for item in manifest['files']:
        path = (root / item['path']).resolve()
        assert path.is_relative_to(root)
        assert path.stat().st_size == item['size_bytes'] and digest(path) == item['sha256'], item['path']
    acceptance = read('records/p2/acceptance.json')
    authorization = read('records/p2/authorization.json')
    assert acceptance['status'] == 'passed'
    assert acceptance['execution_commit'] == manifest['execution_commit']
    assert authorization['P2_authorized'] and not authorization['P3_authorized']
    assert not authorization['centralized_test_authorized']
    assert authorization['hard_cap_gpu_seconds'] == 108000
    assert authorization['confirmed_project_environment_allocation_bytes'] >= 60 * 2**30
    assert acceptance['authorization_sha256'] == digest(root / 'records/p2/authorization.json')
    for file, field in [('cpu_acceptance.json', 'cpu_receipt_sha256'), ('gpu_acceptance.json', 'gpu_receipt_sha256')]:
        assert digest(root / 'records/p2' / file) == acceptance[field]
        child = read('records/p2/' + file)
        assert child['status'] == 'passed' and child['execution_identity'] == acceptance['execution_identity']
        assert child['execution_commit'] == acceptance['execution_commit']
    for file, sha in acceptance['execution_identity']['files'].items():
        assert digest(root / file) == sha, file
    compatibility = acceptance.get('operational_compatibility')
    science_basis = compatibility['scientific_implementation_basis'] if compatibility else acceptance['execution_identity']
    runtime_receipts = {acceptance['execution_commit']: acceptance}
    for path in (root / 'records/p2/acceptance_history').glob('*/acceptance.json'):
        prior = json.loads(path.read_text(encoding='utf-8'))
        assert prior['status'] == 'passed'
        runtime_receipts[prior['execution_commit']] = prior
    matrix = read('configs/run_matrix.json')
    runs = [r for r in matrix['planned_runs'] if r['phase'] == 'P2']
    assert len(runs) == len({r['run_id'] for r in runs}) == 48
    selected = read('selected_hparams.json')
    assert len(selected['configurations']) == 16
    if not args.allow_incomplete:
        assert selected['all_configurations_selected']
    assert not selected['P3_authorized'] and not selected['centralized_test_authorized']
    groups, orders, heads = defaultdict(list), defaultdict(set), defaultdict(set)
    counts = defaultdict(int)
    for config in runs:
        name = 'runs/' + config['run_id'] + '/'
        if not (root / name / 'summary.json').exists():
            assert args.allow_incomplete
            counts['planned'] += 1
            groups[config['configuration_id']].append((None, config['learning_rate'], config['run_id']))
            continue
        assert read(name + 'config.json') == config
        science = read(name + 'scientific.json')
        sha = hashlib.sha256(canonical(science['scientific_config'])).hexdigest()
        assert sha == science['scientific_config_sha256']
        identity = science['scientific_config']
        assert identity['implementation'] == science_basis
        asset = next(a for a in read('records/assets.json')['assets'] if a['key'] == config['model_key'] + '_model.safetensors')
        assert identity['weight_sha256'] == asset['sha256'] and identity['weight_revision'] == asset['revision']
        for key, value in identity['configuration'].items():
            assert config[key] == value, (config['run_id'], key)
        for field, file in [('protocol_sha256', 'configs/protocol.json'),
            ('run_matrix_sha256', 'configs/run_matrix.json'),
            ('dependency_lock_sha256', 'records/dependencies.lock'),
            ('split_sha256', 'records/' + config['dataset'] + '_split.json')]:
            assert identity[field] == digest(root / file)
        assert (config['seed'], config['epochs'], config['scheduler_horizon_epochs'], config['warmup_epochs']) == (123, 15, 50, 5)
        assert (config['micro_batch_size'], config['gradient_accumulation_steps'], config['effective_batch_size']) == (32, 4, 128)
        assert config['autocast_dtype'] == 'bfloat16' and not config['test_during_training']
        summary = read(name + 'summary.json')
        # 历史完成项保持原 commit；恢复项使用真实新 commit，不把原收据改写为新源码。
        assert summary['execution_commit'] in runtime_receipts
        runtime = runtime_receipts[summary['execution_commit']]['execution_identity']
        assert summary['implementation'] == runtime
        assert summary['scientific_config_sha256'] == sha
        assert not summary['test_evaluator_constructed'] and summary['test_metrics'] is None
        status = summary['status']
        allowed = ['completed', 'numerical_failure'] + (['infrastructure_interrupted'] if args.allow_incomplete else [])
        assert status in allowed, (config['run_id'], status)
        counts[status] += 1
        # 数值失败现场只核对身份和保留状态，不将其部分轮次加入完整候选均值。
        if status == 'numerical_failure':
            assert summary['failure'] and summary['failure_state']
            groups[config['configuration_id']].append((None, config['learning_rate'], config['run_id']))
            continue
        completed_epochs = summary['completed_epochs']
        assert completed_epochs == 15 if status == 'completed' else 0 <= completed_epochs < 15
        if completed_epochs == 0:
            groups[config['configuration_id']].append((None, config['learning_rate'], config['run_id']))
            continue
        reference = read(name + 'current.json')
        commits = []
        while reference:
            path = root / name / reference['path']
            assert digest(path) == reference['sha256']
            commit = json.loads(path.read_text(encoding='utf-8'))
            assert commit['scientific_config_sha256'] == sha
            commits.append(commit)
            reference = commit['previous_commit']
        commits.reverse()
        rows = [c['epoch_row'] for c in commits]
        log = [json.loads(line) for line in (root / name / 'epochs.jsonl').read_text(encoding='utf-8').splitlines()]
        assert rows == log and [r['epoch'] for r in rows] == list(range(1, completed_epochs + 1))
        assert [c['completed_epoch'] for c in commits] == list(range(1, completed_epochs + 1))
        clocks = [c['cumulative_gpu_seconds_at_commit'] for c in commits]
        assert all(b >= a for a, b in zip(clocks, clocks[1:]))
        split = read('records/' + config['dataset'] + '_split.json')
        n, nv = len(split['train_indices']), len(split['validation_indices'])
        for row in rows:
            epoch = row['epoch']
            if 'runtime_execution_commit' in row:
                actual_runtime = runtime_receipts[row['runtime_execution_commit']]['execution_identity']
                assert row['runtime_execution_identity_sha256'] == actual_runtime['scope_sha256']
            assert row['status'] == 'completed' and row['scientific_config_sha256'] == sha
            assert row['train']['samples'] == n and row['validation']['samples'] == nv
            assert row['train']['optimizer_updates'] == math.ceil(n / 128)
            assert row['cumulative_optimizer_updates'] == epoch * math.ceil(n / 128)
            assert row['train']['last_update']['samples'] == n % 128
            assert math.isfinite(row['train']['loss']) and math.isfinite(row['validation']['loss'])
            assert math.isclose(row['validation']['top1_percent'], 100 * row['validation']['correct'] / nv)
            orders[(config['dataset'], epoch)].add(row['sampler_order_sha256'])
            counts['epochs'] += 1
            counts['samples'] += n
            counts['updates'] += math.ceil(n / 128)
        heads[(config['dataset'], config['model_key'])].add(summary['model_identity']['initial_classifier_sha256'])
        best = max(rows, key=lambda r: (r['validation']['top1_percent'], -r['epoch']))
        assert commits[-1]['best_epoch'] == summary['selected_epoch'] == best['epoch']
        assert summary['best_val_top1_percent'] == best['validation']['top1_percent']
        assert commits[-1]['best'] == summary['best_checkpoint']
        assert commits[-1]['last'] == summary['last_checkpoint']
        if status != 'completed':
            # 未完成的有效前缀照常核验，但不能把其均值当成完整候选成绩。
            groups[config['configuration_id']].append((None, config['learning_rate'], config['run_id']))
            continue
        correct = sum(r['validation']['correct'] for r in rows[12:15])
        selection = selected['configurations'][config['configuration_id']]
        candidate = next(c for c in selection['candidates'] if c['run_id'] == config['run_id'])
        assert candidate['selection_correct_sum'] == correct and candidate['selection_denominator'] == 3 * nv
        assert math.isclose(candidate['selection_value'], 100 * correct / (3 * nv))
        groups[config['configuration_id']].append((correct, config['learning_rate'], config['run_id']))
    assert all(len(values) == 1 for values in orders.values()), 'paired_sampler_mismatch'
    assert all(len(values) == 1 for values in heads.values()), 'paired_classifier_initialization_mismatch'
    for name, group in groups.items():
        assert len(group) == 3
        if selected['configurations'][name]['status'] == 'incomplete':
            assert args.allow_incomplete
            assert selected['configurations'][name]['selected_from_run_id'] is None
            assert selected['configurations'][name]['learning_rate'] is None
            continue
        eligible = [v for v in group if v[0] is not None]
        winner = min(eligible, key=lambda v: (-v[0], v[1]))
        assert selected['configurations'][name]['selected_from_run_id'] == winner[2]
        assert selected['configurations'][name]['learning_rate'] == winner[1]
    ledger = read('records/p2/gpu_ledger.json')
    attempts = list(ledger['attempts'].values())
    assert all(a['status'] == 'finalized' for a in attempts)
    seconds = sum(a['charged_gpu_seconds'] for a in attempts)
    assert math.isclose(ledger['gpu_seconds_finalized'], seconds)
    assert seconds <= ledger['cap_gpu_seconds'] == 108000
    for attempt in attempts:
        assert math.isclose(attempt['charged_gpu_seconds'], attempt['ended_monotonic'] - attempt['started_monotonic'])
    for device in [0, 1]:
        # 同一卡只允许一个登记 attempt；跨卡重叠按两份实际占用相加。
        intervals = sorted([a for a in attempts if a['device'] == device], key=lambda a: a['started_monotonic'])
        assert all(b['started_monotonic'] >= a['ended_monotonic'] for a, b in zip(intervals, intervals[1:]))
    report = read('reports/P2_report.json')
    assert math.isclose(report['P2_actual_gpu_hours'], seconds / 3600)
    assert report['selected_hparams_sha256'] == digest(root / 'selected_hparams.json')
    assert report['P2_completed_candidates'] == counts['completed']
    assert report['P2_numerical_failures'] == counts['numerical_failure']
    storage = report['storage']
    assert storage['combined_bytes'] + 5 * 2**30 < storage['confirmed_allocation_bytes']
    inventory = read('records/p2/checkpoint_inventory.json')
    assert inventory['P1_retained_and_reverified'] == 40 and inventory['checkpoint_transfers'] == 0
    index = {item['path']: item for item in inventory['P2_files']}
    assert len(index) == len(inventory['P2_files'])
    for config in runs:
        if not (root / 'runs' / config['run_id'] / 'summary.json').exists():
            continue
        summary = read('runs/' + config['run_id'] + '/summary.json')
        for role in ['best_checkpoint', 'last_checkpoint', 'task_delta', 'failure_state']:
            if role not in summary:
                continue
            descriptor = summary[role]
            item = index['runs/' + config['run_id'] + '/' + descriptor['path']]
            assert item['sha256'] == descriptor['sha256'] and item['size_bytes'] == descriptor['size_bytes']
    proposal = read('reports/P3_proposal.json')
    assert not proposal['P3_authorized'] and not proposal['centralized_test_authorized']
    chosen_count = sum(c['status'] == 'selected' for c in selected['configurations'].values())
    assert len(proposal['resolved_runs']) == 80 and proposal['basis_configurations'] == chosen_count
    assert not proposal['common_time_slices_frozen']
    for run in proposal['resolved_runs']:
        choice = selected['configurations'][run['configuration_id']]
        assert run['learning_rate'] == choice['learning_rate'] and run['selected_P2_run_id'] == choice['selected_from_run_id']
        assert run['selected_hparams_sha256'] == report['selected_hparams_sha256']
        assert run['status'] == 'draft_not_authorized' and run['epochs'] == 50
    complete = manifest['candidates_terminal'] and selected['all_configurations_selected']
    result = {'status': 'passed' if complete else 'passed_for_available_evidence',
        'stage_acceptance': 'passed' if complete else 'incomplete_not_passed',
        'execution_commit': manifest['execution_commit'],
        'manifest_files_verified': len(manifest['files']), 'candidate_counts': dict(counts),
        'configurations_selected': chosen_count, 'paired_sampler_groups': len(orders),
        'actual_P2_gpu_hours_all_attempts': seconds / 3600, 'attempts': len(attempts),
        'combined_project_environment_gib': storage['combined_bytes'] / 2**30,
        'P1_checkpoints_reverified_server': 40, 'P2_checkpoint_index_entries': len(index),
        'checkpoint_bytes_rehashed_at_server': True, 'P3_authorized': False, 'centralized_test_authorized': False}
    if args.output:
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
