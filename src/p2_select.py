"""从完整 P2 事务日志确定性选择 LR；不访问 test，不启动 P3。"""

from collections import defaultdict
import math

from tlcls.p2_core import ROOT, STAGE, candidates, file_hash, read, scientific_identity, write
from tlcls.p2_transaction import EpochTransaction


def choose(group):
    """numerical_failure 可以落选；任何基础设施未完成都阻止该配置选择幸存者。"""
    if any(item['status'] not in ['completed', 'numerical_failure'] for item in group):
        return {'status': 'incomplete', 'selected': None, 'reason': 'candidate_not_terminal_or_infrastructure_incomplete'}
    eligible = [item for item in group if item['status'] == 'completed' and item.get('eligible')]
    if not eligible:
        return {'status': 'blocked', 'selected': None, 'reason': 'all_candidates_numerically_failed_or_invalid'}
    assert len({item['selection_denominator'] for item in eligible}) == 1
    # 三轮验证样本数固定，以 correct 总和比较，再按数值 LR 升序处理真正并列。
    selected = min(eligible, key=lambda item: (-item['selection_correct_sum'], item['learning_rate']))
    return {'status': 'selected', 'selected': selected, 'tie_break': 'smaller_numeric_learning_rate'}


def inspect_candidate(config):
    directory = ROOT / 'runs' / config['run_id']
    summary_path = directory / 'summary.json'
    item = {'run_id': config['run_id'], 'configuration_id': config['configuration_id'],
            'learning_rate': config['learning_rate'], 'candidate_index': config['tuning_candidate_index'],
            'status': 'planned', 'eligible': False}
    if not summary_path.exists():
        return item
    summary = read(summary_path)
    item['status'] = summary['status']
    identity = scientific_identity(config)
    if item['status'] != 'completed':
        if (directory / 'scientific.json').exists():
            assert read(directory / 'scientific.json') == identity
        item.update(scientific_config_sha256=identity['scientific_config_sha256'],
                    failure=summary.get('failure'), summary_sha256=file_hash(summary_path))
        return item
    assert read(directory / 'config.json') == config
    assert read(directory / 'scientific.json') == identity
    assert summary['scientific_config_sha256'] == identity['scientific_config_sha256']
    assert summary['phase'] == 'P2' and summary['run_id'] == config['run_id']
    assert summary['test_evaluator_constructed'] is False and summary['test_metrics'] is None
    item.update(scientific_config_sha256=identity['scientific_config_sha256'],
                scientific_config=identity['scientific_config'], summary_sha256=file_hash(summary_path))
    transaction = EpochTransaction(directory, identity['scientific_config_sha256'])
    current, rows = transaction.load(rebuild=False)
    assert current['completed_epoch'] == summary['completed_epochs'] == len(rows) == 15
    samples = read(ROOT / 'records' / (config['dataset'] + '_split.json'))
    n, nv = len(samples['train_indices']), len(samples['validation_indices'])
    steps = math.ceil(n / 128)
    for epoch, row in enumerate(rows, 1):
        assert row['epoch'] == epoch and row['status'] == 'completed'
        assert row['scientific_config_sha256'] == identity['scientific_config_sha256']
        assert row['train']['samples'] == n and row['validation']['samples'] == nv
        assert row['train']['optimizer_updates'] == steps and row['cumulative_optimizer_updates'] == epoch * steps
        assert row['train']['last_update']['samples'] == (n % 128 or 128)
        assert math.isfinite(row['train']['loss']) and math.isfinite(row['validation']['loss'])
        assert math.isfinite(row['validation']['top1_percent'])
        assert 0 <= row['validation']['correct'] <= nv
        assert math.isclose(row['validation']['top1_percent'], 100 * row['validation']['correct'] / nv)
    last = rows[12:15]
    correct = sum(r['validation']['correct'] for r in last)
    item.update(eligible=True, completed_epochs=15, selection_epochs=[13, 14, 15],
        validation_top1_values=[r['validation']['top1_percent'] for r in last],
        selection_correct_sum=correct, selection_denominator=3 * nv,
        selection_value=100 * correct / (3 * nv),
        epochs_sha256=file_hash(directory / 'epochs.jsonl'),
        final_commit=read(directory / 'current.json'))
    # 派生日志也必须与事务行一致，不能用手工改过的 CSV 参与选择。
    assert [read_json_line(line) for line in (directory / 'epochs.jsonl').read_text().splitlines()] == rows
    return item


def read_json_line(line):
    import json
    return json.loads(line)


def main():
    grouped = defaultdict(list)
    for config in candidates():
        grouped[config['configuration_id']].append(inspect_candidate(config))
    results = {}
    for name, group in sorted(grouped.items()):
        decision = choose(group)
        selected = decision.pop('selected')
        results[name] = dict(decision, candidates=group, configuration_id=name,
            learning_rate=selected['learning_rate'] if selected else None,
            head_lr_multiplier=10.0 if '-full_ft' in name else 1.0,
            selected_from_run_id=selected['run_id'] if selected else None,
            selection_metric='mean_val_top1_epochs_13_to_15',
            selection_value=selected['selection_value'] if selected else None,
            selected_scientific_config_sha256=selected['scientific_config_sha256'] if selected else None)
    value = {'protocol_id': 'TL-CLS-20261020-v1.0', 'phase_source': 'P2',
        'selection_rule': 'maximum sum of validation correct at epochs 13,14,15; exact ties use smaller LR',
        'configurations': results, 'all_configurations_selected': all(x['status'] == 'selected' for x in results.values()),
        'P3_authorized': False, 'centralized_test_authorized': False,
        'authorization_sha256': file_hash(STAGE / 'authorization.json')}
    write(ROOT / 'selected_hparams.json', value)
    print({k: v['status'] for k, v in results.items()}, flush=True)


if __name__ == '__main__':
    main()
