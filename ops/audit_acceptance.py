"""独立核对回收后的 P0/P1 证据链，不运行模型、不接触测试评价器。"""

from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path):
    # JSON 标准不允许 NaN/Infinity；在读取时即拒绝，避免把坏指标当作有效数值。
    def reject(value):
        raise ValueError('non-finite JSON token: ' + value)
    return json.loads(path.read_text(encoding='utf-8'), parse_constant=reject)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = load(ROOT / 'records/evidence_manifest.json')
    for entry in manifest['files']:
        path = ROOT / entry['path']
        assert path.resolve().is_relative_to(ROOT.resolve())
        assert path.stat().st_size == entry['size_bytes'] and digest(path) == entry['sha256'], entry['path']
    report = load(ROOT / 'reports/P0_P1_report.json')
    matrix = load(ROOT / 'configs/run_matrix.json')
    expected = {r['run_id']: r for r in matrix['planned_runs'] if r['phase'] == 'P1'}
    assert report['P0']['status'] == 'passed' and report['P0']['completed_configurations'] == 16
    assert not report['P0']['failures']
    assert report['P1_completed_runs'] == len(expected) == 16
    assert {item['run_id'] for item in report['P1']} == set(expected)
    assert len(list((ROOT / 'runs').glob('*/summary.json'))) == 16
    head_hashes, order_hashes = defaultdict(set), defaultdict(set)
    devices = defaultdict(Counter)
    rows, processed, updates = [], 0, 0
    revision = load(ROOT / 'records/scheduling_revision.json')
    allowed_commits = {revision['old_code_commit'], revision['new_code_commit']}
    for item in report['P1']:
        run_id = item['run_id']
        directory = ROOT / 'runs' / run_id
        summary, config = load(directory / 'summary.json'), load(directory / 'config.json')
        epochs = [json.loads(line) for line in (directory / 'epochs.jsonl').read_text().splitlines()]
        assert summary['status'] == 'completed' and summary['completed_epochs'] == len(epochs) == 3
        assert summary['test_evaluator_constructed'] is False and summary['test_metrics'] is None
        assert summary['provenance']['code_commit'] in allowed_commits
        assert summary['provenance']['resolved_config_sha256'] == digest(directory / 'config.json')
        assert summary['provenance']['dependency_lock_sha256'] == digest(ROOT / 'records/dependencies.lock')
        assert summary['provenance']['protocol_sha256'] == digest(ROOT / 'configs/protocol.json')
        assert config['seed'] == 314159 and config['learning_rate'] == expected[run_id]['learning_rate']
        assert config['no_feature_cache'] and config['no_torch_compile'] and not config['tf32']
        dataset = report['data']['datasets'][item['dataset']]
        split = ROOT / 'records' / (item['dataset'] + '_split.json')
        assert summary['provenance']['split_sha256'] == digest(split)
        n, nv = dataset['train_n'], dataset['validation_n']
        steps = math.ceil(n / 128)
        group = (item['dataset'], item['model_key'])
        head_hashes[group].add(summary['model_identity']['initial_classifier_sha256'])
        devices[item['method']][summary['hardware']['index']] += 1
        for index, epoch in enumerate(epochs, 1):
            assert epoch['status'] == 'completed' and epoch['epoch'] == index
            assert epoch['train']['samples'] == n and epoch['validation']['samples'] == nv
            assert epoch['train']['optimizer_updates'] == steps
            assert epoch['cumulative_optimizer_updates'] == steps * index
            assert epoch['train']['last_update']['samples'] == (n % 128 or 128)
            assert math.isclose(epoch['validation']['top1_percent'], 100 * epoch['validation']['correct'] / nv)
            order_hashes[(group, index)].add(epoch['sampler_order_sha256'])
            processed += n
            updates += steps
        assert summary['optimizer_updates'] == steps * 3
        best = max(epochs, key=lambda e: (e['validation']['top1_percent'], -e['epoch']))
        assert summary['selected_epoch'] == best['epoch']
        assert summary['best_val_top1_percent'] == best['validation']['top1_percent']
        p = summary['parameters']
        p0 = next(check for check in report['P0']['checks'] if check['run_id'] == run_id)
        assert p == p0['counts'], 'P1 参数整数必须与已通过的 P0 配置一致'
        rows.append({'run_id': run_id, 'dataset': item['dataset'], 'model': item['model_key'],
            'method': item['method'], 'epochs': 3, 'validation_best_top1_percent': summary['best_val_top1_percent'],
            'selected_epoch': summary['selected_epoch'], 'total_parameters': p['total_params'],
            'trainable_parameters': p['trainable_params'], 'adapter_parameters': p['added_params'],
            'head_parameters': p['head_params'], 'trainable_fraction': p['trainable_ratio'],
            'epoch2_3_fit_seconds': item['epoch2_3_mean_fit_seconds'],
            'gpu_hours': summary['gpu_hours'], 'physical_gpu': summary['hardware']['index'],
            'peak_allocated_gib': summary['peak_allocated_bytes'] / 2**30,
            'peak_reserved_gib': summary['peak_reserved_bytes'] / 2**30,
            'best_val_mib': summary['best_val_checkpoint_bytes'] / 2**20,
            'resumable_last_mib': summary['last_checkpoint_bytes'] / 2**20,
            'task_delta_mib': summary.get('task_delta_bytes', 0) / 2**20,
            'test_top1': None, 'status': summary['status']})
    assert all(len(values) == 1 for values in head_hashes.values())
    assert all(len(values) == 1 for values in order_hashes.values())
    assert all(counts == Counter({0: 2, 1: 2}) for counts in devices.values())
    ledger = report['budget_ledger']
    assert ledger['gpu_hours'] <= 4 and ledger['status'] == 'P1_completed'
    assert math.isclose(ledger['gpu_seconds'], sum(e['gpu_seconds'] for e in ledger['entries']))
    source = load(ROOT / 'records/final_source_verification.json')
    assert source['training_model_data_implementation_unchanged'] and not source['git_status_porcelain']
    assert report['next_budget']['complete_basis'] and report['long_run_authorization'] is False
    inventory = load(ROOT / 'records/checkpoint_inventory.json')
    assert inventory['status'] == 'verified' and len(inventory['files']) == 40
    assert len(report['concurrency']) == 2
    result = {'status': 'passed', 'verified_evidence_files': len(manifest['files']),
        'P0_configurations_passed': 16, 'P1_runs_completed': 16, 'P1_complete_epochs': 48,
        'processed_training_samples': processed, 'optimizer_updates': updates,
        'paired_classifier_hashes': True, 'paired_epoch_sampler_hashes': True,
        'method_gpu_counts': {method: dict(counts) for method, counts in devices.items()},
        'checkpoint_files_verified_on_server': 40, 'test_evaluations_executed': 0,
        'actual_gpu_hours': ledger['gpu_hours'], 'cap_gpu_hours': 4,
        'concurrency_above_10_percent': [x['run_id'] for x in report['concurrency'] if x['above_10_percent']],
        'scope': 'P0/P1 engineering acceptance only; no formal accuracy ranking or long-run stability claim'}
    (ROOT / 'records/acceptance_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    with (ROOT / 'reports/P1_acceptance_table.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
