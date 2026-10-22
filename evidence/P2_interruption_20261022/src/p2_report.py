"""P2 全候选、成本、checkpoint 与 P3 草案；输出本身不授予后续执行权限。"""

import csv
from collections import defaultdict
import math
from pathlib import Path
import statistics

from tlcls.p2_budget import Budget, Storage
from tlcls.p2_core import ROOT, STAGE, candidates, code_commit, file_hash, read, write


def optional(path):
    return read(path) if path.exists() else None


def main():
    authorization = read(STAGE / 'authorization.json')
    runs = candidates()
    ledger = Budget().snapshot()
    assert all(a['status'] == 'finalized' for a in ledger['attempts'].values())
    selected = read(ROOT / 'selected_hparams.json')
    report = {'phase': 'P2', 'execution_commit': code_commit(), 'authorization': authorization,
        'acceptance': optional(STAGE / 'acceptance.json'), 'gpu_ledger': ledger,
        'cost_calibration': optional(STAGE / 'cost_calibration.json'),
        'stage_status': optional(STAGE / 'stage_status.json'), 'candidates': [],
        'P3_authorized': False, 'centralized_test_authorized': False, 'billing_applicable': False}
    report['technical_revisions'] = optional(STAGE / 'technical_revisions.json')
    table, curves, inventory = [], [], []
    by_configuration = {}
    for config in runs:
        directory = ROOT / 'runs' / config['run_id']
        summary = optional(directory / 'summary.json') or {'status': 'planned', 'completed_epochs': 0}
        epochs = [__import__('json').loads(line) for line in (directory / 'epochs.jsonl').read_text().splitlines()] if (directory / 'epochs.jsonl').exists() else []
        timings = [read(p) for p in sorted((directory / 'timings').glob('epoch_*.json'))]
        attempts = [a for a in ledger['attempts'].values() if a['run_id'] == config['run_id']]
        actual = sum(a['charged_gpu_seconds'] for a in attempts)
        chosen = selected['configurations'][config['configuration_id']]
        candidate_selection = next(c for c in chosen['candidates'] if c['run_id'] == config['run_id'])
        item = {'config': config, 'summary': summary, 'epochs': epochs, 'timings': timings,
                'attempts': attempts, 'actual_gpu_seconds_parent': actual, 'selection': candidate_selection}
        report['candidates'].append(item)
        table.append({'run_id': config['run_id'], 'configuration_id': config['configuration_id'],
            'learning_rate': config['learning_rate'], 'status': summary['status'],
            'completed_epochs': summary.get('completed_epochs', 0),
            'validation_top1_epoch13': next((r['validation']['top1_percent'] for r in epochs if r['epoch'] == 13), None),
            'validation_top1_epoch14': next((r['validation']['top1_percent'] for r in epochs if r['epoch'] == 14), None),
            'validation_top1_epoch15': next((r['validation']['top1_percent'] for r in epochs if r['epoch'] == 15), None),
            'selection_mean_top1': candidate_selection.get('selection_value'),
            'selected': chosen['selected_from_run_id'] == config['run_id'],
            'actual_gpu_hours_parent': actual / 3600, 'attempt_count': len(attempts),
            'cpu_rss_final_bytes': summary.get('cpu_rss_final_bytes'),
            'main_process_peak_rss_bytes_sampled': max([0] + [a['main_process_peak_rss_bytes_sampled'] for a in attempts]),
            'process_tree_peak_rss_bytes_sampled': max([0] + [a['process_tree_peak_rss_bytes_sampled'] for a in attempts]),
            'peak_allocated_bytes': summary.get('peak_allocated_bytes'), 'peak_reserved_bytes': summary.get('peak_reserved_bytes'),
            'scientific_config_sha256': summary.get('scientific_config_sha256')})
        for row in epochs:
            curves.append({'run_id': config['run_id'], 'configuration_id': config['configuration_id'],
                'learning_rate': config['learning_rate'], 'epoch': row['epoch'],
                'train_loss': row['train']['loss'], 'validation_loss': row['validation']['loss'],
                'validation_top1_percent': row['validation']['top1_percent'],
                'scientific_config_sha256': row['scientific_config_sha256']})
        if chosen['selected_from_run_id'] == config['run_id'] and len(timings) >= 3:
            stable = statistics.median(t['fit_seconds'] for t in timings if t['epoch'] >= 2)
            fixed = max(0.0, actual - sum(t['fit_seconds'] for t in timings))
            fixed += max(0, timings[0]['fit_seconds'] - stable)
            by_configuration[config['configuration_id']] = {'epoch_seconds': stable,
                'startup_terminal_first_epoch_excess_seconds': fixed, 'source_run_id': config['run_id'],
                'source_attempt_count': len(attempts), 'timing_clock': 'complete epoch transaction wall time'}
        # 只将现存实际 checkpoint 纳入清单；已回收的旧代次身份留在各 run 的历史记录中。
        for path in sorted((directory / 'checkpoints').glob('*.pt')):
            inventory.append({'path': str(path.relative_to(ROOT)), 'size_bytes': path.stat().st_size,
                              'sha256': file_hash(path), 'phase': 'P2'})
    for path in sorted((ROOT / 'technical_revisions').rglob('*.pt')):
        inventory.append({'path': str(path.relative_to(ROOT)), 'size_bytes': path.stat().st_size,
                          'sha256': file_hash(path), 'phase': 'P2_technical_revision_not_selection_eligible'})
    # P1 的 40 份已验收模型继续保留，并与旧清单重新核对，不复写旧收据。
    p1 = read(ROOT / 'records/checkpoint_inventory.json')
    for item in p1['files']:
        path = ROOT / item['path']
        assert path.stat().st_size == item['size_bytes'] and file_hash(path) == item['sha256']
    write(STAGE / 'checkpoint_inventory.json', {'status': 'verified', 'P2_files': inventory,
        'P1_retained_and_reverified': len(p1['files']), 'checkpoint_transfers': 0})
    report['storage'] = Storage(authorization['confirmed_project_environment_allocation_bytes']).snapshot()
    report['P2_actual_gpu_hours'] = Budget.used(ledger) / 3600
    report['P2_completed_candidates'] = sum(x['summary']['status'] == 'completed' for x in report['candidates'])
    report['P2_numerical_failures'] = sum(x['summary']['status'] == 'numerical_failure' for x in report['candidates'])
    report['selection_all_configurations_complete'] = selected['all_configurations_selected']
    report['selected_hparams_sha256'] = file_hash(ROOT / 'selected_hparams.json')
    formal = [r for r in read(ROOT / 'configs/run_matrix.json')['planned_runs'] if r['phase'] == 'P3']
    resolved = []
    p3_training = 0.0
    times = defaultdict(dict)
    for run in formal:
        selection = selected['configurations'][run['configuration_id']]
        draft = dict(run, learning_rate=selection['learning_rate'], selected_hparams_sha256=file_hash(ROOT / 'selected_hparams.json'),
                     selected_P2_run_id=selection['selected_from_run_id'], status='draft_not_authorized')
        resolved.append(draft)
        if run['configuration_id'] in by_configuration:
            estimate = by_configuration[run['configuration_id']]
            seconds = 50 * estimate['epoch_seconds'] + estimate['startup_terminal_first_epoch_excess_seconds']
            p3_training += seconds / 3600
            times[(run['dataset'], run['model_key'])][run['method']] = seconds
    slice_groups = {f'{key[0]}-{key[1]}': {'B_seconds_proposal': min(methods.values()),
        'fractions': [.25, .5, 1.0], 'slice_seconds_proposal': [min(methods.values()) * f for f in [.25, .5, 1.0]],
        'method_50_epoch_seconds_estimates': methods} for key, methods in times.items()}
    # 切片存储按每次最多三份额外完整模型给上界，不用未来同 hash 复用来缩减硬需求。
    selected_sizes = [c['summary']['best_checkpoint']['size_bytes'] for c in report['candidates']
                      if c['config']['configuration_id'] in by_configuration and
                      selected['configurations'][c['config']['configuration_id']]['selected_from_run_id'] == c['config']['run_id']]
    p2_live_bytes = sum(x['size_bytes'] for x in inventory)
    selected_runs = [c for c in report['candidates'] if
        selected['configurations'][c['config']['configuration_id']]['selected_from_run_id'] == c['config']['run_id']]
    prefix_save_hours = sum(15 * statistics.median(t['checkpoint_seconds'] for t in c['timings']) / 3600
                            for c in selected_runs if c['timings'])
    evaluation_hours = 0.0
    data_identity = read(ROOT / 'records/data_integrity.json')['datasets']
    for c in selected_runs:
        d = data_identity[c['config']['dataset']]
        evaluation_hours += (25 * statistics.median(t['validation_seconds'] for t in c['timings']) *
                             d['official_test_n'] / d['validation_n'] / 3600)
    selected_checkpoint_bytes = sum(c['summary'][key]['size_bytes'] for c in selected_runs
                                   for key in ['best_checkpoint', 'last_checkpoint', 'task_delta']
                                   if key in c['summary'])
    complete_proposal = len(by_configuration) == len(selected_runs) == 16
    prefix_bytes = sum(selected_sizes) * 15 if complete_proposal else None
    proposal = {'status': 'draft_waiting_PI', 'P3_authorized': False, 'centralized_test_authorized': False,
        'resolved_runs': resolved, 'selected_timing_basis': by_configuration,
        'basis_configurations': len(by_configuration),
        'P3_base_training_gpu_hours': p3_training if len(by_configuration) == 16 else None,
        'P3_base_training_with_25_percent_margin_gpu_hours': p3_training * 1.25 if len(by_configuration) == 16 else None,
        'contention': 'P2 complete-epoch measurements already include actual overlap; do not multiply the same slowdown twice',
        'common_time_slices': slice_groups, 'common_time_slices_frozen': False,
        'B_rule_proposal': 'min of four timing-only 50-epoch estimates within dataset/backbone; P2 recalibration of original P1 timing formula requires PI freeze',
        'extra_prefix_checkpoint_upper_bound_bytes': prefix_bytes,
        'P3_best_last_delta_storage_estimate_bytes': selected_checkpoint_bytes * 5 if complete_proposal else None,
        'P3_extra_prefix_save_gpu_hours_upper_scenario': prefix_save_hours if complete_proposal else None,
        'prefix_save_estimation_basis': 'three per seed; each charged a full measured P2 transaction as conservative save scenario',
        'centralized_evaluation_gpu_hours_upper_scenario': evaluation_hours if complete_proposal else None,
        'evaluation_estimation_basis': '80 best + 80 last + up to 240 prefix, scaled P2 validation timing; no model test scores accessed',
        'diagnostic_reserve_gpu_hours_proposal': 2.0,
        'inference_benchmark_reserve_gpu_hours_proposal': .5,
        'P3_training_prefix_evaluation_reserves_with_25_percent_margin_gpu_hours':
            1.25 * (p3_training + prefix_save_hours + evaluation_hours + 2.5) if complete_proposal else None,
        'project_environment_full_retention_with_prefix_and_25_percent_margin_bytes':
            1.25 * (report['storage']['combined_bytes'] + 5 * selected_checkpoint_bytes + prefix_bytes)
            if complete_proposal else None,
        'max_evaluation_roles': {'primary_best_val': 80, 'secondary_last': 80, 'budget_prefix': 240},
        'evaluation_compute_and_prefix_save_compute': 'upper scenarios proposed, not frozen; no test evaluation performed in P2',
        'required_before_P3': ['PI budget and storage decision', '80-run P3 entry and acceptance',
            'immutable prefix commits and timing freeze', 'P3 seal and evaluation manifest generator',
            'unique evaluation_id and run+checkpoint prediction reuse', 'strict evaluation identity/status checks before TestData']}
    write(ROOT / 'reports/P3_proposal.json', proposal)
    write(ROOT / 'reports/P2_report.json', report)
    for name, rows in [('P2_candidate_results.csv', table), ('P2_epoch_curves.csv', curves)]:
        with (ROOT / 'reports' / name).open('w', encoding='utf-8-sig', newline='') as file:
            if rows:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print({'P2_completed': report['P2_completed_candidates'], 'GPU_hours': report['P2_actual_gpu_hours'],
           'P3_started': False}, flush=True)


if __name__ == '__main__':
    main()
