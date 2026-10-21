"""从原始证据汇总 P0/P1 与后续预算；未知字段保持 null，不代入旧论文数字。"""

import csv
import datetime
import json
import statistics

from tlcls.common import ROOT, atomic_json, load_json, provenance, sha256


def read_optional(path):
    return load_json(path) if path.exists() else None


def export_result(config, summary, epochs):
    result = load_json(ROOT / 'configs/result_record.template.json')
    result.update(template_only=False, run_id=config['run_id'], configuration_id=config['configuration_id'],
                  phase='P1', status=summary['status'], measurement_source='measured', seed=config['seed'],
                  dataset=config['dataset'], model_key=config['model_key'], method=config['method'],
                  planned_epochs=config['epochs'], completed_epochs=summary.get('completed_epochs', 0),
                  completed_utc=summary.get('completed_utc'))
    p = summary.get('provenance', {})
    identity = summary.get('model_identity', {})
    installed = load_json(ROOT / 'records/environment_installation.json')['versions']
    assets = load_json(ROOT / 'records/assets.json')['assets']
    weight = next(a for a in assets if a['key'] == config['model_key'] + '_model.safetensors')
    archives = [a for a in assets if a['key'].startswith('svhn_')] if config['dataset'] == 'svhn' else [next(a for a in assets if a['key'] == 'cifar100_archive')]
    result['provenance'].update(code_commit_sha=p.get('code_commit'),
        resolved_config_sha256=p.get('resolved_config_sha256'), dependency_lock_sha256=p.get('dependency_lock_sha256'),
        model_id=config['model_id'], weight_sha256=p.get('weight_sha256'), split_manifest_sha256=p.get('split_sha256'),
        initial_classifier_sha256=identity.get('initial_classifier_sha256'),
        checkpoint_includes_buffers=(summary.get('best_val_checkpoint_bytes') is not None),
        python_version='3.12.3', torch_version=installed['torch'], torchvision_version=installed['torchvision'],
        timm_version=installed['timm'], weight_revision=weight['revision'], weight_filename='model.safetensors',
        dataset_archive_sha256={a['key']: a['sha256'] for a in archives})
    params = summary.get('parameters', {})
    result['parameters_and_storage'].update(total_parameters=params.get('total_params'),
        trainable_parameters=params.get('trainable_params'), trainable_fraction=params.get('trainable_ratio'),
        trainable_classifier_parameters=params.get('head_params'), trainable_adapter_parameters=params.get('added_params'),
        base_parameters_excluding_classifier_and_adapters=params.get('base_params'),
        added_adapter_parameters=params.get('added_params'), classifier_parameters=params.get('head_params'),
        buffers_numel=params.get('buffer_elements'), self_contained_inference_checkpoint_bytes=summary.get('best_val_checkpoint_bytes'),
        resumable_training_checkpoint_bytes=summary.get('last_checkpoint_bytes'), task_delta_bytes=summary.get('task_delta_bytes'),
        trainable_backbone_parameters=params.get('base_params') if config['method'] in ['scratch', 'full_ft'] else 0,
        frozen_backbone_parameters=params.get('base_params') if config['method'] in ['linear_probe', 'adapter'] else 0)
    costs = summary.get('costs', {})
    result['timing'].update(source='measured', gpu_count=1, train_seconds=costs.get('train_seconds'),
        validation_seconds=costs.get('validation_seconds'), checkpoint_seconds=costs.get('checkpoint_seconds'),
        fit_wall_seconds=costs.get('fit_wall_seconds'), gpu_reserved_seconds=costs.get('gpu_reserved_seconds'),
        gpu_hours=summary.get('gpu_hours'), run_startup_seconds=summary.get('startup_seconds'),
        timing_mode='single_gpu_exclusive_full_epoch', parallel_peer_run_ids=[],
        recomputed_training_seconds=costs.get('recomputed_training_seconds'), cuda_synchronization_method='torch.cuda.synchronize')
    result['precision'].update(autocast_enabled=True, autocast_dtype='bfloat16', parameter_dtype='float32',
        loss_dtype='float32', gradient_scaler_enabled=False, tf32_matmul=False, tf32_cudnn=False,
        cudnn_benchmark=False, deterministic_algorithms=True, torch_compile=False, activation_checkpointing=False,
        dropout_rate=0.0, attention_dropout_rate=0.0, drop_path_rate=0.0)
    selected = summary.get('selected_epoch')
    result['checkpoint_selection'].update(selected_epoch=selected,
        selected_validation_top1_percent=summary.get('best_val_top1_percent'),
        checkpoint_sha256=summary.get('best_val_checkpoint_sha256'), test_used_in_selection=False)
    result['metrics'].update(validation_selected_top1_percent=summary.get('best_val_top1_percent'),
        final_epoch_validation_top1_percent=epochs[-1]['validation']['top1_percent'] if epochs else None)
    result['resolved_optimization'].update(learning_rate=config['learning_rate'],
        head_learning_rate=config['learning_rate'] * config['head_lr_multiplier'], optimizer='AdamW',
        micro_batch_size=32, gradient_accumulation_steps=4, effective_batch_size=128,
        optimizer_steps=summary.get('optimizer_updates'), skipped_optimizer_steps=0,
        scheduler_horizon_epochs=50, warmup_epochs=5)
    result['memory'].update(peak_allocated_bytes=summary.get('peak_allocated_bytes'),
        peak_reserved_bytes=summary.get('peak_reserved_bytes'), peak_cpu_rss_bytes=summary.get('cpu_rss_bytes'),
        measurement_region='entire_training_process_no_peak_reset_after_optimizer_initialization')
    result['failure'].update(occurred=summary['status'] not in ['running', 'completed'],
        reason=summary.get('failure', {}).get('reason'))
    result['notes'] = ['P1 profile only; not part of formal five-seed statistics.', 'No test evaluator or test accuracy.']
    hardware = summary.get('hardware', {})
    fields = dict(zip(hardware.get('fields', []), hardware.get('values', [])))
    result['hardware'].update(gpu_uuid=fields.get('uuid'), gpu_model=fields.get('name'),
        pci_bus_id=fields.get('pci.bus_id'), driver_version=fields.get('driver_version'),
        power_limit_w=float(fields['power.limit']) if fields.get('power.limit') else None,
        cpu_quota_vcpu=16, memory_quota_bytes=64 * 2**30, concurrent_run_count=1)
    result['evidence_files'] = ['config.json', 'summary.json', 'epochs.jsonl', 'parameter_groups.json']
    atomic_json(ROOT / 'runs' / config['run_id'] / 'result_record.json', result)


def main():
    runs = [r for r in load_json(ROOT / 'configs/run_matrix.json')['planned_runs'] if r['phase'] == 'P1']
    report = {'protocol_id': 'TL-CLS-20261020-v1.0', 'provenance': provenance(),
              'generated_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'environment': read_optional(ROOT / 'records/environment_installation.json'),
              'data': read_optional(ROOT / 'records/data_integrity.json'),
              'P0': read_optional(ROOT / 'records/implementation_checks.json'),
              'budget_ledger': read_optional(ROOT / 'records/gpu_budget.json'),
              'P1': [], 'long_run_authorization': False,
              'tenant_storage_quota_bytes': None, 'storage_planning_cap_gib': 100,
              'legacy_static_review': {
                'original_code_available_in_supplied_inputs': False,
                'source_manuscript_available': True,
                'legacy_exact_reproduction_claim': False,
                'numerical_failure_root_cause': None,
                'documented_issues_to_avoid': ['SVHN ViT Partial FT misidentified as LP',
                    'Residual connection alone does not guarantee initial adapter identity',
                    'SVHN horizontal flipping', 'Test-set checkpoint selection',
                    'Interpreting NaN argmax as valid accuracy'],
                'scope': 'Document-based static review; no historical executable code or checkpoints supplied.'}}
    estimates, rows = [], []
    for config in runs:
        folder = ROOT / 'runs' / config['run_id']
        summary = read_optional(folder / 'summary.json')
        epochs = [json.loads(line) for line in (folder / 'epochs.jsonl').read_text().splitlines()] if (folder / 'epochs.jsonl').exists() else []
        stable = [epoch['train_seconds'] + epoch['validation_seconds'] + epoch['checkpoint_seconds']
                  for epoch in epochs if epoch['epoch'] in [2, 3]]
        t = statistics.mean(stable) if len(stable) == 2 else None
        status = summary['status'] if summary else 'planned'
        item = {'run_id': config['run_id'], 'configuration_id': config['configuration_id'],
                'dataset': config['dataset'], 'model_key': config['model_key'], 'method': config['method'],
                'status': status, 'completed_epochs': summary.get('completed_epochs', 0) if summary else 0,
                'epoch2_3_mean_fit_seconds': t, 'epochs': epochs,
                'summary': summary}
        report['P1'].append(item)
        rows.append({'run_id': config['run_id'], 'status': status,
            'completed_epochs': item['completed_epochs'], 'epoch2_3_mean_fit_seconds': t,
            'best_val_top1_percent': summary.get('best_val_top1_percent') if summary else None,
            'gpu_hours': summary.get('gpu_hours') if summary else None,
            'peak_allocated_gib': summary.get('peak_allocated_bytes', 0) / 2**30 if summary else None,
            'peak_reserved_gib': summary.get('peak_reserved_bytes', 0) / 2**30 if summary else None})
        if summary:
            export_result(config, summary, epochs)
        if t is not None:
            startup = summary['startup_seconds'] + summary.get('terminal_overhead_seconds', 0)
            if epochs:
                first = epochs[0]
                startup += max(0.0, first['train_seconds'] + first['validation_seconds']
                               + first['checkpoint_seconds'] - t)
            estimates.append({'configuration_id': config['configuration_id'], 'epoch_seconds': t,
                'per_run_fixed_and_first_epoch_excess_seconds': startup,
                'P2_gpu_hours_estimate': (45 * t + 3 * startup) / 3600,
                'P3_gpu_hours_estimate': (250 * t + 5 * startup) / 3600,
                'basis_profile_run_id': config['run_id']})
    concurrency = []
    for index in [0, 1]:
        solo = read_optional(ROOT / 'records' / f'benchmark_solo_{index}.json')
        concurrent = read_optional(ROOT / 'records' / f'benchmark_concurrent_{index}.json')
        if solo and concurrent:
            ratio = concurrent['seconds'] / solo['seconds']
            concurrency.append({'device': index, 'run_id': solo['run_id'], 'solo': solo, 'concurrent': concurrent,
                                'slowdown_ratio': ratio, 'above_10_percent': ratio > 1.1})
    complete_budget_basis = len(estimates) == 16
    p2 = sum(row['P2_gpu_hours_estimate'] for row in estimates)
    p3 = sum(row['P3_gpu_hours_estimate'] for row in estimates)
    profile_checkpoint_gib = sum(sum((item['summary'] or {}).get(key, 0) for key in
        ['best_val_checkpoint_bytes', 'last_checkpoint_bytes', 'task_delta_bytes']) for item in report['P1']) / 2**30
    additional_slice_upper_gib = sum((item['summary'] or {}).get('best_val_checkpoint_bytes', 0)
                                    for item in report['P1']) * 5 * 3 / 2**30
    budget = {'measurement_kind': 'estimated_from_measured_P1', 'basis_configurations': len(estimates),
              'required_configurations': 16, 'complete_basis': complete_budget_basis,
              'per_configuration': estimates, 'P2_gpu_hours': p2 if complete_budget_basis else None,
              'P3_gpu_hours': p3 if complete_budget_basis else None,
              'P2_P3_gpu_hours_with_25_percent_margin': 1.25 * (p2 + p3) if complete_budget_basis else None,
              'ideal_two_gpu_wall_hours_lower_bound': (p2 + p3) / 2 if complete_budget_basis else None,
              'partial_measured_configurations_P2_P3_subtotal_gpu_hours': p2 + p3,
              'centralized_evaluation_gpu_hours': None, 'bounded_diagnostics_gpu_hours': None,
              'billing_price_per_gpu_hour': None, 'currency_cost': None,
              'margin_is_spent_compute': False,
              'routine_checkpoint_seconds_in_epoch_cost': True,
              'external_model_pretraining_included': False,
              'storage_estimate': {'policy': 'best_val + last + metadata per run; shared data/weights once',
                    'P2_archive_policy': 'retain raw logs and selected model identities, delete optimizer only after stage review',
                    'P3_time_slices': 'not_budgeted_until_P2_timing_freeze',
                    'P1_measured_checkpoint_gib': profile_checkpoint_gib,
                    'P2_full_retention_checkpoint_gib': 3 * profile_checkpoint_gib if complete_budget_basis else None,
                    'P3_best_last_delta_estimated_gib': 5 * profile_checkpoint_gib if complete_budget_basis else None,
                    'P1_P2_P3_full_retention_checkpoint_gib': 9 * profile_checkpoint_gib if complete_budget_basis else None,
                    'P3_three_additional_slice_weights_upper_bound_gib': additional_slice_upper_gib if complete_budget_basis else None,
                    'dataset_weight_disk_bytes': sum(a['size_bytes'] for a in load_json(ROOT / 'records/assets.json')['assets']),
                    'storage_plan_must_include_extracted_data_and_25_percent_margin': True}}
    if complete_budget_basis:
        by_configuration = {item['configuration_id']: item for item in estimates}
        schedules = {}
        # 按实测外推时长优先排长任务，再分配到累计负载较低的卡，避免固定方法绑定物理 GPU。
        for phase in ['P2', 'P3']:
            future = [r for r in load_json(ROOT / 'configs/run_matrix.json')['planned_runs'] if r['phase'] == phase]
            def duration(run):
                item = by_configuration[run['configuration_id']]
                return run['epochs'] * item['epoch_seconds'] + item['per_run_fixed_and_first_epoch_excess_seconds']
            loads = [0.0, 0.0]
            assignments = []
            for run in sorted(future, key=lambda r: (-duration(r), r['run_id'])):
                device = 0 if loads[0] <= loads[1] else 1
                seconds = duration(run)
                assignments.append({'run_id': run['run_id'], 'proposed_physical_gpu_index': device,
                                    'start_after_seconds_estimate': loads[device], 'duration_seconds_estimate': seconds})
                loads[device] += seconds
            factor = max([1.0] + [item['slowdown_ratio'] for item in concurrency])
            schedules[phase] = {'assignments': assignments, 'single_gpu_baseline_load_hours': [value / 3600 for value in loads],
                                'two_gpu_wall_hours_before_contention': max(loads) / 3600,
                                'conservative_measured_contention_scenario_factor': factor,
                                'two_gpu_wall_hours_with_contention_scenario': max(loads) * factor / 3600,
                                'assignment_status': 'proposal_not_frozen'}
        budget['two_gpu_schedule_proposal'] = schedules
        budget['contention_investigation_required_before_freeze'] = any(item['above_10_percent'] for item in concurrency)
        budget['two_gpu_wall_hours_with_25_percent_margin'] = 1.25 * sum(
            schedules[phase]['two_gpu_wall_hours_with_contention_scenario'] for phase in ['P2', 'P3'])
    report['concurrency'] = concurrency
    if report['budget_ledger']:
        for item in report['P1']:
            entry = next((entry for entry in reversed(report['budget_ledger']['entries'])
                          if entry['label'] == item['run_id']), None)
            item['parent_monitor'] = entry
    report['P1_completed_runs'] = sum(item['status'] == 'completed' for item in report['P1'])
    report['P1_status'] = 'completed' if report['P1_completed_runs'] == 16 else 'incomplete'
    report['next_budget'] = budget
    (ROOT / 'reports').mkdir(exist_ok=True)
    atomic_json(ROOT / 'records/budget_plan.json', budget)
    atomic_json(ROOT / 'reports/P0_P1_report.json', report)
    with (ROOT / 'reports/P1_summary.csv').open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({'P0': (report['P0'] or {}).get('status'),
                      'P1_completed_runs': report['P1_completed_runs'],
                      'actual_gpu_hours': (report['budget_ledger'] or {}).get('gpu_hours'),
                      'budget_complete_basis': complete_budget_basis}, ensure_ascii=False))


if __name__ == '__main__':
    main()
