"""从回收文本独立复核 P3；不导入训练实现，不加载 checkpoint 或测试数据。"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

REPO = Path(__file__).resolve().parents[1]
SCIENCE = '7247c66175f385e990d61580f1a7ef4f9ef7bf26'
RUNTIME = '700a1adcba85e03896dfafa303959f8d035008aa'
OPS = {'src/tlcls/p3_core.py', 'src/tlcls/p3_budget.py', 'src/p3_schedule.py',
       'src/p3_train.py', 'src/p3_report.py', 'src/p3_acceptance.py'}

def read(p):
    return json.loads(p.read_text(encoding='utf8'))

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def value_hash(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, ensure_ascii=False,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def write(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8', newline='\n')

def csv_write(p, rows):
    with p.open('w', encoding='utf8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n'); w.writeheader(); w.writerows(rows)

def verify(snapshot):
    s = snapshot / 'records/p3'
    manifest = read(snapshot / 'snapshot_manifest.json')
    for name, item in manifest['files'].items():
        p = snapshot / name
        assert p.stat().st_size == item['size_bytes'] and sha(p) == item['sha256'], name
    seal = read(snapshot / 'records/P3_sealed.json')
    for name, key in [('gpu_ledger.json', 'gpu_ledger_sha256'),
            ('run_manifest.json', 'run_manifest_sha256'),
            ('evaluation_manifest.draft.json', 'evaluation_manifest_sha256'),
            ('checkpoint_inventory.json', 'checkpoint_inventory_sha256')]:
        assert sha(s / name) == seal[key]
    assert sha(snapshot / 'selected_hparams.json') == seal['selected_hparams_sha256']
    assert seal['status'] == 'sealed' and seal['run_status_counts'] == {'completed': 80}
    assert not seal['P3_C_authorized'] and not seal['centralized_test_authorized']
    auth = read(s / 'authorization.json')
    supplementary = []
    for name, h in auth['frozen_inputs'].items():
        path = snapshot / name
        if not path.exists():
            path = REPO / name
            supplementary.append(name)
        assert sha(path) == h, name
    receipt = read(s / 'acceptance.json')
    assert receipt['status'] == 'passed' and receipt['six_gates_passed']
    assert receipt['authorization_sha256'] == sha(s / 'authorization.json')
    for n, k in [('cpu_acceptance.json', 'cpu_receipt_sha256'),
                 ('gpu_acceptance.json', 'gpu_receipt_sha256'),
                 ('entry_preflight.json', 'entry_preflight_sha256')]:
        assert sha(s / n) == receipt[k] and read(s / n)['status'] == 'passed'
    identities = {}
    for commit in [SCIENCE, RUNTIME]:
        files = {name: hashlib.sha256(subprocess.check_output(
            ['git', 'show', commit + ':' + name], cwd=REPO)).hexdigest()
            for name in auth['execution_identity']['files']}
        pipeline = {n: files[n] for n in files if n in
            ['src/tlcls/' + x + '.py' for x in ['common', 'data', 'models', 'engine']]}
        identities[commit] = {'files': files, 'scope_sha256': value_hash(files),
                             'mathematical_pipeline_sha256': value_hash(pipeline)}
    changed = [n for n in identities[SCIENCE]['files']
               if identities[SCIENCE]['files'][n] != identities[RUNTIME]['files'][n]]
    assert set(changed) <= OPS
    assert auth['execution_commit'] == RUNTIME and auth['execution_identity'] == identities[RUNTIME]
    assert all(sha(snapshot / n) == h for n, h in identities[RUNTIME]['files'].items())
    selected = read(snapshot / 'selected_hparams.json')['configurations']
    assets = {a['key']: a for a in read(snapshot / 'records/assets.json')['assets']}
    runs = read(s / 'run_manifest.json')['runs']
    assert len(runs) == len({r['run_id'] for r in runs}) == 80
    assert set(seal['run_ids']) == {r['run_id'] for r in runs}
    groups = defaultdict(list)
    ledger = read(s / 'gpu_ledger.json'); attempts = list(ledger['attempts'].values())
    assert ledger['cap_gpu_seconds'] == 576000
    assert all(a['status'] == 'finalized' and a['charged_gpu_seconds'] >= 0 for a in attempts)
    used = sum(a['charged_gpu_seconds'] for a in attempts)
    assert used < 576000 - 5272.695245229139
    for device in [0, 1]:
        queue = sorted((a for a in attempts if a['device'] == device), key=lambda a: a['started_monotonic'])
        assert all(a['ended_monotonic'] <= b['started_monotonic'] for a, b in zip(queue, queue[1:]))
        assert all(math.isclose(a['charged_gpu_seconds'], a['ended_monotonic'] - a['started_monotonic'], abs_tol=1e-5) for a in queue)
    inventory = read(s / 'checkpoint_inventory.json')
    cps = {i['path']: i for i in inventory['items']}
    assert len(cps) == len(inventory['items'])
    prefixes = read(s / 'prefix_manifest.json')['items']
    evaluations = read(s / 'evaluation_manifest.draft.json')['evaluations']
    assert len({e['evaluation_id'] for e in evaluations}) == len(evaluations)
    assert all(cps[e['checkpoint']]['sha256'] == e['checkpoint_sha256'] for e in evaluations)
    frozen = read(s / 'time_budget_frozen.json')
    assert frozen['source_sha256'] == sha(snapshot / 'reports/P3_proposal.json')
    proposal = read(snapshot / 'reports/P3_proposal.json')
    for key, g in frozen['groups'].items():
        assert g['B_seconds'] == proposal['common_time_slices'][key]['B_seconds_proposal']
        assert g['slice_seconds'] == [f * g['B_seconds'] for f in [.25, .5, 1.]]
    totals = Counter(); sources = Counter(); final_sources = Counter(); table = []; epochs = []; prefix_table = []
    samplers = defaultdict(dict); heads = defaultdict(set); resumed = []
    for c in runs:
        rid = c['run_id']; d = snapshot / 'runs' / rid
        summary = read(d / 'summary.json'); science = read(d / 'scientific.json')
        config = read(d / 'config.json')
        assert config == c
        assert summary['status'] == 'completed' and summary['completed_epochs'] == 50
        assert not summary['test_evaluator_constructed'] and summary['test_metrics'] is None
        own_attempts = sorted((a for a in attempts if a['run_id'] == rid), key=lambda a: a['started_monotonic'])
        launch_records = [read(d / 'attempts' / (a['attempt_id'] + '.json')) for a in own_attempts]
        assert launch_records[0]['resumed_generation'] is None and not launch_records[0]['launch']['resume']
        assert summary['fresh_initialization'] == (len(own_attempts) == 1)
        for launch in launch_records[1:]:
            assert launch['launch']['resume'] and launch['resumed_generation']
            ref = launch['resumed_generation']
            assert sha(d / ref['path']) == ref['sha256']
            resumed.append({'run_id': rid, 'completed_epoch': read(d / ref['path'])['completed_epoch']})
        assert sha(d / 'summary.json') == seal['summary_sha256'][rid]
        assert c['learning_rate'] == selected[c['configuration_id']]['learning_rate']
        assert c['checkpoint_revision'] == assets[c['model_key'] + '_model.safetensors']['revision']
        assert c['checkpoint_sha256'] == assets[c['model_key'] + '_model.safetensors']['sha256']
        assert (c['epochs'], c['scheduler_horizon_epochs'], c['warmup_epochs']) == (50, 50, 5)
        assert (c['micro_batch_size'], c['gradient_accumulation_steps'], c['effective_batch_size']) == (32, 4, 128)
        assert not c['resume_pilot_checkpoint'] and not c['test_during_training']
        assert c['head_lr_multiplier'] == (10 if c['method'] == 'full_ft' else 1)
        assert c['selected_hparams_sha256'] == seal['selected_hparams_sha256']
        body = science['scientific_config']
        assert body['configuration'] == {k: v for k, v in c.items() if k != 'physical_gpu_index'}
        assert body['implementation'] == identities[SCIENCE]
        assert all(sha(snapshot / n) == h for n, h in body['inputs'].items())
        assert value_hash(body) == science['scientific_config_sha256'] == summary['scientific_config_sha256']
        assert summary['implementation'] == identities[summary['execution_commit']]
        assert summary['scientific_implementation_basis'] == identities[SCIENCE]
        mapping = summary['operational_compatibility']
        assert mapping['actual_execution_identity'] == summary['implementation']
        if summary['execution_commit'] == SCIENCE:
            assert mapping['cross_commit_resume_allowed'] is False
        else:
            assert mapping['scientific_base_commit'] == SCIENCE
            assert set(mapping['changed_operational_files']) == set(changed)
        final_sources[summary['execution_commit']] += 1
        groups[c['configuration_id']].append(c['seed'])
        rows = [json.loads(x) for x in (d / 'epochs.jsonl').read_text().splitlines()]
        assert len(rows) == 50
        pointer = read(d / 'current.json'); chain = []
        while pointer:
            assert sha(d / pointer['path']) == pointer['sha256']
            commit = read(d / pointer['path']); chain.append((pointer, commit)); pointer = commit['previous_commit']
        chain.reverse(); assert len(chain) == 50
        n = 45000 if c['dataset'] == 'cifar100' else 65931
        nv = 5000 if c['dataset'] == 'cifar100' else 7326
        previous_clock = -1
        for e, ((ref, commit), row) in enumerate(zip(chain, rows), 1):
            assert commit['completed_epoch'] == row['epoch'] == e and commit['epoch_row'] == row
            assert row['train']['samples'] == n and row['validation']['samples'] == nv
            assert row['cumulative_optimizer_updates'] == e * math.ceil(n / 128)
            assert row['scientific_config_sha256'] == science['scientific_config_sha256']
            assert commit['cumulative_gpu_seconds_at_commit'] >= previous_clock
            previous_clock = commit['cumulative_gpu_seconds_at_commit']
            source = row['runtime_execution_commit']; assert source in identities
            assert row['runtime_execution_identity_sha256'] == identities[source]['scope_sha256']
            assert ledger['attempts'][row['attempt_id']]['run_id'] == rid
            eligible = read(d / 'eligible' / f'epoch_{e:03d}.json')
            completion = d / 'completions' / f'epoch_{e:03d}.json'
            assert eligible['commit'] == ref and eligible['durable_completion_marker_sha256'] == sha(completion)
            assert read(completion)['all_checkpoint_bytes_hashed_and_fsynced']
            assert eligible['transaction_completed_gpu_seconds'] >= previous_clock
            sources[source] += 1
            key = (c['dataset'], c['model_key'], c['seed'], e)
            samplers[key][c['method']] = row['sampler_order_sha256']
            totals['train_samples'] += n; totals['validation_samples'] += nv
            epochs.append({'run_id': rid, 'configuration_id': c['configuration_id'], 'seed': c['seed'],
                'epoch': e, 'train_loss': row['train']['loss'], 'validation_loss': row['validation']['loss'],
                'train_top1': row['train']['top1_percent'], 'validation_top1': row['validation']['top1_percent'],
                'train_seconds': row['train_seconds'], 'validation_seconds': row['validation_seconds'],
                'checkpoint_seconds': row['checkpoint_seconds'], 'execution_commit': source})
        totals['updates'] += rows[-1]['cumulative_optimizer_updates']
        best = max(rows, key=lambda row: (row['validation']['top1_percent'], -row['epoch']))
        assert best['epoch'] == summary['selected_epoch'] == chain[-1][1]['best_epoch']
        assert best['validation']['top1_percent'] == summary['best_val_top1_percent']
        for field, k in [('best_checkpoint', 'best'), ('last_checkpoint', 'last')]:
            cp = summary[field]; assert cp == chain[-1][1][k]
            assert cps['runs/' + rid + '/' + cp['path']]['sha256'] == cp['sha256']
        heads[(c['dataset'], c['model_key'], c['seed'])].add(summary['model_identity']['initial_classifier_sha256'])
        for p in [p for p in prefixes if p['run_id'] == rid]:
            assert p['status'] == 'available'
            assert p['budget_seconds'] == p['budget_fraction'] * frozen['groups'][c['dataset'] + '-' + c['model_key']]['B_seconds']
            eligible = [read(d / 'eligible' / f'epoch_{e:03d}.json') for e in range(1, 51)]
            legal = [x for x in eligible if x['transaction_completed_gpu_seconds'] <= p['budget_seconds']]
            assert p['completion'] == max(legal, key=lambda x: x['epoch'])
            assert p['checkpoint'] == p['completion']['checkpoint']
            assert cps['runs/' + rid + '/' + p['checkpoint']['path']]['sha256'] == p['checkpoint']['sha256']
            row = rows[p['completed_epoch'] - 1]
            prefix_table.append({'run_id': rid, 'configuration_id': c['configuration_id'], 'seed': c['seed'],
                'budget_fraction': p['budget_fraction'], 'budget_seconds': p['budget_seconds'],
                'epoch': p['completed_epoch'], 'validation_top1': row['validation']['top1_percent'],
                'checkpoint_sha256': p['checkpoint']['sha256']})
        table.append({'run_id': rid, 'configuration_id': c['configuration_id'], 'seed': c['seed'],
            'learning_rate': c['learning_rate'], 'best_epoch': best['epoch'],
            'best_validation_top1': summary['best_val_top1_percent'],
            'last_validation_top1': rows[-1]['validation']['top1_percent'],
            'GPU_hours': sum(a['charged_gpu_seconds'] for a in attempts if a['run_id'] == rid) / 3600,
            'peak_allocated_GiB': summary['peak_allocated_bytes'] / 2**30,
            'peak_reserved_GiB': summary['peak_reserved_bytes'] / 2**30,
            'execution_commit_final_attempt': summary['execution_commit']})
    assert len(groups) == 16 and all(sorted(v) == [17, 42, 2026, 3407, 65537] for v in groups.values())
    assert all(len(v) == 4 and len(set(v.values())) == 1 for v in samplers.values())
    assert all(len(v) == 1 for v in heads.values())
    assert len(prefixes) == len(prefix_table) == 240 and len(evaluations) == 400
    preserved = {}
    for n in ['resume_decision.json', 'slowdown_resume_decision.json']:
        v = read(s / n)
        for rid, fs in v['preserved_completed_runs'].items():
            for name, h in fs.items():
                assert sha(snapshot / 'runs' / rid / name) == h
        preserved[n] = len(v['preserved_completed_runs'])
    result = {'schema': 'P3_independent_delivery_audit_v1', 'status': 'passed',
        'snapshot_files_verified': len(manifest['files']), 'seal_sha256': sha(snapshot / 'records/P3_sealed.json'),
        'frozen_inputs_verified_from_P2_archive_when_not_exported': supplementary,
        'gpu_ledger_sha256': sha(s / 'gpu_ledger.json'), 'runs_completed': 80, 'configurations': 16,
        'committed_epochs': len(epochs), 'committed_training_samples': totals['train_samples'],
        'committed_validation_samples': totals['validation_samples'], 'optimizer_updates': totals['updates'],
        'all_attempt_physical_samples': 'UNKNOWN: uncommitted interrupted work is not fully logged',
        'GPU_hours': used / 3600, 'attempts': len(attempts),
        'attempt_kind_counts': dict(Counter(a['kind'] for a in attempts)),
        'attempt_termination_counts': dict(Counter(a['termination_reason'] for a in attempts)),
        'actual_epoch_execution_commits': dict(sources), 'final_attempt_execution_commits': dict(final_sources),
        'scientific_base_commit': SCIENCE, 'changed_operational_files': changed,
        'preserved_completed_run_hashes': preserved, 'checkpoint_count': len(cps),
        'checkpoint_bytes': inventory['total_bytes'], 'prefixes_available': len(prefix_table),
        'evaluation_roles': len(evaluations), 'distinct_prediction_cache_keys': len({e['inference_cache_key'] for e in evaluations}),
        'matching_initial_head_groups': len(heads), 'matching_sampler_epoch_groups': len(samplers),
        'fresh_initial_runs_verified_from_first_attempt': 80, 'resumed_at_committed_epochs': resumed,
        'fresh_initialization_summary_field_semantics': 'last attempt only; false for four legal infrastructure resumes',
        'P3_checkpoint_binary_hashes': 'verified on server by sealed terminal audit; only inventory consistency checked locally',
        'centralized_test_authorized': False, 'test_predictions_generated': False}
    return result, table, epochs, prefix_table

def main():
    p = argparse.ArgumentParser(); p.add_argument('--snapshot', type=Path, required=True); args = p.parse_args()
    result, table, epochs, prefixes = verify(args.snapshot)
    write(REPO / 'records/p3/delivery_audit.json', result)
    csv_write(REPO / 'reports/P3_validation_runs.csv', table)
    csv_write(REPO / 'reports/P3_epoch_curves.csv', epochs)
    csv_write(REPO / 'reports/P3_prefix_validation.csv', prefixes)
    configs = []
    for key in sorted({r['configuration_id'] for r in table}):
        rows = [r for r in table if r['configuration_id'] == key]
        configs.append({'configuration_id': key, 'n': len(rows), 'learning_rate': rows[0]['learning_rate'],
            'best_validation_mean': statistics.mean(r['best_validation_top1'] for r in rows),
            'best_validation_SD': statistics.stdev(r['best_validation_top1'] for r in rows),
            'last_validation_mean': statistics.mean(r['last_validation_top1'] for r in rows),
            'last_validation_SD': statistics.stdev(r['last_validation_top1'] for r in rows),
            'GPU_hours': sum(r['GPU_hours'] for r in rows)})
    csv_write(REPO / 'reports/P3_validation_configurations.csv', configs)
    print(json.dumps(result, ensure_ascii=False))

if __name__ == '__main__':
    main()
