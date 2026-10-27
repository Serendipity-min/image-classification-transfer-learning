"""从已核验文本快照复算交付；不连接服务器、不读取模型或原始预测。"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [17, 42, 2026, 3407, 65537]
ROLES = ['primary_best_val', 'secondary_last', 'budget_prefix']


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8', newline='\n')


def table(path):
    with path.open(encoding='utf8') as file:
        return list(csv.DictReader(file))


def save_csv(path, rows):
    with path.open('w', encoding='utf8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--storage', type=Path, required=True)
    args = parser.parse_args(); snapshot = args.snapshot.resolve()
    manifest = read(snapshot / 'snapshot_manifest.json')
    assert not manifest['launch_only_NOT_FINAL']
    for name, item in manifest['files'].items():
        path = snapshot / name
        assert path.resolve().is_relative_to(snapshot) and path.stat().st_size == item['size_bytes'] and sha(path) == item['sha256']
    s = snapshot / 'records/p3c'
    audit = read(s / 'independent_delivery_audit.json'); seal = read(s / 'P3C_sealed.json')
    for field, name in [('shared_gpu_ledger_sha256', 'records/p3/gpu_ledger.json'),
                        ('evaluation_ledger_sha256', 'records/p3c/evaluation_ledger.json'),
                        ('evaluation_receipts_sha256', 'records/p3c/evaluation_receipts.jsonl'),
                        ('independent_audit_sha256', 'records/p3c/independent_delivery_audit.json')]:
        assert sha(snapshot / name) == seal[field]
    assert audit['status'] == 'passed' and audit['prediction_metrics_independently_recomputed'] == 400
    assert sha(ROOT / 'records/P3_sealed.json') == audit['P3B_seal_sha256']
    frozen = ROOT / 'records/p3/evaluation_manifest.draft.json'
    assert sha(frozen) == audit['frozen_manifest_sha256']
    items = {x['evaluation_id']: x for x in read(frozen)['evaluations']}
    runs = {x['run_id']: x for x in read(ROOT / 'records/p3/run_manifest.json')['runs']}
    outputs = [read(p) for p in (s / 'results').glob('*/completed.json')]
    receipts = [json.loads(line) for line in (s / 'evaluation_receipts.jsonl').read_text().splitlines()]
    assert {x['evaluation_id']: x for x in outputs} == {x['evaluation_id']: x for x in receipts}
    assert len(outputs) == len(items) == 400 and len({x['evaluation_id'] for x in outputs}) == 400
    rows = table(snapshot / 'reports/P3_C_逐角色测试指标.csv')
    csv_by_id = {x['evaluation_id']: x for x in rows}
    assert len(csv_by_id) == 400
    for output in outputs:
        item = items[output['evaluation_id']]; config = runs[item['run_id']]; metric = output['metrics']
        for name in ['run_id', 'checkpoint_sha256', 'evaluation_role', 'inference_cache_key']:
            assert output[name] == item[name]
        assert output['manifest_sha256'] == sha(frozen) and output['status'] == 'completed' and not output['test_used_for_selection']
        confusion = np.asarray(metric['confusion_matrix'], dtype=np.int64)
        n = 10000 if config['dataset'] == 'cifar100' else 26032
        assert metric['n'] == n == confusion.sum() and metric['correct'] == int(np.trace(confusion))
        assert metric['top1_percent'] == 100 * metric['correct'] / n
        recall = np.diag(confusion) / confusion.sum(1)
        f1 = 2 * np.diag(confusion) / (confusion.sum(0) + confusion.sum(1))
        assert abs(recall.mean() - metric['balanced_accuracy']) < 1e-12
        assert abs(f1.mean() - metric['macro_f1']) < 1e-12 and np.isfinite(metric['nll'])
        row = csv_by_id[output['evaluation_id']]
        assert row['configuration_id'] == config['configuration_id'] and int(row['seed']) == config['seed']
        for name in ['n', 'correct', 'top1_percent', 'macro_f1', 'balanced_accuracy', 'nll', 'GPU_hours_role_transaction']:
            assert abs(float(row[name]) - metric[name]) < (1e-5 if name == 'nll' else 1e-12)
        assert row['predictions_sha256'] == output['predictions_sha256']
    counts = Counter(x['evaluation_role'] for x in outputs)
    assert counts == {'primary_best_val':80, 'secondary_last':80, 'budget_prefix':240}
    dispatch = read(s / 'dispatch.json')['claims']
    ordered = sorted(items, key=lambda k:(ROLES.index(items[k]['evaluation_role']), items[k]['run_id'], items[k]['budget_fraction'] or 0))
    assert ordered == read(s / 'execution_order.json')['evaluation_ids']
    assert ordered == sorted(dispatch, key=lambda k:dispatch[k]['ordinal'])
    for previous, following in zip(ROLES, ROLES[1:]):
        end = max(x['completed_unix'] for x in outputs if x['evaluation_role'] == previous)
        start = min(dispatch[x['evaluation_id']]['claimed_unix'] for x in outputs if x['evaluation_role'] == following)
        assert start >= end
    shared = read(snapshot / 'records/p3/gpu_ledger.json'); baseline = read(s / 'p3b_gpu_ledger.sealed.json')
    assert len(baseline['attempts']) == 92
    assert all(shared['attempts'][k] == value for k, value in baseline['attempts'].items())
    own = [a for a in shared['attempts'].values() if a['kind'].startswith('p3c_')]
    phase = read(s / 'evaluation_ledger.json')
    assert set(phase['attempt_ids']) == {a['attempt_id'] for a in own}
    assert len(own) == 4 and all(a['status']=='finalized' and a['returncode']==0 and a['termination_reason']=='completed' for a in own)
    assert {a['execution_commit'] for a in own} == {'700a1adcba85e03896dfafa303959f8d035008aa'}
    for device in [0, 1]:
        attempts = sorted([a for a in own if a['device']==device], key=lambda a:a['started_monotonic'])
        assert all(a['ended_monotonic'] <= b['started_monotonic'] for a,b in zip(attempts, attempts[1:]))
    seconds = sum(a['charged_gpu_seconds'] for a in own)
    total = sum(a['charged_gpu_seconds'] for a in shared['attempts'].values())
    assert seconds <= 10800 and total <= 576000
    assert abs(seconds/3600 - audit['P3C_GPU_hours']) < 1e-12 and abs(total/3600 - audit['P3_total_GPU_hours']) < 1e-12
    tip = phase['chain_tip_sha256']; seen = set()
    while tip:
        assert tip not in seen; seen.add(tip)
        path = s / 'ledger_chain' / (tip + '.json'); node = read(path)
        encoded = json.dumps(node, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        assert hashlib.sha256(encoded).hexdigest() == tip
        assert node['P3_B_baseline_sha256'] == sha(s / 'p3b_gpu_ledger.sealed.json')
        tip = node['previous_sha256']
    assert len(seen) == audit['signed_append_chain_links'] == 12
    storage = read(args.storage)
    assert storage['raw_prediction_files'] == 400 and storage['exit_reserve_satisfied'] and not storage['wrapper_alive']
    assert {x['evaluation_id']:x['sha256'] for x in storage['raw_prediction_inventory']} == {x['evaluation_id']:x['predictions_sha256'] for x in outputs}
    cpu = read(s/'fault_injection.json'); assert cpu['status']=='passed' and cpu['tests_run']==28
    assert all(sha(ROOT/'tools'/name)==value==storage['signed_runtime_tool_sha256']['tools/'+name] for name,value in cpu['source_sha256'].items())
    primary = [r for r in rows if r['role']=='primary_best_val']
    pair_rows = []; tests = []
    for group, method in [('cifar100-vitb16','linear_probe'),('svhn-r50','adapter')]:
        differences=[]
        for seed in SEEDS:
            a = float(next(r['top1_percent'] for r in primary if r['configuration_id']==group+'-'+method and int(r['seed'])==seed))
            b = float(next(r['top1_percent'] for r in primary if r['configuration_id']==group+'-full_ft' and int(r['seed'])==seed))
            differences.append(a-b)
            pair_rows.append({'comparison':group+'-'+method+'-minus-full_ft','seed':seed,'method_top1':a,'full_ft_top1':b,'difference_pp':a-b})
        mean=statistics.mean(differences); sd=statistics.stdev(differences); half=float(stats.t.ppf(.975,4))*sd/5**.5
        tests.append({'comparison':group+'-'+method+'-minus-full_ft','mean_difference_pp':mean,'sample_SD_pp':sd,
            'CI95_low_pp':mean-half,'CI95_high_pp':mean+half,'paired_t_two_sided_p':float(stats.ttest_rel(
                [p['method_top1'] for p in pair_rows[-5:]], [p['full_ft_top1'] for p in pair_rows[-5:]]).pvalue)})
    rank=sorted(range(2),key=lambda i:tests[i]['paired_t_two_sided_p']); floor=0
    for position,i in enumerate(rank):
        floor=max(floor,min(1,(2-position)*tests[i]['paired_t_two_sided_p'])); tests[i]['Holm_adjusted_p']=floor
    server_tests=table(snapshot/'reports/P3_C_逐种子配对统计.csv')
    for local,server in zip(tests,server_tests):
        assert local['comparison']==server['comparison'] and int(server['n_complete_pairs'])==5
        assert all(abs(value-float(server[key]))<1e-11 for key,value in local.items() if key!='comparison')
    configurations=[]
    for config in sorted({r['configuration_id'] for r in rows}):
        for role in ROLES:
            for fraction in ([.25,.5,1.] if role=='budget_prefix' else [None]):
                subset=[r for r in rows if r['configuration_id']==config and r['role']==role and (fraction is None or float(r['budget_fraction'])==fraction)]
                assert sorted(int(r['seed']) for r in subset)==SEEDS
                result={'configuration_id':config,'role':role,'budget_fraction':fraction,'n_seeds':5}
                for metric in ['top1_percent','macro_f1','balanced_accuracy','nll']:
                    values=[float(r[metric]) for r in subset]
                    result[metric+'_mean']=statistics.mean(values);result[metric+'_sample_SD']=statistics.stdev(values)
                configurations.append(result)
    destination=ROOT/'evidence/P3C_final_20261028';destination.mkdir(parents=True,exist_ok=True)
    shutil.copy2(snapshot.parent/'text_evidence.tar.gz',destination/'text_evidence.tar.gz')
    shutil.copy2(snapshot/'snapshot_manifest.json',destination/'snapshot_manifest.json')
    for name in manifest['files']:
        if '/results/' in name:continue
        target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(snapshot/name,target)
    shutil.copy2(args.storage,ROOT/'records/p3c/storage_closeout.json')
    save_csv(ROOT/'reports/P3_C_配置汇总.csv',configurations)
    save_csv(ROOT/'reports/P3_C_主比较差值.csv',pair_rows)
    save_csv(ROOT/'reports/P3_C_GPU逐attempt台账.csv',own)
    role_cost={role:sum(x['metrics']['GPU_hours_role_transaction'] for x in outputs if x['evaluation_role']==role) for role in ROLES}
    acceptance={'schema':'P3C_local_delivery_acceptance_v1','status':'PASS','completed':400,'by_role':dict(counts),
        'text_archive_sha256':sha(destination/'text_evidence.tar.gz'),'text_archive_bytes':(destination/'text_evidence.tar.gz').stat().st_size,
        'text_snapshot_files_verified':len(manifest['files']),'sealed_artifact_hashes_verified':4,
        'confusion_count_top1_f1_balanced_recomputed_locally':400,'NLL_independently_recomputed_server_side':400,
        'paired_statistics_recomputed_locally':tests,'all_five_seeds_per_configuration_role':True,
        'frozen_dispatch_order_and_role_completion_barriers_verified':True,'P3B_92_attempts_unchanged':True,
        'chain_content_hash_links_verified_locally':len(seen),'chain_HMAC_verified_server_side':True,
        'CPU_acceptance_tests':28,'P3C_attempts':4,'P3C_failures':0,'P3C_technical_retries':0,
        'P3C_GPU_hours':seconds/3600,'P3_total_GPU_hours':total/3600,
        'P3C_cap_GPU_hours':3,'total_cap_GPU_hours':160,'P3C_remaining_GPU_hours':3-seconds/3600,
        'total_remaining_GPU_hours':160-total/3600,'synthetic_acceptance_GPU_hours':sum(a['charged_gpu_seconds'] for a in own if a['kind']=='p3c_gpu_acceptance')/3600,
        'formal_reservation_GPU_hours':sum(a['charged_gpu_seconds'] for a in own if a['kind']=='p3c_evaluation')/3600,
        'role_transaction_GPU_hours':role_cost,'formal_startup_authorization_idle_exit_overhead_GPU_hours':sum(a['charged_gpu_seconds'] for a in own if a['kind']=='p3c_evaluation')/3600-sum(role_cost.values()),
        'peak_GPU_allocated_bytes_after_model_load':max(x['metrics']['peak_allocated_bytes'] for x in outputs),
        'peak_GPU_reserved_bytes_after_model_load':max(x['metrics']['peak_reserved_bytes'] for x in outputs),
        'CPU_process_RSS_peak':'not sampled by P3C guardian; ledger zeros do not mean zero usage',
        'storage_receipt_sha256':sha(ROOT/'records/p3c/storage_closeout.json'),
        'original_predictions_and_checkpoint_downloaded':False,'live_authorization_or_key_published':False,
        'G2_independent_issuance_waived_by_explicit_PI':True,'new_training_or_test_driven_selection':False,
        'paper_finalization_authorized':False,'additional_test_authorized':False}
    write(ROOT/'records/p3c/delivery_acceptance.json',acceptance)
    write(destination/'collection_receipt.json',{'archive_sha256':acceptance['text_archive_sha256'],
        'archive_size_bytes':acceptance['text_archive_bytes'],'verified_files':len(manifest['files']),
        'server_export_id':snapshot.parent.name,'all_original_archive_bytes_preserved':True,
        'raw_predictions_downloaded':False,'checkpoint_downloaded':False,'key_downloaded':False,
        'live_test_authorization_published':False,'storage_receipt_collected_separately':acceptance['storage_receipt_sha256']})
    print(json.dumps({'status':'PASS','completed':400,'P3C_GPU_hours':seconds/3600,'P3_total_GPU_hours':total/3600,'role_cost':role_cost,'formal_overhead_hours':acceptance['formal_startup_authorization_idle_exit_overhead_GPU_hours']},ensure_ascii=False))


if __name__=='__main__':
    main()
