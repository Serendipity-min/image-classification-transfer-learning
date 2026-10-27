"""P3-C 授权后重算逐预测指标及预注册配对统计；原始预测留服务器。"""
import csv
import json
from pathlib import Path
import statistics
import sys

ROOT = Path('/root/user_data/tl_cls_v1')
sys.path.insert(0, str(ROOT / 'src'))

def paired_statistics(rows):
    from scipy import stats
    seeds = [17, 42, 2026, 3407, 65537]
    comparisons = [('cifar100-vitb16', 'linear_probe'), ('svhn-r50', 'adapter')]
    results = []
    for group, method in comparisons:
        values = []
        for seed in seeds:
            a = [r for r in rows if r['configuration_id'] == group + '-' + method and r['seed'] == seed]
            b = [r for r in rows if r['configuration_id'] == group + '-full_ft' and r['seed'] == seed]
            if len(a) != 1 or len(b) != 1:
                continue
            values.append(a[0]['top1_percent'] - b[0]['top1_percent'])
        result = {'comparison': group + '-' + method + '-minus-full_ft', 'n_required': 5,
            'n_complete_pairs': len(values), 'status': 'complete' if len(values) == 5 else 'incomplete',
            'mean_difference_pp': None, 'sample_SD_pp': None, 'CI95_low_pp': None, 'CI95_high_pp': None,
            'paired_t_two_sided_p': None, 'Holm_adjusted_p': None, 'superiority_Holm_alpha_0_05': None,
            'noninferiority_margin_pp': -.5 if group == 'cifar100-vitb16' else None,
            'noninferiority_CI95_lower_rule': None}
        if len(values) == 5:
            mean = statistics.mean(values); sd = statistics.stdev(values)
            half = float(stats.t.ppf(.975, 4)) * sd / 5**.5
            p = float(stats.ttest_1samp(values, 0).pvalue) if sd else (1. if mean == 0 else 0.)
            result.update(mean_difference_pp=mean, sample_SD_pp=sd, CI95_low_pp=mean-half,
                CI95_high_pp=mean+half, paired_t_two_sided_p=p)
            if group == 'cifar100-vitb16':
                result['noninferiority_CI95_lower_rule'] = mean - half > -.5
        results.append(result)
    if all(r['status'] == 'complete' for r in results):
        ranked = sorted(range(2), key=lambda i: results[i]['paired_t_two_sided_p'])
        first = min(1., results[ranked[0]]['paired_t_two_sided_p'] * 2)
        results[ranked[0]]['Holm_adjusted_p'] = first
        results[ranked[1]]['Holm_adjusted_p'] = max(first, results[ranked[1]]['paired_t_two_sided_p'])
        for result in results:
            result['superiority_Holm_alpha_0_05'] = result['Holm_adjusted_p'] < .05 and result['mean_difference_pp'] > 0
    return results

def independent_metrics(logits, labels):
    import numpy as np
    prediction = logits.argmax(1); n, classes = logits.shape
    confusion = np.bincount(labels * classes + prediction, minlength=classes**2).reshape(classes, classes)
    tp = np.diag(confusion).astype(float)
    recall = np.divide(tp, confusion.sum(1), out=np.zeros(classes), where=confusion.sum(1) != 0)
    f1 = np.divide(2*tp, confusion.sum(0)+confusion.sum(1), out=np.zeros(classes),
        where=(confusion.sum(0)+confusion.sum(1)) != 0)
    x = logits.astype(np.float64); maximum = x.max(axis=1)
    log_denominator = maximum + np.log(np.exp(x-maximum[:,None]).sum(axis=1))
    return {'n': n, 'correct': int((prediction == labels).sum()),
        'top1_percent': 100 * int((prediction == labels).sum()) / n,
        'macro_f1': float(f1.mean()), 'balanced_accuracy': float(recall.mean()),
        'nll': float((log_denominator - x[np.arange(n), labels]).mean()),
        'confusion_matrix': confusion.tolist()}

def csv_write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('',encoding='utf8');return
    with path.open('w', encoding='utf8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(rows)

def main():
    import numpy as np
    from p3c_runtime import read, write, sha, canonical, verify_output, verify_chain, Phase
    from p3c_supervise import signed_scope
    from tlcls.p3_evaluation import require_test_authorization
    key, auth = signed_scope()
    manifest = require_test_authorization(ROOT / 'records/p3/evaluation_manifest.draft.json')
    stage = ROOT / 'records/p3c'; configs = {r['run_id']: r for r in read(ROOT / 'records/p3/run_manifest.json')['runs']}
    all_rows = []; missing = []
    for item in manifest['evaluations']:
        value = verify_output(ROOT, item, auth['evaluation_manifest_sha256'])
        if value is None:
            missing.append(item['evaluation_id']); continue
        config = configs[item['run_id']]
        with np.load(ROOT / value['predictions_path'], allow_pickle=False) as data:
            logits, labels = data['logits'], data['labels']
            assert np.isfinite(logits).all() and len(np.unique(data['sample_id'])) == len(labels)
            assert np.array_equal(data['prediction'], logits.argmax(1))
            actual = independent_metrics(logits, labels)
        for k in ['n', 'correct', 'top1_percent', 'macro_f1', 'balanced_accuracy', 'nll', 'confusion_matrix']:
            expected = value['metrics'][k]
            if k in ['nll', 'macro_f1', 'balanced_accuracy']:
                assert abs(actual[k]-expected) < (1e-5 if k == 'nll' else 1e-12)
            else:
                assert actual[k] == expected
        all_rows.append({'evaluation_id':item['evaluation_id'], 'run_id':item['run_id'],
            'configuration_id':config['configuration_id'], 'seed':config['seed'], 'role':item['evaluation_role'],
            'budget_fraction':item.get('budget_fraction'), 'checkpoint_sha256':item['checkpoint_sha256'],
            **{k:actual[k] for k in actual if k != 'confusion_matrix'},
            'GPU_hours_role_transaction':value['metrics']['GPU_hours_role_transaction'],
            'predictions_sha256':value['predictions_sha256']})
    shared = read(ROOT / 'records/p3/gpu_ledger.json')
    assert all(a['status'] == 'finalized' for a in shared['attempts'].values())
    baseline = read(stage / 'p3b_gpu_ledger.sealed.json')
    assert all(shared['attempts'][i] == a for i,a in baseline['attempts'].items())
    phase_used, total = Phase(ROOT, key).used(); assert phase_used <= 10800 and total <= 576000
    links = verify_chain(ROOT, key)
    primary = [r for r in all_rows if r['role'] == 'primary_best_val']
    statistics_rows = paired_statistics(primary)
    csv_write(ROOT / 'reports/P3_C_逐角色测试指标.csv', all_rows)
    csv_write(ROOT / 'reports/P3_C_逐种子配对统计.csv', statistics_rows)
    csv_write(ROOT / 'reports/P3_C_主比较逐种子.csv', [r for r in primary if r['configuration_id'] in
        ['cifar100-vitb16-linear_probe','cifar100-vitb16-full_ft','svhn-r50-adapter','svhn-r50-full_ft']])
    audit = {'schema':'P3C_independent_delivery_v1','status':'passed' if not missing else 'incomplete',
        'completed':len(all_rows),'required':400,'missing_evaluation_ids':missing,
        'P3C_GPU_hours':phase_used/3600,'P3_total_GPU_hours':total/3600,
        'all_attempts_settled':True,'prediction_metrics_independently_recomputed':len(all_rows),
        'P3B_historical_attempts_unchanged':True,'signed_append_chain_links':links,
        'P3B_seal_sha256':auth['seal_sha256'],'frozen_manifest_sha256':auth['evaluation_manifest_sha256'],
        'new_training_or_test_driven_selection':False,'raw_predictions_remain_on_server':True,
        'paper_finalization_authorized':False}
    write(stage / 'independent_delivery_audit.json', audit)
    report = '# P3-C 官方测试结果\n\n'
    report += f"状态：{audit['status']}；完成{len(all_rows)}/400；新增{phase_used/3600:.6f}/3 GPU-hours；P3总{total/3600:.6f}/160。\n\n"
    report += '主指标为预先在验证集选择的best checkpoint的官方test Top-1。没有使用test重新选参或选模。原始预测和混淆矩阵留服务器，完整逐角色指标及逐seed数据见CSV。\n\n'
    report += '|配置（主best）|n/5|Test Top-1均值 ± 样本SD (%)|\n|---|---:|---:|\n'
    for config in sorted({r['configuration_id'] for r in primary}):
        values = [r['top1_percent'] for r in primary if r['configuration_id']==config]
        report += f"|{config}|{len(values)}|{statistics.mean(values):.3f} ± {statistics.stdev(values) if len(values)>1 else float('nan'):.3f}|\n"
    report += '\n两项预注册比较（其余均探索性）：\n\n'
    for value in statistics_rows:
        report += '- ' + json.dumps(value,ensure_ascii=False,allow_nan=False) + '\n'
    report += '\n双侧配对t检验仅用于两项固定比较，按Holm校正；非劣性只对CIFAR/ViT LP−FT用95%双侧区间下界严格大于−0.5pp判定。预算前缀属于同轨迹重复测量，不能当作独立样本。缺少任一配对seed时不作完整主推断。\n'
    (ROOT / 'reports/P3_C_正式测试结果.md').write_text(report,encoding='utf8')
    if not missing:
        write(stage / 'P3C_sealed.json', {'schema':'P3C_seal_v1','status':'sealed_waiting_PI_review',
            'P3B_seal_sha256':auth['seal_sha256'],'shared_gpu_ledger_sha256':sha(ROOT/'records/p3/gpu_ledger.json'),
            'evaluation_ledger_sha256':sha(stage/'evaluation_ledger.json'),
            'evaluation_receipts_sha256':sha(stage/'evaluation_receipts.jsonl'),
            'independent_audit_sha256':sha(stage/'independent_delivery_audit.json'),'paper_finalization_authorized':False})
    print(json.dumps(audit,ensure_ascii=False))

if __name__ == '__main__':
    main()
