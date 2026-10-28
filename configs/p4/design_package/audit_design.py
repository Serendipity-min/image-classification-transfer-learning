"""Static protocol validation only. No model training, inference or network access."""
from pathlib import Path
from collections import Counter, defaultdict
import json

ROOT = Path(__file__).resolve().parent

def main() -> None:
    protocol = json.loads((ROOT / '02_P4_protocol.json').read_text(encoding='utf-8'))
    matrix = json.loads((ROOT / '03_P4_run_matrix.json').read_text(encoding='utf-8'))
    rows = matrix['training_jobs']
    evaluations = matrix['new_dataset_evaluations']
    passed = []
    def check(name: str, condition: bool) -> None:
        if not condition:
            raise AssertionError(name)
        passed.append(name)
    expected = {'P4B-anchor':4, 'P4B-sensitivity':8, 'P4C-profile':8, 'P4C-tuning':24, 'P4C-formal':40}
    check('protocol identifiers agree', matrix['protocol_id'] == protocol['protocol_id'])
    check('no execution authorization is implied', protocol['execution_authorized'] is False)
    check('run counts by stage', dict(Counter(r['stage'] for r in rows)) == expected)
    check('84 unique planned training/diagnostic job IDs', len(rows) == len({r['run_id'] for r in rows}) == 84)
    check('all jobs remain planned', all(r['status'] == 'planned_NOT_AUTHORIZED' for r in rows))
    check('no training process accesses test', all(r['test_during_training'] is False for r in rows))
    check('scientific scheduler and batch remain fixed', all((r['scheduler_horizon_epochs'], r['warmup_epochs'],r['micro_batch_size'],r['gradient_accumulation_steps'],r['autocast_dtype']) == (50,5,32,4,'bfloat16') for r in rows))
    check('scratch never loads pretrained weights', all(r['pretrained_loaded'] == (r['method'] != 'scratch') for r in rows))
    check('FT head LR remains tenfold', all(r['head_lr_multiplier'] == (10 if r['method']=='full_ft' else 1) for r in rows))
    check('no P2/P3 model continuation', all(r['fresh_initialization'] and not r['resume_from_P2_or_P3'] for r in rows))
    b = [r for r in rows if r['stage'].startswith('P4B')]
    check('B is seed123 and blocks old-test predictions', all(r['seed']==123 and r['new_test_scope']=='NO_NEW_OLD_TEST_PREDICTIONS' for r in b))
    check('B anchors3 and candidates15 epochs', all(r['epochs'] == (3 if r['stage']=='P4B-anchor' else 15) for r in b))
    bgroup=Counter((r['dataset'],r['model_key'],r['method']) for r in b if r['stage']=='P4B-sensitivity')
    check('balanced two additional LR points per B group', len(bgroup)==4 and set(bgroup.values())=={2})
    c = [r for r in rows if r['stage'].startswith('P4C')]
    check('only one new dataset', {r['dataset'] for r in c}=={'flowers102'})
    check('official Flowers split numbers', (protocol['C']['train_n'],protocol['C']['val_n'],protocol['C']['test_n']) == (1020,1020,6149))
    check('development labels account for validation', protocol['C']['supervised_development_labels']==2040)
    check('Flowers update and tail arithmetic', (1020+127)//128==8 and 1020%128==124 and 50*8==400)
    tunegroup=defaultdict(list)
    formalgroup=defaultdict(list)
    for r in c:
        if r['stage']=='P4C-tuning':tunegroup[(r['model_key'],r['method'])].append(r)
        if r['stage']=='P4C-formal':formalgroup[(r['model_key'],r['method'])].append(r)
    check('eight C groups each three LR candidates', len(tunegroup)==8 and all(len(v)==3 for v in tunegroup.values()))
    check('C candidate grids match protocol', all(sorted(x['learning_rate'] for x in v)==sorted(protocol['C']['training_grids'][k[1]]) for k,v in tunegroup.items()))
    check('C formal seeds complete and fixed', len(formalgroup)==8 and all(sorted(x['seed'] for x in v)==protocol['C']['formal_seeds'] for v in formalgroup.values()))
    check('C formal rates intentionally unresolved, not invented', all(r['learning_rate'] is None and r['selected_lr_ref'] and r['epochs']==50 for v in formalgroup.values() for r in v))
    check('80 evaluation roles remain locked', len(evaluations)==80 and all(r['status']=='LOCKED_PENDING_TRAINING_AND_SEAL' for r in evaluations))
    check('40 best and 40 last, no new prefixes', dict(Counter(e['role'] for e in evaluations)) == {'primary_best_val':40,'secondary_last':40} and not protocol['C']['new_time_prefixes'])
    check('no fabricated evaluation metric or checkpoint', all(e['metric'] is None and e['checkpoint_sha256'] is None and e['evaluation_id'] is None for e in evaluations))
    ids={r['run_id'] for r in rows if r['formal_run']}
    check('every planned evaluation references a registered formal job', {e['run_id'] for e in evaluations}==ids)
    check('occlusion arithmetic', protocol['A']['occlusion']['total_image_forwards'] == 40*8*(14*14+1) == 63040 and protocol['A']['occlusion']['models']==16)
    check('A estimands not pooled across seed-image observations', protocol['A']['bootstrap']['prohibited']=='treat_5_times_N_predictions_as_independent')
    check('one new C primary family', protocol['C']['primary_question']['primary_family_size']==1 and protocol['C']['primary_question']['margin_pp']==-.5)
    budget=protocol['budget']
    check('budget allocation sums to20', abs(sum(budget[k] for k in ['A3_plan','B_plan','C_plan','shared_engineering_reserve'])-budget['total_cap'])<1e-12 and budget['total_cap']==20)
    check('paper rewriting deferred until consolidated review', 'after P4-A, P4-B and P4-C' in protocol['rewrite_when'])
    result={'status':'PASS_STATIC_DESIGN_ONLY','checks_passed':len(passed),'checks':passed,'training_and_diagnostic_job_count':len(rows),'formal_new_training_count':40,'planned_new_test_role_count':80,'new_GPU_hours_spent':0,'server_execution_verified':False,'note':'This audits design consistency only. It is not CPU/GPU execution acceptance of a training implementation.'}
    (ROOT/'05_设计一致性校验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
