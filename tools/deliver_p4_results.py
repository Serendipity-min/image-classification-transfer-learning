"""仅从已核验文本快照独立复算P4交付，不读取模型或原始预测。"""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
from scipy import stats

ROOT=Path(__file__).resolve().parents[1]
SEEDS=[17,42,2026,3407,65537]


def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def value_sha(v):return hashlib.sha256(json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8',newline='\n')
def load(p):
    with p.open(encoding='utf8') as file:return list(csv.DictReader(file))
def save(p,rows):
    with p.open('w',encoding='utf8',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def paired(values):
    mean=statistics.mean(values);sd=statistics.stdev(values);half=float(stats.t.ppf(.975,4))*sd/math.sqrt(5)
    return {'mean_difference_pp':mean,'sample_SD_pp':sd,'CI95_low_pp':mean-half,'CI95_high_pp':mean+half}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,required=True);args=parser.parse_args();snap=args.snapshot.resolve()
    manifest=read(snap/'snapshot_manifest.json')
    for name,d in manifest['files'].items():
        p=snap/name;assert p.resolve().is_relative_to(snap) and sha(p)==d['sha256'] and p.stat().st_size==d['size_bytes']
    stage=snap/'records/p4';frozen=read(stage/'protocol_frozen.json');identity=read(stage/'execution_identity.json')
    assert sha(stage/'protocol_frozen.json')==identity['protocol_frozen_sha256']
    for name,digest in identity['files'].items():assert sha(ROOT/name)==digest
    for name,digest in identity['historical_read_only_files'].items():assert sha(ROOT/name)==digest
    ledger=read(stage/'gpu_ledger.json');attempts=list(ledger['attempts'].values());assert len(attempts)==189 and all(a['status']=='finalized' for a in attempts)
    for a in attempts:assert a['charged_gpu_seconds']==a['ended_monotonic']-a['started_monotonic']
    for device in [0,1]:
        sequence=sorted([a for a in attempts if a['device']==device],key=lambda a:a['started_monotonic'])
        assert all(a['ended_monotonic']<=b['started_monotonic'] for a,b in zip(sequence,sequence[1:]))
    cost=sum(a['charged_gpu_seconds'] for a in attempts)/3600;assert cost<=20
    audit=read(stage/'delivery_audit.json');assert abs(cost-audit['P4_GPU_hours'])<1e-12
    bad=[a for a in attempts if a['returncode']!=0]
    assert len(bad)==1 and bad[0]['kind']=='P4B-anchor' and bad[0]['returncode']==-9 and bad[0]['termination_reason']=='P4_technical_cost_forecast_revision'
    jobs=frozen['design_matrix']['training_jobs'];runs=[];configuration_rows={};epochs_total=0;actual_commits=Counter()
    for job in jobs:
        directory=snap/'runs'/job['run_id'];summary=read(directory/'summary.json');scientific=read(directory/'scientific.json');digest=value_sha(scientific)
        assert summary['status']=='completed' and summary['completed_epochs']==job['epochs'] and summary['scientific_config_sha256']==digest
        reference=read(directory/'current.json');chain=[]
        while reference:
            p=directory/reference['path'];assert sha(p)==reference['sha256'];commit=read(p);assert commit['scientific_config_sha256']==digest
            chain.append(commit);reference=commit['previous_commit']
        chain.reverse();rows=[c['epoch_row'] for c in chain]
        assert [r['epoch'] for r in rows]==list(range(1,job['epochs']+1));epochs_total+=len(rows)
        assert [json.loads(s) for s in (directory/'epochs.jsonl').read_text().splitlines()]==rows
        assert chain[-1]['last']==summary['last_checkpoint'] and chain[-1]['best']==summary['best_checkpoint']
        config=scientific['configuration'];configuration_rows[job['run_id']]=config
        assert config['seed']==job['seed'] and config['scheduler_horizon_epochs']==50 and config['warmup_epochs']==5
        if job['dataset']=='flowers102':
            assert all(r['train']['samples']==1020 and r['train']['optimizer_updates']==8 and r['validation']['samples']==1020 for r in rows)
            if job['formal_run']:assert summary['optimizer_updates']==400
        if job['epochs']==15:assert summary['selection_correct_sum']==sum(r['validation']['correct'] for r in rows[-3:])
        if job['stage']=='P4B-anchor':
            source=read(ROOT/'selected_hparams.json')['configurations'][config['configuration_id']]['selected_from_run_id']
            old=[json.loads(s) for s in (ROOT/'runs'/source/'epochs.jsonl').read_text().splitlines()][:3]
            for a,b in zip(old,rows):
                assert all(a[k]==b[k] for k in ['sampler_order_sha256','cumulative_optimizer_updates','lr_by_group_next_update'])
                assert all(a['validation'][k]==b['validation'][k] for k in ['samples','correct','top1_percent','loss'])
                assert all(a['train'][k]==b['train'][k] for k in ['samples','optimizer_updates','top1_percent','loss'])
        own=[a for a in attempts if a['run_id']==job['run_id']]
        parameters=summary['parameters'];count=max([a['process_tree_peak_rss_bytes_sampled'] for a in own])
        runs.append({'run_id':job['run_id'],'stage':job['stage'],'configuration_id':config['configuration_id'],'dataset':job['dataset'],
            'model_key':job['model_key'],'method':job['method'],'seed':job['seed'],'learning_rate':config['learning_rate'],'epochs':summary['completed_epochs'],
            'selected_epoch':summary['selected_epoch'],'best_validation_top1':summary['best_val_top1_percent'],
            'GPU_hours':sum(a['charged_gpu_seconds'] for a in own)/3600,'attempts':len(own),
            'peak_allocated_GiB':summary['peak_allocated_bytes']/2**30,'peak_reserved_GiB':summary['peak_reserved_bytes']/2**30,
            'sampled_process_tree_RSS_peak_GiB':count/2**30,'total_params':parameters['total_params'],'trainable_params':parameters['trainable_params'],
            'trainable_ratio':parameters['trainable_ratio'],'source_commit_final_summary':summary['execution_source_commit']})
        actual_commits[summary['execution_source_commit']]+=1
    assert len(runs)==84 and epochs_total==2516
    selected=read(stage/'C_selected_hparams.json')['configurations']
    for key,selection in selected.items():
        candidates=[read(snap/'runs'/j['run_id']/'summary.json') for j in jobs if j['stage']=='P4C-tuning' and configuration_rows[j['run_id']]['configuration_id']==key]
        winner=min(candidates,key=lambda r:(-r['selection_correct_sum'],r['learning_rate']))
        assert winner['run_id']==selection['run_id'] and winner['learning_rate']==selection['learning_rate']
    cseal=read(stage/'C_training_sealed.json');assert cseal['selection_sha256']==sha(stage/'C_selected_hparams.json')
    assert cseal['evaluation_manifest_sha256']==sha(stage/'C_evaluation_manifest.json')
    for run_id,digest in cseal['summary_sha256'].items():assert sha(snap/'runs'/run_id/'summary.json')==digest
    metrics=load(snap/'reports/P4/C_official_test_metrics.csv');by_id={r['evaluation_id']:r for r in metrics}
    assert len(metrics)==len(by_id)==80
    cmap=read(stage/'C_evaluation_manifest.json')['evaluations'];assert len(cmap)==80
    for item in cmap:
        value=read(stage/'C_results'/item['evaluation_id']/'completed.json');row=by_id[item['evaluation_id']]
        assert value['item']==item and value['identity']==value_sha(item) and not value['test_used_for_selection']
        for k in ['run_id','role','checkpoint_sha256']:assert row[k]==item[k]
        metric=value['metrics'];assert metric['n']==6149 and metric['top1_percent']==100*metric['correct']/6149
        confusion=metric['confusion_matrix'];assert sum(map(sum,confusion))==6149 and sum(confusion[i][i] for i in range(102))==metric['correct']
        for k in ['top1_percent','macro_f1','balanced_accuracy','nll']:assert abs(metric[k]-float(row[k]))<1e-12
    primary=[r for r in metrics if r['role']=='primary_best_val'];contrasts=[];pair_rows=[]
    for model in ['r50','vitb16']:
        for method in ['scratch','linear_probe','adapter']:
            group='flowers102-'+model;values=[]
            for seed in SEEDS:
                a=float(next(r['top1_percent'] for r in primary if r['configuration_id']==group+'-'+method and int(r['seed'])==seed))
                b=float(next(r['top1_percent'] for r in primary if r['configuration_id']==group+'-full_ft' and int(r['seed'])==seed))
                values.append(a-b);pair_rows.append({'comparison':group+'-'+method+'-minus-full_ft','seed':seed,'A_top1':a,'FT_top1':b,'difference_pp':a-b})
            contrasts.append({'comparison':group+'-'+method+'-minus-full_ft','role':'primary_noninferiority' if model=='vitb16' and method=='adapter' else 'exploratory',**paired(values)})
    ni=contrasts[-1];server_ni=read(stage/'C_primary_noninferiority.json')
    assert abs(ni['mean_difference_pp']-server_ni['mean_difference_pp'])<1e-12
    assert abs(ni['CI95_low_pp']-server_ni['CI95_pp'][0])<1e-12 and abs(ni['CI95_high_pp']-server_ni['CI95_pp'][1])<1e-12
    assert (ni['CI95_low_pp']>-.5)==server_ni['noninferiority_supported']
    configurations=[]
    for key in sorted({r['configuration_id'] for r in metrics}):
        formal=[r for r in runs if r['stage']=='P4C-formal' and r['configuration_id']==key]
        for role in ['primary_best_val','secondary_last']:
            subset=[r for r in metrics if r['configuration_id']==key and r['role']==role];assert sorted(int(r['seed']) for r in subset)==SEEDS
            row={'configuration_id':key,'role':role,'n_seeds':5,'learning_rate':selected[key]['learning_rate']}
            for metric in ['top1_percent','macro_f1','balanced_accuracy','nll']:
                vals=[float(r[metric]) for r in subset];row[metric+'_mean']=statistics.mean(vals);row[metric+'_SD']=statistics.stdev(vals)
            for name in ['GPU_hours','peak_allocated_GiB','peak_reserved_GiB','sampled_process_tree_RSS_peak_GiB']:
                vals=[r[name] for r in formal];row[name+'_mean']=statistics.mean(vals);row[name+'_SD']=statistics.stdev(vals)
            row.update(trainable_params=formal[0]['trainable_params'],total_params=formal[0]['total_params'],trainable_ratio=formal[0]['trainable_ratio'])
            configurations.append(row)
    btable=load(snap/'reports/P4/B_five_point_LR.csv');bsummary=read(stage/'B_sensitivity.json')
    for key,result in bsummary['configurations'].items():
        candidates=[r for r in btable if r['configuration_id']==key];assert len(candidates)==5
        best=min(candidates,key=lambda r:(-float(r['score_percent']),float(r['learning_rate'])))
        assert float(best['score_percent'])==result['S_ext'] and float(best['learning_rate'])==result['winner_lr']
    artifact=ROOT/'evidence/P4_final_20261028';artifact.mkdir(parents=True,exist_ok=True)
    for name in ['text_evidence.tar.gz'] :shutil.copy2(snap.parent/name,artifact/name)
    shutil.copy2(snap/'snapshot_manifest.json',artifact/'snapshot_manifest.json')
    for name in manifest['files']:
        if name.startswith('reports/P4/') or (name.startswith('records/p4/') and '/C_results/' not in name and '/A3_results/' not in name):
            target=ROOT/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(snap/name,target)
    output=ROOT/'reports/P4'
    save(output/'training_runs.csv',runs);save(output/'C_configuration_summary.csv',configurations)
    save(output/'C_paired_comparisons.csv',contrasts);save(output/'C_paired_seed_differences.csv',pair_rows)
    save(output/'GPU_attempt_ledger.csv',attempts)
    calibrations=load(output/'A2_calibration_by_seed.csv')+load(output/'C_A2_calibration_by_seed.csv');calibration_summaries=[]
    for key in sorted({r['configuration_id'] for r in calibrations}):
        for bins in [10,15,30]:
            subset=[r for r in calibrations if r['configuration_id']==key and int(r['bins'])==bins];assert sorted(int(r['seed']) for r in subset)==SEEDS
            row={'configuration_id':key,'bins':bins,'n_seeds':5}
            for metric in ['ece','brier_no_class_division','nll','top1_percent']:
                vals=[float(r[metric]) for r in subset];row[metric+'_mean']=statistics.mean(vals);row[metric+'_SD']=statistics.stdev(vals)
            calibration_summaries.append(row)
    save(output/'calibration_configuration_summary.csv',calibration_summaries)
    stagecost={key:sum(a['charged_gpu_seconds'] for a in attempts if a['kind']==key)/3600 for key in sorted({a['kind'] for a in attempts})}
    proof={'schema':'P4_local_delivery_v1','status':'PASSED_DATA_AND_RUNTIME_WITH_DISCLOSED_ORIGINAL_SEAL_BINDING_DEFECT',
        'registered_training_diagnostic_runs':84,'committed_epochs':2516,'C_formal_runs':40,'C_test_roles':80,'CPU_analysis_models':120,
        'attempts':189,'nonzero_return_attempts':1,'nonzero_return_cause':'authorized technical forecast correction; exact-config epoch resume',
        'numerical_failure_runs':0,'all_attempts_settled':True,'same_device_reservations_do_not_overlap':True,'historical_input_files_unchanged':True,
        'four_anchor_trajectories_rechecked_against_P2':True,'source_commits_in_final_run_summaries':dict(actual_commits),'C_primary_recomputed':ni,
        'original_operational_seal_raw_ledger_hash_matches':read(stage/'P4_sealed.json')['gpu_ledger_sha256']==sha(stage/'gpu_ledger.json'),
        'original_sealed_ledger_bytes_available':False,'corrected_delivery_seal_required':True,'P4_GPU_hours':cost,'cap_GPU_hours':20,'remaining_GPU_hours_not_authorization':20-cost,
        'GPU_hours_by_kind':stagecost,'text_snapshot_files_verified':len(manifest['files']),'text_archive_sha256':sha(artifact/'text_evidence.tar.gz'),
        'binary_prediction_or_checkpoint_downloaded':False,'C_last_selected_by_text_deviation_disclosed':True}
    write(ROOT/'records/p4/local_delivery_acceptance.json',proof)
    write(artifact/'collection_receipt.json',{'archive_sha256':proof['text_archive_sha256'],'archive_size_bytes':(artifact/'text_evidence.tar.gz').stat().st_size,
        'verified_text_files':len(manifest['files']),'text_bytes':manifest['text_bytes'],'raw_prediction_downloaded':False,'checkpoint_downloaded':False})
    print(json.dumps({'status':proof['status'],'runs':84,'epochs':2516,'C_test':80,'GPU_hours':cost,'NI':ni},ensure_ascii=False))


if __name__=='__main__':main()
