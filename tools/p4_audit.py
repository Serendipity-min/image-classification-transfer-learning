"""终态 CPU 复核和小型研究表；不重跑模型、不修改历史。"""
from collections import defaultdict
import csv
import json
import time
import numpy as np
from p4_core import ROOT,STAGE,require_scope,jobs,configuration,budget,read,write,file_hash
from p4_statistics import primary_noninferiority,calibration


def csv_write(name,rows):
    path=ROOT/'reports/P4'/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def B_report():
    selected=read(ROOT/'selected_hparams.json')['configurations'];groups=defaultdict(list)
    for job in jobs():
        if job['stage']!='P4B-sensitivity':continue
        config=configuration(job);key=config['configuration_id'];summary=read(ROOT/'runs'/job['run_id']/'summary.json')
        score=summary['selection_correct_sum']/(3*summary['selection_validation_n'])*100 if summary['status']=='completed' else None
        groups[key].append({'configuration_id':key,'run_id':job['run_id'],'source':'P4B_new','learning_rate':job['learning_rate'],
                            'status':summary['status'],'score_percent':score})
    summaries={}
    for key,rows in groups.items():
        for old in selected[key]['candidates']:
            summary=read(ROOT/'runs'/old['run_id']/'summary.json')
            epochs=[json.loads(s) for s in (ROOT/'runs'/old['run_id']/'epochs.jsonl').read_text().splitlines()]
            eligible=summary['status']=='completed' and len(epochs)==15
            score=sum(e['validation']['correct'] for e in epochs[-3:])/(3*epochs[-1]['validation']['samples'])*100 if eligible else None
            rows.append({'configuration_id':key,'run_id':old['run_id'],'source':'P2_original','learning_rate':old['learning_rate'],'status':summary['status'],'score_percent':score})
        valid=[r for r in rows if r['score_percent'] is not None];winner=min(valid,key=lambda r:(-r['score_percent'],r['learning_rate']))
        original=min([r for r in valid if r['source']=='P2_original'],key=lambda r:(-r['score_percent'],r['learning_rate']))
        outer=min(r['learning_rate'] for r in rows) if key.startswith('cifar100') else max(r['learning_rate'] for r in rows)
        summaries[key]={'S_old':original['score_percent'],'S_ext':winner['score_percent'],'winner_lr':winner['learning_rate'],
                        'improvement_pp':winner['score_percent']-original['score_percent'],'winner_at_external_boundary':winner['learning_rate']==outer,
                        'valid_new_candidates':sum(r['source']=='P4B_new' and r['score_percent'] is not None for r in rows)}
    contrasts=[]
    for group,method in [('cifar100-vitb16','linear_probe'),('svhn-r50','adapter')]:
        a=summaries[group+'-'+method];b=summaries[group+'-full_ft'];old=a['S_old']-b['S_old'];extended=a['S_ext']-b['S_ext']
        contrasts.append({'comparison':group+'-'+method+'-minus-full_ft','g_old_pp':old,'g_ext_pp':extended,'difference_change_pp':extended-old,
            'sensitive_action_threshold':bool(old*extended<0 or max(a['improvement_pp'],b['improvement_pp'])>=.2 or abs(extended-old)>=.2),
            'boundary_unresolved':a['winner_at_external_boundary'] or b['winner_at_external_boundary'] or min(a['valid_new_candidates'],b['valid_new_candidates'])<2})
    csv_write('B_five_point_LR.csv',[r for rows in groups.values() for r in sorted(rows,key=lambda r:r['learning_rate'])])
    write(STAGE/'B_sensitivity.json',{'status':'completed','configurations':summaries,'contrasts':contrasts,'old_test_predictions_for_new_B_models':0,
          'scope':'seed123, 15 epochs, finite five-point union only; does not replace P3 test comparisons'})


def main():
    _,_,identity=require_scope();B_report();snap=budget().snapshot();assert all(a['status']=='finalized' for a in snap['attempts'].values())
    assert budget().used(snap)<=72000
    rows=[];curves=[]
    for job in jobs():
        summary=read(ROOT/'runs'/job['run_id']/'summary.json');assert summary['status'] in ['completed','numerical_failure','oom']
        if summary['status']=='completed':
            epochs=[json.loads(s) for s in (ROOT/'runs'/job['run_id']/'epochs.jsonl').read_text().splitlines()]
            assert len(epochs)==job['epochs'] and [r['epoch'] for r in epochs]==list(range(1,job['epochs']+1))
            if job['dataset']=='flowers102':
                assert all(r['train']['samples']==1020 and r['train']['optimizer_updates']==8 for r in epochs)
                if job['formal_run']:assert summary['optimizer_updates']==400
            curves.extend({'run_id':job['run_id'],'stage':job['stage'],'epoch':r['epoch'],'validation_correct':r['validation']['correct'],
                           'validation_n':r['validation']['samples'],'validation_top1':r['validation']['top1_percent'],'train_loss':r['train']['loss']} for r in epochs)
    from p3c_analyze import independent_metrics
    manifest=read(STAGE/'C_evaluation_manifest.json');calibrations=[];reliability=[];metrics={}
    for item in manifest['evaluations']:
        result=read(STAGE/'C_results'/item['evaluation_id']/'completed.json');assert result['item']==item and not result['test_used_for_selection']
        path=ROOT/result['data_path'];assert file_hash(path)==result['data_sha256']
        with np.load(path,allow_pickle=False) as data:
            x=data['logits'];y=data['labels'];assert len(set(data['sample_id']))==6149 and np.array_equal(data['prediction'],x.argmax(1))
            actual=independent_metrics(x,y)
            for key,value in actual.items():
                expected=result['metrics'][key]
                if isinstance(value,float):assert abs(value-expected)<1e-10
                else:assert value==expected
            if item['role']=='primary_best_val':
                for bins in [15,10,30]:
                    c=calibration(x,y,bins);base={'run_id':item['run_id'],'configuration_id':item['configuration_id'],'seed':item['seed'],'bins':bins}
                    calibrations.append(dict(base,**{k:v for k,v in c.items() if k!='reliability'}))
                    reliability.extend(dict(base,**b) for b in c['reliability'])
        rows.append(dict(item,**{k:v for k,v in actual.items() if k!='confusion_matrix'},predictions_sha256=result['data_sha256']))
        if item['role']=='primary_best_val':metrics[(item['configuration_id'],item['seed'])]=actual['top1_percent']
    assert len(rows)==80 and len(metrics)==40
    differences=[metrics[('flowers102-vitb16-adapter',s)]-metrics[('flowers102-vitb16-full_ft',s)] for s in [17,42,2026,3407,65537]]
    primary=primary_noninferiority(differences);primary['paired_seed_differences_pp']=differences
    model_files=list((STAGE/'A3_models').glob('*.json'));assert len(model_files)==16
    grids=0
    for p in model_files:
        d=read(p);assert d['status']=='completed' and d['BN_buffers_unchanged'] and d['images']==20;grids+=len(d['grids'])
    assert grids==320 and read(STAGE/'A1_A2_acceptance.json')['status']=='completed'
    csv_write('C_official_test_metrics.csv',rows);csv_write('training_epoch_curves.csv',curves)
    csv_write('C_A2_calibration_by_seed.csv',calibrations);csv_write('C_A2_reliability_bins.csv',reliability)
    write(STAGE/'C_primary_noninferiority.json',primary)
    acceptance={'schema':'P4_terminal_audit_v1','status':'passed_for_scientific_consolidated_review','training_diagnostic_runs':84,
          'new_C_test_roles':80,'new_C_prediction_files_independently_recomputed':80,'historical_best_calibration_models':80,
          'new_best_calibration_models':40,'A3_maps':320,'A3_required_image_forwards':63040,'P4_GPU_hours':budget().used(snap)/3600,
          'all_attempts_settled':True,'historical_files_still_match_frozen_inputs':True,'new_B_models_old_test_predictions':0,
          'P4_D_started':False,'new_tests_or_training_outside_registered_matrix':False,'audited_unix':time.time()}
    write(STAGE/'delivery_audit.json',acceptance)
    write(STAGE/'P4_sealed.json',{'status':'sealed_waiting_consolidated_review','protocol_sha256':file_hash(STAGE/'protocol_frozen.json'),
          'gpu_ledger_sha256':file_hash(STAGE/'gpu_ledger.json'),'delivery_audit_sha256':file_hash(STAGE/'delivery_audit.json'),
          'C_primary_sha256':file_hash(STAGE/'C_primary_noninferiority.json'),'scientific_input_identity':identity})
    print('P4 registered execution audited; no paper rewritten before consolidated review.')


if __name__=='__main__':main()
