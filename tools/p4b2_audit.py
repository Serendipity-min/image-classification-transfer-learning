"""B2独立终态复算与只读封存；不导入torch、不访问官方test。"""
import csv
import json
import math
from pathlib import Path
import time
from p4b2 import ROOT,STAGE,require_scope,manifest,read,write,file_hash,value_hash,readonly_ledger,process_alive


def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def main():
    _,m,identity=require_scope();launch=read(STAGE/'launch.json')
    assert not process_alive(launch['pid'],launch['start_ticks'])
    before=file_hash(STAGE/'gpu_ledger.json');ledger=readonly_ledger()
    assert all(a['status']=='finalized' for a in ledger['attempts'].values())
    used=sum(a['charged_gpu_seconds'] for a in ledger['attempts'].values());assert used<=1800
    for a in ledger['attempts'].values():
        assert a['run_id'] in {j['run_id'] for j in m['runs']}
        assert a['charged_gpu_seconds']==a['ended_monotonic']-a['started_monotonic']
        assert a['gpu_uuid']==read(STAGE/'preflight.json')['GPU_inventory'][str(a['device'])]['uuid']
    for device in [0,1]:
        attempts=sorted([a for a in ledger['attempts'].values() if a['device']==device],key=lambda a:a['started_monotonic'])
        assert all(a['ended_monotonic']<=b['started_monotonic'] for a,b in zip(attempts,attempts[1:]))
    for item in read(STAGE/'historical_checkpoint_stat_index.json')['files']:
        stat=(ROOT/item['path']).stat();assert (stat.st_size,stat.st_mtime_ns,stat.st_ino)==(item['bytes'],item['mtime_ns'],item['inode'])
    old=read(ROOT/'records/p4/C_selected_hparams.json')['configurations'];table=[];curves=[];summaries={};artifacts=[]
    for item in old['flowers102-r50-full_ft']['candidates']:
        directory=ROOT/'runs'/item['run_id'];rows=[json.loads(s) for s in (directory/'epochs.jsonl').read_text().splitlines()]
        total=sum(r['validation']['correct'] for r in rows[-3:]);assert len(rows)==15 and total==item['selection_correct_sum']
        table.append({'run_id':item['run_id'],'source':'original_P4C1','backbone_LR':item['learning_rate'],'head_LR':item['learning_rate']*10,
            'status':'completed','completed_epochs':15,'correct13':rows[12]['validation']['correct'],'correct14':rows[13]['validation']['correct'],
            'correct15':rows[14]['validation']['correct'],'correct_sum':total,'denominator':3060,'score_percent':100*total/3060,'GPU_hours_B2':None})
    for job in m['runs']:
        directory=ROOT/'runs'/job['run_id'];summary=read(directory/'summary.json');summaries[job['run_id']]=summary
        assert summary['status'] in ['completed','numerical_failure','oom','infrastructure_interrupted']
        assert summary['test_evaluator_constructed'] is False
        scientific=read(directory/'scientific.json');assert scientific['configuration']==job
        digest=value_hash(scientific);assert digest==summary['scientific_config_sha256']
        rows=[json.loads(s) for s in (directory/'epochs.jsonl').read_text().splitlines()] if (directory/'epochs.jsonl').exists() else []
        pointer=read(directory/'current.json') if (directory/'current.json').exists() else None;commits=[]
        while pointer:
            p=directory/pointer['path'];assert file_hash(p)==pointer['sha256'];value=read(p);assert value['scientific_config_sha256']==digest
            commits.append(value);pointer=value['previous_commit']
        assert [c['epoch_row'] for c in reversed(commits)]==rows
        assert [r['epoch'] for r in rows]==list(range(1,len(rows)+1))
        for row in rows:
            assert row['train']['samples']==1020 and row['train']['optimizer_updates']==8 and row['validation']['samples']==1020
            assert row['cumulative_optimizer_updates']==row['epoch']*8
            assert math.isfinite(row['train']['loss']) and math.isfinite(row['validation']['loss'])
            curves.append({'run_id':job['run_id'],'backbone_LR':job['learning_rate'],'head_peak_LR':job['learning_rate']*10,
                'epoch':row['epoch'],'train_loss':row['train']['loss'],'train_top1':row['train']['top1_percent'],
                'validation_correct':row['validation']['correct'],'validation_n':1020,'validation_top1':row['validation']['top1_percent'],
                'optimizer_updates_cumulative':row['cumulative_optimizer_updates'],'sampler_order_sha256':row['sampler_order_sha256'],
                'grad_norm_before_clip_last_window':row['train']['last_update']['grad_norm_before_clip'],
                'physical_gpu_index':row['physical_gpu_index'],'attempt_id':row['attempt_id']})
        complete=summary['status']=='completed'
        if complete:
            assert len(rows)==15 and summary['optimizer_updates']==120
            assert summary['model_identity']['initial_classifier_sha256']==m['head_initial_sha256']
            groups=read(directory/'parameter_groups.json')['groups']
            for group in groups:assert group['peak_lr']==job['learning_rate']*(10 if group['name'].startswith('fc.') else 1)
        score=sum(r['validation']['correct'] for r in rows[-3:]) if complete else None
        if complete:assert score==summary['selection_correct_sum']
        own=[a for a in ledger['attempts'].values() if a['run_id']==job['run_id']]
        table.append({'run_id':job['run_id'],'source':'new_P4B2','backbone_LR':job['learning_rate'],'head_LR':job['learning_rate']*10,
            'status':summary['status'],'completed_epochs':len(rows),'correct13':rows[12]['validation']['correct'] if len(rows)>=13 else None,
            'correct14':rows[13]['validation']['correct'] if len(rows)>=14 else None,'correct15':rows[14]['validation']['correct'] if len(rows)>=15 else None,
            'correct_sum':score,'denominator':3060,'score_percent':100*score/3060 if score is not None else None,
            'GPU_hours_B2':sum(a['charged_gpu_seconds'] for a in own)/3600})
        for key in ['best_checkpoint','last_checkpoint','failure_state']:
            if key in summary:
                d=summary[key];p=directory/d['path'];assert p.stat().st_size==d['size_bytes'] and file_hash(p)==d['sha256']
                artifacts.append({'run_id':job['run_id'],'role':key,'path':p.relative_to(ROOT).as_posix(),**{k:v for k,v in d.items() if k!='path'}})
        assert not list(directory.rglob('*.npz'))
    table.sort(key=lambda r:r['backbone_LR']);valid=[r for r in table if r['correct_sum'] is not None]
    best=min(valid,key=lambda r:(-r['correct_sum'],r['backbone_LR']));oldbest=max(r['correct_sum'] for r in table if r['source']=='original_P4C1')
    adapter=old['flowers102-r50-adapter']['selection_correct_sum'];assert adapter==2777 and oldbest==2502
    result={'S_old_percent':100*oldbest/3060,'S_ext_percent':best['score_percent'],'delta_FT_early_pp':100*(best['correct_sum']-oldbest)/3060,
        'selected_LR_in_five_point_diagnostic':best['backbone_LR'],'winner_at_high_external_boundary':best['backbone_LR']==.003,
        'old_Adapter_reference_correct_sum':adapter,'old_Adapter_reference_score_percent':100*adapter/3060,
        'Adapter_minus_original_FT_validation_pp':100*(adapter-oldbest)/3060,
        'Adapter_minus_expanded_FT_validation_pp':100*(adapter-best['correct_sum'])/3060,
        'new_candidate_wins':best['source']=='new_P4B2','early_LR_sensitivity_action_threshold_pp':.2,
        'early_LR_sensitive':100*(best['correct_sum']-oldbest)/3060>=.2,
        'unsearched_LR':.0003,'official_test_results_changed':False}
    output=ROOT/'reports/P4B2';output.mkdir(parents=True,exist_ok=True)
    csv_write(output/'P4B2_LR_validation.csv',table);csv_write(output/'P4B2_epoch_curves.csv',curves)
    csv_write(output/'P4B2_GPU_attempts.csv',list(ledger['attempts'].values()))
    write(STAGE/'comparison.json',result);write(STAGE/'checkpoint_inventory.json',{'files':artifacts,'raw_checkpoints_remain_on_server':True})
    for _ in range(3):readonly_ledger()
    assert file_hash(STAGE/'gpu_ledger.json')==before
    acceptance={'schema':'P4B2_independent_acceptance_v1','status':'passed' if all(v['status'] in ['completed','numerical_failure','oom'] for v in summaries.values()) else 'incomplete',
        'runs':{k:{'status':v['status'],'completed_epochs':v['completed_epochs'],'scientific_config_sha256':v['scientific_config_sha256']} for k,v in summaries.items()},
        'GPU_hours':used/3600,'GPU_cap_hours':.5,'warning_hours':.15,'warning_reached':used>=540,
        'all_attempts_settled':True,'old_844_checkpoint_stats_unchanged':True,'old_source_seals_scores_ledgers_unchanged':True,
        'only_configuration_changes':'backbone LR and matching10x head LR; run/stage/test-scope identifiers only',
        'official_test_constructed':False,'new_test_predictions':0,'new50epoch_formal_runs':0,'new_seeds':0,
        'original_test_scores_not_reestimated':True,'head_initial_sha256':m['head_initial_sha256'],'execution_identity_sha256':value_hash(identity),
        'source_auditor_sha256':file_hash(Path(__file__)),'terminal_ledger_pure_reads_preserve_bytes':True,
        'paper_phase_authorized':False,'audited_unix':time.time()}
    write(STAGE/'acceptance.json',acceptance)
    sealed={'schema':'P4B2_terminal_seal_v1','status':acceptance['status'],'audit_baseline':'3faa771c959e04b27c72e15c97d4895ed82b3cef',
        'execution_identity_sha256':file_hash(STAGE/'execution_identity.json'),'manifest_sha256':file_hash(STAGE/'run_manifest.json'),
        'CPU_acceptance_sha256':file_hash(STAGE/'CPU_acceptance.json'),'GPU_ledger_sha256':before,'acceptance_sha256':file_hash(STAGE/'acceptance.json'),
        'comparison_sha256':file_hash(STAGE/'comparison.json'),'validation_table_sha256':file_hash(output/'P4B2_LR_validation.csv'),
        'epoch_curves_sha256':file_hash(output/'P4B2_epoch_curves.csv'),'checkpoint_inventory_sha256':file_hash(STAGE/'checkpoint_inventory.json'),
        'summary_sha256':{j['run_id']:file_hash(ROOT/'runs'/j['run_id']/'summary.json') for j in m['runs']},
        'epoch_logs_sha256':{j['run_id']:file_hash(ROOT/'runs'/j['run_id']/'epochs.jsonl') for j in m['runs']},
        'no_mutating_budget_snapshot_after_seal':True,'GPU_hours':used/3600,'paper_deferred':True}
    write(STAGE/'P4B2_sealed.json',sealed)
    assert file_hash(STAGE/'gpu_ledger.json')==before
    print(json.dumps({'acceptance':acceptance['status'],'GPU_hours':used/3600,'comparison':result},ensure_ascii=False))


if __name__=='__main__':main()
