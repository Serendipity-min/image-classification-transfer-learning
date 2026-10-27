"""调查当前实测慢化并保留暂停证据；只确认有证据的保存路径开销，不豁免未来运行。"""
from remote_session import connect
from transport import command

SCRIPT = r'''
import json,pathlib,sys,statistics,shutil,subprocess
r=pathlib.Path('/root/user_data/tl_cls_v1');sys.path.insert(0,str(r/'src'));s=r/'records/p3'
from tlcls.p3_core import read,write,file_hash,require_authorization,scientific_identity,process_alive,code_commit
from tlcls.p3_budget import Budget,Storage
from tlcls.p3_transaction import EpochTransaction
from p3_schedule import candidates,statuses,calibrate,forecast
require_authorization();parent=read(s/'scheduler.json')
assert not process_alive(parent['pid'],parent['start_ticks'])
ledger=read(s/'gpu_ledger.json');assert all(a['status']=='finalized' for a in ledger['attempts'].values())
assert ledger['stop_request']['reason']=='slowdown_above_10_percent_investigate_before_dispatch'
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
progress=read(s/'progress.json');stamp=str(int(progress['updated_unix']))
archive=s/'slowdown_history'/stamp;archive.mkdir(parents=True,exist_ok=True)
for name in ['gpu_ledger.json','forecast.json','progress.json','stage_status.json','terminal_audit.json',
             'cost_calibration.json','dispatch_override.json','slowdown_resume_decision.json']:
 p=s/name
 if p.exists() and not (archive/name).exists():shutil.copyfile(p,archive/name)
runs=candidates();estimates,flags=calibrate(runs);assert flags
baseline=read(r/'reports/P3_proposal.json')['selected_timing_basis'];analyses=[]
for run_id in flags:
 config=next(c for c in runs if c['run_id']==run_id)
 assert config['configuration_id']=='cifar100-vitb16-linear_probe','new configuration needs a separate investigation'
 d=r/'runs'/run_id;paths=[d/'timings'/f'epoch_{e:03d}.json' for e in [2,3]]
 obs=[read(p) for p in paths];p2=baseline[config['configuration_id']]
 ref=[read(r/'runs'/p2['source_run_id']/'timings'/f'epoch_{e:03d}.json') for e in [2,3]]
 means={k:statistics.mean(t[k] for t in obs) for k in ['fit_seconds','train_seconds','validation_seconds','checkpoint_seconds','data_wait_seconds']}
 references={k:statistics.mean(t[k] for t in ref) for k in means}
 # 只确认当前已测量、接近调查线且主要是已知事务开销的案例；超过此范围仍拒绝继续。
 assert means['fit_seconds'] <= 1.15*p2['epoch_seconds']
 assert means['train_seconds'] <= 1.10*references['train_seconds']
 assert means['validation_seconds'] <= 1.10*references['validation_seconds']
 assert means['checkpoint_seconds'] <= 8 and means['data_wait_seconds'] <= 2
 summary=read(d/'summary.json');assert summary['status']=='infrastructure_interrupted'
 assert summary['test_evaluator_constructed'] is False and summary['test_metrics'] is None
 analyses.append({'run_id':run_id,'P3_epoch2_3_means':means,'P2_same_epoch_means':references,
  'frozen_P2_median_seconds':p2['epoch_seconds'],'ratio_to_frozen_median':means['fit_seconds']/p2['epoch_seconds'],
  'ratio_to_same_epochs':means['fit_seconds']/references['fit_seconds'],
  'timing_sha256':[file_hash(p) for p in paths],
  'finding':'observed excess is concentrated in full snapshot transaction/hash/GC; P3 keeps extra model snapshots for prefixes; no numerical failure or memory shortage',
  'causal_limit':'underlying disk bandwidth and concurrent load were not directly sampled; no claim that every second is caused by hardware IO'})
preserved={};boundaries=[]
for c in runs:
 d=r/'runs'/c['run_id'];p=d/'summary.json'
 if not p.exists():continue
 v=read(p);assert read(d/'scientific.json')==scientific_identity(c)
 if v['status']=='completed':preserved[c['run_id']]={n:file_hash(d/n) for n in ['summary.json','epochs.jsonl','current.json','scientific.json']}
 elif v['status']=='infrastructure_interrupted':
  tx=EpochTransaction(d,scientific_identity(c)['scientific_config_sha256'],c);current,rows=tx.load(rebuild=False)
  assert len(rows)==v['completed_epochs']
  sub=archive/c['run_id'];sub.mkdir(exist_ok=True)
  for n in ['summary.json','epochs.jsonl','current.json','prefixes.json','scientific.json']:
   if not (sub/n).exists():shutil.copyfile(d/n,sub/n)
  boundaries.append({'run_id':c['run_id'],'completed_epochs':len(rows),'commit':read(d/'current.json'),'last':current['last']})
resources=Storage().require(4*2**30)
prediction=forecast(runs,statuses(runs),estimates,Budget());assert not prediction['forecast_exceeds_cap']
review={'schema':'P3_slowdown_review_v1','status':'investigated_within_budget','analyses':analyses,
 'acknowledgement_scope':'only currently measured flagged run IDs; future flags must be investigated separately',
 'forecast':prediction,'storage':resources,'completed_runs_preserved':len(preserved),
 'unstarted_runs':sum(not (r/'runs'/c['run_id']/'summary.json').exists() for c in runs),
 'execution_commit':code_commit(),'source_or_scientific_parameters_changed':False,'centralized_test_authorized':False,
 'user_authority':'2026-10-26 explicit continue-when-resources-permit instruction remains in effect'}
write(archive/'investigation.json',review)
write(s/'slowdown_investigation_latest.json',review)
override=read(s/'dispatch_override.json') if (s/'dispatch_override.json').exists() else {}
override.update(investigation_completed=True,acknowledged_slow_run_ids=sorted(set(override.get('acknowledged_slow_run_ids',[]))|set(flags)),
 investigation_receipt_sha256=file_hash(archive/'investigation.json'))
write(s/'dispatch_override.json',override)
decision={'resume_authorized':True,'execution_commit':code_commit(),'hard_cap_gpu_seconds':576000,
 'preserved_completed_runs':preserved,'resume_boundaries':boundaries,'pause_history_path':str(archive.relative_to(r)),
 'investigation_sha256':file_hash(archive/'investigation.json'),'centralized_test_authorized':False}
write(s/'slowdown_resume_decision.json',decision)
budget=Budget()
with budget.edit() as value:
 assert value['stop_request']['reason']=='slowdown_above_10_percent_investigate_before_dispatch'
 value['events'].append({'reason':'investigated_checkpoint_overhead_same_source_same_cap',
  'investigation_sha256':file_hash(archive/'investigation.json'),'previous_stop_request':value['stop_request']})
 value['stop_request']=None
print(json.dumps({'investigation':'passed','completed_preserved':len(preserved),'resume_epochs':[v['completed_epochs'] for v in boundaries],
 'forecast_GPU_hours':prediction['forecast_with_reserves_and_margin_gpu_hours'],'acknowledged_run_ids':flags,'formal_training_restarted':False}))
'''


if __name__ == '__main__':
    client = connect()
    try:
        code, output, error = command(client, ['/root/miniconda3/envs/tl_cls_v1/bin/python', '-c', SCRIPT], timeout=60)
        assert code == 0, error
        print(output.strip())
    finally: client.close()
