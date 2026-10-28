"""P4 固定阶段后台调度；独立 guardian 结算两卡共享预算。"""
import argparse
import ctypes
import os
import signal
import subprocess
import sys
import time
import uuid
from p4_core import ROOT,STAGE,CAP,EXIT,require_scope,jobs,configuration,budget,storage,read,write,file_hash,value_hash,lock,process_alive,process_ticks,stage_seconds,selection_stage


PYTHON='/root/miniconda3/envs/tl_cls_v1/bin/python'


def parent_guard():ctypes.CDLL(None).prctl(1,signal.SIGKILL)


def kill_group(pid,sig):
    try:os.killpg(pid,sig)
    except ProcessLookupError:pass


def guardian(args):
    require_scope();ledger=budget();process=None;reason='completed'
    ledger.register(args.attempt,args.id,args.device,args.kind,args.owner,args.owner_ticks)
    try:
        env=dict(os.environ,TLCLS_ATTEMPT_ID=args.attempt,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        if args.kind.startswith('P4A3') or args.kind=='P4C-test':
            command=[PYTHON,'tools/p4_infer.py','--mode','A3' if args.kind.startswith('P4A3') else 'C3','--id',args.id,'--device',str(args.device)]
        else:
            command=[PYTHON,'tools/p4_train.py','--run-id',args.id,'--device',str(args.device)]
            if args.resume:command.append('--resume')
        process=subprocess.Popen(command,cwd=ROOT,env=env,start_new_session=True,preexec_fn=parent_guard)
        ledger.update(args.attempt,child_pid=process.pid,child_pgid=process.pid)
        import psutil
        termination=None
        while process.poll() is None:
            snap=ledger.snapshot();now=time.monotonic()
            if not process_alive(args.owner,args.owner_ticks):ledger.stop('P4_owner_lost',10)
            if ledger.used(snap)>=CAP-EXIT:ledger.stop('P4_total_hard_cap',10)
            if args.kind.startswith('P4A3') and stage_seconds(snap,'P4A3')>=1800-EXIT:ledger.stop('P4A3_subcap',10)
            if args.kind=='P4C-profile' and stage_seconds(snap,'P4C-profile')>=3600-EXIT:ledger.stop('P4C0_subcap',10)
            if int(now)%10==0:
                try:storage()
                except RuntimeError:ledger.stop('P4_physical_storage_reserve',10)
            stop=ledger.snapshot()['stop_request']
            if stop and termination is None:
                reason=stop['reason'];termination=now
                try:os.kill(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
            if termination is not None and now-termination>=10:kill_group(process.pid,signal.SIGKILL)
            try:
                parent=psutil.Process(process.pid);rss=parent.memory_info().rss
                tree=rss+sum(p.memory_info().rss for p in parent.children(recursive=True) if p.is_running())
                item=snap['attempts'][args.attempt]
                ledger.update(args.attempt,main_process_peak_rss_bytes_sampled=max(item['main_process_peak_rss_bytes_sampled'],rss),
                    process_tree_peak_rss_bytes_sampled=max(item['process_tree_peak_rss_bytes_sampled'],tree),samples=item['samples']+1)
            except psutil.Error:pass
            time.sleep(.5)
        process.wait();kill_group(process.pid,signal.SIGKILL)
    except BaseException:
        reason='P4_guardian_exception'
        if process is not None:kill_group(process.pid,signal.SIGKILL);process.wait()
        raise
    finally:
        ledger.finalize(args.attempt,process.returncode if process else -1,reason,time.monotonic())
    return process.returncode


def launch_task(identifier,kind,device):
    attempt='p4-'+uuid.uuid4().hex
    command=[PYTHON,'tools/p4_supervise.py','guardian','--id',identifier,'--kind',kind,'--device',str(device),
             '--attempt',attempt,'--owner',str(os.getpid()),'--owner-ticks',process_ticks(os.getpid())]
    summary=ROOT/'runs'/identifier/'scientific.json'
    if summary.exists():command.append('--resume')
    log=STAGE/'logs'/(attempt+'.log');log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('ab') as file:process=subprocess.Popen(command,cwd=ROOT,stdout=file,stderr=subprocess.STDOUT,start_new_session=True)
    return process,attempt


def stage_run(stage,items):
    ledger=budget();pending=list(items);active={};finished=[]
    while pending or active:
        snap=ledger.snapshot()
        if snap['stop_request']:
            for process,_ in active.values():process.wait()
            raise RuntimeError('P4_shared_stop')
        for device,(process,item) in list(active.items()):
            code=process.poll()
            if code is None:continue
            numerical_terminal = code==2 and item['kind'] in ['P4B-sensitivity','P4C-tuning','P4C-formal']
            if code!=0 and not numerical_terminal:
                ledger.stop('P4_unresolved_attempt_failure',10)
                for other,_ in active.values():other.wait()
                raise RuntimeError('P4_attempt_failed_requires_evidence_review')
            if code==2 and stage in ['P4B-anchor','P4C-profile']:
                ledger.stop('P4_anchor_or_profile_numerical_failure',10)
                for other,_ in active.values():other.wait()
                raise RuntimeError('P4_entry_run_failed')
            finished.append(item['id']);del active[device]
            if stage=='P4C-formal':C_forecast()
        for device in [0,1]:
            if device in active:continue
            position=next((i for i,x in enumerate(pending) if x.get('device',device)==device),None)
            if position is None:continue
            item=pending.pop(position)
            if already_done(item):finished.append(item['id']);continue
            storage()
            process,_=launch_task(item['id'],item['kind'],device);active[device]=(process,item)
        write(STAGE/'status.json',{'status':'running','stage':stage,'pending':len(pending),'active':[x[1]['id'] for x in active.values()],
              'finished_this_stage':finished,'P4_GPU_seconds_including_active':ledger.used(ledger.snapshot()),'updated_unix':time.time()})
        time.sleep(2)


def already_done(item):
    if item['kind']=='P4C-test':return (STAGE/'C_results'/item['id']/'completed.json').exists()
    if item['kind']=='P4A3-inference':return (STAGE/'A3_models'/(item['id']+'.json')).exists()
    path=ROOT/'runs'/item['id']/'summary.json'
    return path.exists() and read(path)['status'] in ['completed','numerical_failure','oom']


def check_anchors():
    selected=read(ROOT/'selected_hparams.json')['configurations'];results=[]
    for job in jobs():
        if job['stage']!='P4B-anchor':continue
        key=configuration(job)['configuration_id'];old=selected[key]['selected_from_run_id']
        original=[__import__('json').loads(s) for s in (ROOT/'runs'/old/'epochs.jsonl').read_text().splitlines()][:3]
        new=[__import__('json').loads(s) for s in (ROOT/'runs'/job['run_id']/'epochs.jsonl').read_text().splitlines()]
        assert len(new)==3
        for a,b in zip(original,new):
            for name in ['sampler_order_sha256','cumulative_optimizer_updates','lr_by_group_next_update']:assert a[name]==b[name]
            for name in ['samples','correct','top1_percent','loss']:assert a['validation'][name]==b['validation'][name]
            for name in ['samples','optimizer_updates','loss','top1_percent']:assert a['train'][name]==b['train'][name]
        assert read(ROOT/'runs'/old/'summary.json')['model_identity']['initial_classifier_sha256']==read(ROOT/'runs'/job['run_id']/'summary.json')['model_identity']['initial_classifier_sha256']
        results.append({'run_id':job['run_id'],'P2_reference':old,'three_epoch_numerical_trajectory_exact':True})
    write(STAGE/'B0_anchor_acceptance.json',{'status':'passed','anchors':results,'tolerance_relaxed':False})


def C_forecast():
    import statistics
    estimates={};projected_future=0
    for job in jobs():
        if job['stage']!='P4C-profile':continue
        directory=ROOT/'runs'/job['run_id'];summary=read(directory/'summary.json')
        times=[read(p)['fit_seconds'] for p in sorted((directory/'timings').glob('*.json'))]
        assert len(times)==3
        key=configuration(job)['configuration_id'];steady=statistics.median(times[1:]);fixed=summary['startup_seconds']+max(0,times[0]-steady)+2
        measured=[];fixed_measurements=[]
        for candidate in jobs():
            if candidate['stage']!='P4C-tuning' or f"{candidate['dataset']}-{candidate['model_key']}-{candidate['method']}"!=key:continue
            candidate_directory=ROOT/'runs'/candidate['run_id']
            candidate_times=[read(p)['fit_seconds'] for p in sorted((candidate_directory/'timings').glob('*.json'))]
            if len(candidate_times)==15:
                candidate_summary=read(candidate_directory/'summary.json')
                measured.extend(candidate_times[1:])
                fixed_measurements.append(candidate_summary['startup_seconds']+max(0,candidate_times[0]-statistics.median(candidate_times[1:]))+2)
        if measured:steady=statistics.median(measured);fixed=max(fixed_measurements)
        estimates[key]={'steady_full_epoch_seconds':steady,'fixed_startup_first_epoch_exit_seconds':fixed,
                        'checkpoint_last_bytes':summary['last_checkpoint']['size_bytes'],'checkpoint_best_bytes':summary['best_checkpoint']['size_bytes'],
                        'steady_timing_source':'all complete C1 15-epoch candidate timings' if measured else 'C0 epochs2-3'}
        for candidate in jobs():
            candidate_key=f"{candidate['dataset']}-{candidate['model_key']}-{candidate['method']}"
            if candidate['stage'] not in ['P4C-tuning','P4C-formal'] or candidate_key!=key:continue
            path=ROOT/'runs'/candidate['run_id']/'summary.json'
            previous=read(path) if path.exists() else None
            if previous and previous['status'] in ['completed','numerical_failure','oom']:continue
            completed=previous['completed_epochs'] if previous else 0
            projected_future+=(0 if completed else fixed)+(candidate['epochs']-completed)*steady
    # Flowers 全测试的预估另含逐角色进程启动、读模型和按验证样本规模外推。
    evaluation=sum(10*(e['fixed_startup_first_epoch_exit_seconds']+6149/1020*e['steady_full_epoch_seconds']) for e in estimates.values())
    spent=budget().used(budget().snapshot());forecast=spent+1.25*(projected_future+evaluation)+EXIT
    value={'schema':'P4_C_forecast_v1','actual_spent_seconds':spent,'future_training_seconds':projected_future,'future_test_seconds_conservative':evaluation,
           'predicted_total_with_25percent_future_margin_and_exit_seconds':forecast,'total_cap_seconds':CAP,'admitted':forecast<=CAP,'estimates':estimates,
           'new_training_checkpoint_estimate_bytes':sum(8*(e['checkpoint_last_bytes']+e['checkpoint_best_bytes']) for e in estimates.values())}
    write(STAGE/'C_admission_forecast.json',value)
    if forecast>CAP:budget().stop('P4_C_full_matrix_forecast_over_20_hours',10);raise RuntimeError('P4_budget_forecast_rejected')


def C_seal():
    summaries={};evaluations=[]
    for job in jobs():
        if job['stage']!='P4C-formal':continue
        summary=read(ROOT/'runs'/job['run_id']/'summary.json')
        assert summary['status']=='completed' and summary['completed_epochs']==50,'C3 remains closed when formal matrix incomplete'
        summaries[job['run_id']]=file_hash(ROOT/'runs'/job['run_id']/'summary.json')
        config=configuration(job)
        for role,key in [('primary_best_val','best_checkpoint'),('secondary_last','last_checkpoint')]:
            checkpoint=summary[key];item={'run_id':job['run_id'],'role':role,'seed':job['seed'],'configuration_id':config['configuration_id'],
                'checkpoint':f"runs/{job['run_id']}/{checkpoint['path']}",'checkpoint_sha256':checkpoint['sha256'],
                'scientific_config_sha256':summary['scientific_config_sha256'],'selected_by':'validation_Top1_then_earliest_epoch'}
            item['evaluation_id']='P4C-eval-'+value_hash(item);evaluations.append(item)
    evaluations.sort(key=lambda r:({'primary_best_val':0,'secondary_last':1}[r['role']],r['run_id']))
    manifest={'schema':'P4C_frozen_test_v1','evaluations':evaluations,'test_results_seen':False}
    path=STAGE/'C_evaluation_manifest.json'
    if path.exists():assert read(path)==manifest
    else:write(path,manifest)
    value={'schema':'P4C_training_seal_v1','summary_sha256':summaries,'selection_sha256':file_hash(STAGE/'C_selected_hparams.json'),
        'evaluation_manifest_sha256':file_hash(path),'protocol_frozen_sha256':file_hash(STAGE/'protocol_frozen.json'),
        'conditional_PI_test_approval':True,'test_accessed_before_seal':False}
    path=STAGE/'C_training_sealed.json'
    if path.exists():assert read(path)==value
    else:write(path,value)
    return evaluations


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['launch','guardian']);parser.add_argument('--id');parser.add_argument('--kind');parser.add_argument('--device',type=int)
    parser.add_argument('--attempt');parser.add_argument('--owner',type=int);parser.add_argument('--owner-ticks');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.mode=='guardian':return guardian(args)
    _,_,identity=require_scope()
    with lock(STAGE/'launcher.lock',blocking=False):
        cpu=read(STAGE/'CPU_acceptance.json')
        assert cpu['status']=='passed' and cpu['source_identity_sha256']==value_hash(identity)
        assert all(a['status']=='finalized' for a in budget().snapshot()['attempts'].values()),'Unsettled P4 attempt'
        assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'Foreign/orphan GPU process'
        storage();cpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        analysis=None
        if not (STAGE/'A1_A2_acceptance.json').exists() or read(STAGE/'A1_A2_acceptance.json')['status']!='completed':
            with (STAGE/'A1_A2.log').open('ab') as file:analysis=subprocess.Popen([PYTHON,'tools/p4_analyze.py'],cwd=ROOT,env=cpu_env,stdout=file,stderr=subprocess.STDOUT)
        try:
            selected=read(ROOT/'selected_hparams.json')['configurations']
            for stage in ['P4B-anchor','P4B-sensitivity']:
                items=[]
                for job in jobs():
                    if job['stage']!=stage:continue
                    item={'id':job['run_id'],'kind':stage}
                    if stage=='P4B-anchor':
                        original=selected[configuration(job)['configuration_id']]['selected_from_run_id']
                        item['device']=read(ROOT/'runs'/original/'summary.json')['physical_gpu_index']
                    items.append(item)
                stage_run(stage,items)
                if stage=='P4B-anchor':check_anchors()
            stage_run('P4A3',[{'id':x['run_id'],'kind':'P4A3-inference'} for x in read(STAGE/'A3_models_manifest.json')['models']])
            subprocess.run([PYTHON,'tools/p4_prepare_data.py'],cwd=ROOT,env=cpu_env,check=True)
            stage_run('P4C-profile',[{'id':j['run_id'],'kind':j['stage']} for j in jobs() if j['stage']=='P4C-profile'])
            C_forecast()
            stage_run('P4C-tuning',[{'id':j['run_id'],'kind':j['stage']} for j in jobs() if j['stage']=='P4C-tuning'])
            selection_stage();C_forecast()
            stage_run('P4C-formal',[{'id':j['run_id'],'kind':j['stage']} for j in jobs() if j['stage']=='P4C-formal'])
            evaluations=C_seal()
            for role in ['primary_best_val','secondary_last']:
                stage_run('P4C-test-'+role,[{'id':e['evaluation_id'],'kind':'P4C-test'} for e in evaluations if e['role']==role])
            if analysis is not None:assert analysis.wait()==0
            subprocess.run([PYTHON,'tools/p4_audit.py'],cwd=ROOT,env=cpu_env,check=True)
            write(STAGE/'status.json',{'status':'all_registered_execution_terminal_ready_for_delivery_audit','P4_GPU_seconds':budget().used(budget().snapshot()),
                'P4_D_requires_consolidated_review':True,'updated_unix':time.time()})
        except BaseException as error:
            write(STAGE/'status.json',{'status':'paused_requires_evidence_review','failure_type':type(error).__name__,
                  'P4_GPU_seconds':budget().used(budget().snapshot()),'updated_unix':time.time(),'no_budget_or_scientific_parameters_changed':True})
            raise


if __name__=='__main__':raise SystemExit(main())
