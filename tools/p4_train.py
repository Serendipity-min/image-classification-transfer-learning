"""登记矩阵的训练入口；旧数学更新和 epoch 事务原样复用。"""
import argparse
import math
import os
import signal
import time
from p4_core import ROOT,STAGE,require_scope,jobs,configuration,budget,storage,read,write,value_hash,file_hash


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run-id',required=True);parser.add_argument('--device',type=int,required=True);parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    _,frozen,identity=require_scope();job=next(j for j in jobs() if j['run_id']==args.run_id);config=configuration(job)
    ledger=budget();attempt=os.environ['TLCLS_ATTEMPT_ID'];lease=ledger.snapshot()['attempts'][attempt]
    assert lease['run_id']==args.run_id and lease['device']==args.device and lease['status']=='running'
    import torch
    from tlcls.common import setup,rng_state,restore_rng,sync_time
    from tlcls.data import TrainValData,loader,epoch_order_hash
    from tlcls.models import build_model,make_optimizer,parameter_counts,apply_strategy
    from tlcls.engine import UpdateScheduler,train_epoch,validate
    from tlcls.p2_transaction import EpochTransaction
    from p4_data_models import build_flowers,FlowerData
    directory=ROOT/'runs'/args.run_id;directory.mkdir(parents=True,exist_ok=True)
    scientific={'configuration':config,'mathematical_files':{n:h for n,h in identity['files'].items() if n.startswith('src/tlcls/') or n=='tools/p4_data_models.py'},
                'protocol_frozen_sha256':identity['protocol_frozen_sha256'],'Flowers_integrity_sha256':file_hash(STAGE/'flowers_integrity.json') if config['dataset']=='flowers102' else None}
    digest=value_hash(scientific);path=directory/'scientific.json'
    if path.exists():assert args.resume and read(path)==scientific
    else:assert not args.resume;write(path,scientific)
    transaction=EpochTransaction(directory,digest);current,rows=transaction.load()
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    torch.cuda.set_device(args.device);setup(config['seed']);device=torch.device('cuda',args.device)
    torch.cuda.reset_peak_memory_stats(device)
    summary={'run_id':args.run_id,'stage':job['stage'],'configuration_id':config['configuration_id'],'learning_rate':config['learning_rate'],
             'seed':config['seed'],'status':'running','planned_epochs':config['epochs'],'completed_epochs':current['completed_epoch'] if current else 0,
             'scientific_config_sha256':digest,'execution_identity_sha256':value_hash(identity),'execution_source_commit':identity['P4_source_commit'],
             'test_evaluator_constructed':False,'selection_correct_sum':None,'actual_device':args.device}
    write(directory/'summary.json',summary);model=optimizer=scheduler=None
    started=time.monotonic()
    try:
        if config['dataset']=='flowers102':
            model,model_info=build_flowers(config,device)
            train_data=FlowerData(config['model_key'],'train',config['seed']);val_data=FlowerData(config['model_key'],'validation',config['seed'])
        else:
            model,model_info=build_model(config,device)
            train_data=TrainValData(config['dataset'],config['model_key'],'train',config['seed'])
            val_data=TrainValData(config['dataset'],config['model_key'],'validation',config['seed'],raw=train_data.raw)
        optimizer,groups=make_optimizer(model,config);scheduler=UpdateScheduler(optimizer,math.ceil(len(train_data)/128))
        train_loader,sampler=loader(train_data,config['seed'],True);val_loader,_=loader(val_data,config['seed'],False)
        write(directory/'parameter_groups.json',{'groups':groups,'scientific_config_sha256':digest})
        if current:
            payload=torch.load(transaction.path(current['last']['path']),map_location='cpu',weights_only=False)
            assert payload['scientific_config_sha256']==digest and payload['config']==config
            model.load_state_dict(payload['model'],strict=True);optimizer.load_state_dict(payload['optimizer']);scheduler.load_state_dict(payload['scheduler'])
            apply_strategy(model,config['model_key'],config['method'],training=True);restore_rng(payload['rng'])
            del payload
        summary.update(model_identity=model_info,parameters=parameter_counts(model,config['model_key']),startup_seconds=time.monotonic()-started)
        best_epoch=current['best_epoch'] if current else None;best=max([r['validation']['top1_percent'] for r in rows],default=-1)
        for epoch in range(len(rows),config['epochs']):
            snapshot=ledger.snapshot()
            if stopped or snapshot['stop_request']:raise InterruptedError('P4_stop_before_epoch')
            storage();sampler.epoch=epoch
            def heartbeat(value):write(directory/'progress.json',dict(value,attempt_id=attempt,updated_unix=time.time()))
            begin=sync_time();training=train_epoch(model,optimizer,scheduler,train_loader,sampler,config,epoch,device,
                deadline=time.monotonic()+max(0,(ledger.cap-ledger.used(snapshot)-60)/2),heartbeat=heartbeat)
            train_seconds=sync_time()-begin;begin=sync_time();validation=validate(model,val_loader,device);val_seconds=sync_time()-begin
            improved=validation['top1_percent']>best
            if improved:best=validation['top1_percent'];best_epoch=epoch+1
            payload={'schema':'P4_epoch_v1','model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                     'rng':rng_state(),'config':config,'scientific_config_sha256':digest,'completed_epoch':epoch+1,'best_epoch':best_epoch,
                     'execution_identity_sha256':value_hash(identity)}
            selected={'model':model.state_dict(),'config':config,'scientific_config_sha256':digest,'epoch':epoch+1,'validation':validation}
            row={'epoch':epoch+1,'attempt_id':attempt,'train':training,'validation':validation,'train_seconds':train_seconds,'validation_seconds':val_seconds,
                 'sampler_order_sha256':epoch_order_hash(sampler),'cumulative_optimizer_updates':scheduler.updates,
                 'lr_by_group_next_update':[g['lr'] for g in optimizer.param_groups]}
            begin=sync_time()
            current,row=transaction.commit(epoch+1,row,lambda file:torch.save(payload,file),
                (lambda file:torch.save(selected,file)) if improved else None,best_epoch,lambda:ledger.run_seconds(args.run_id))
            checkpoint_seconds=sync_time()-begin
            write(directory/'timings'/f'epoch_{epoch+1:03d}.json',{'epoch':epoch+1,'train_seconds':train_seconds,'validation_seconds':val_seconds,
                  'checkpoint_seconds':checkpoint_seconds,'fit_seconds':train_seconds+val_seconds+checkpoint_seconds})
            rows.append(row);summary.update(completed_epochs=epoch+1,selected_epoch=best_epoch,best_val_top1_percent=best,
                best_checkpoint=current['best'],last_checkpoint=current['last'],optimizer_updates=scheduler.updates)
            write(directory/'summary.json',summary)
            print(f'{args.run_id} epoch={epoch+1} committed',flush=True)
        summary['status']='completed'
        if config['epochs']==15:
            assert [r['epoch'] for r in rows[-3:]]==[13,14,15]
            summary['selection_correct_sum']=sum(r['validation']['correct'] for r in rows[-3:])
            summary['selection_validation_n']=len(val_data)
    except (InterruptedError,TimeoutError,KeyboardInterrupt) as error:
        summary.update(status='infrastructure_interrupted',failure_type=type(error).__name__)
    except (FloatingPointError,torch.cuda.OutOfMemoryError) as error:
        summary.update(status='numerical_failure' if isinstance(error,FloatingPointError) else 'oom',failure_type=type(error).__name__,
                       failure_diagnostics=getattr(error,'diagnostics',{}))
        if model is not None:
            summary['failure_state']=transaction.artifact('failure',summary['completed_epochs']+1,
                lambda file:torch.save({'model':model.state_dict(),'optimizer':optimizer.state_dict() if optimizer else None,'failure':summary},file))
    finally:
        summary.update(peak_allocated_bytes=torch.cuda.max_memory_allocated(device),peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                       finalized_unix=time.time(),attempt_gpu_seconds_observed=ledger.run_seconds(args.run_id))
        write(directory/'summary.json',summary)
    return 0 if summary['status']=='completed' else 75 if summary['status']=='infrastructure_interrupted' else 2


if __name__=='__main__':raise SystemExit(main())
