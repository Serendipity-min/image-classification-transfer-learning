"""CPU交付补审：原始预测/遮挡网格在服务器读取，模型仅核对库存大小。"""
from collections import Counter,defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path('/root/user_data/tl_cls_v1');sys.path.insert(0,str(ROOT/'tools'));sys.path.insert(0,str(ROOT/'src'))
from p4_core import STAGE,require_scope,read,write,file_hash,value_hash


def csv_write(path,rows):
    with path.open('w',encoding='utf8',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def size(directory):
    seen=set();logical=0;count=0
    for folder,dirs,files in os.walk(directory):
        for name in files:
            p=Path(folder)/name;st=p.lstat()
            if not p.is_file() or p.is_symlink():continue
            if (st.st_dev,st.st_ino) in seen:continue
            seen.add((st.st_dev,st.st_ino));logical+=st.st_size;count+=1
    return {'logical_regular_bytes_deduplicated':logical,'files':count}


def main():
    _,_,identity=require_scope();report=ROOT/'reports/P4';report.mkdir(parents=True,exist_ok=True)
    ledger=read(STAGE/'gpu_ledger.json');assert all(a['status']=='finalized' for a in ledger['attempts'].values())
    seal=read(STAGE/'P4_sealed.json')
    for key,path in [('protocol_sha256','protocol_frozen.json'),('delivery_audit_sha256','delivery_audit.json'),('C_primary_sha256','C_primary_noninferiority.json')]:assert seal[key]==file_hash(STAGE/path)
    ledger_raw_sha=file_hash(STAGE/'gpu_ledger.json')
    ledger_seal_match=seal['gpu_ledger_sha256']==ledger_raw_sha
    spent=sum(a['charged_gpu_seconds'] for a in ledger['attempts'].values())
    assert abs(spent-read(STAGE/'delivery_audit.json')['P4_GPU_hours']*3600)<1e-6
    for a in ledger['attempts'].values():
        assert a['charged_gpu_seconds']==a['ended_monotonic']-a['started_monotonic']
    from scipy.io import loadmat
    official_labels=loadmat(ROOT/'data/flowers-102/imagelabels.mat',squeeze_me=True)['labels']-1
    official_ids=read(STAGE/'flowers_integrity.json')['splits']['test']['sample_ids']
    lookup={f'flowers102/image_{i:05d}':int(label) for i,label in enumerate(official_labels,1)}
    test_count=0;test_bytes=0
    for p in (STAGE/'C_results').glob('*/completed.json'):
        receipt=read(p);prediction=ROOT/receipt['data_path'];assert file_hash(prediction)==receipt['data_sha256']
        with np.load(prediction,allow_pickle=False) as d:
            ids=d['sample_id'].astype(str).tolist();labels=d['labels'];assert ids==official_ids
            assert labels.tolist()==[lookup[i] for i in ids] and len(ids)==6149
        test_count+=1;test_bytes+=prediction.stat().st_size
    assert test_count==80
    records=[];grid_cells=[];actual_calls=0;grid_bytes=0
    for p in sorted((STAGE/'A3_results').glob('*/*/completed.json')):
        receipt=read(p);path=ROOT/receipt['data_path'];assert file_hash(path)==receipt['data_sha256']
        with np.load(path,allow_pickle=False) as d:
            x=np.concatenate([d['original_logits'][None,:],d['masked_logits']],axis=0);label=int(d['label']);sample=str(d['sample_id'])
            assert x.shape[0]==197 and np.isfinite(x).all()
            shifted=x.astype(np.float64)-x.max(1,keepdims=True);prob=np.exp(shifted);prob/=prob.sum(1,keepdims=True)
            py=(prob[0,label]-prob[1:,label]).reshape(14,14);ly=(x[0,label]-x[1:,label]).reshape(14,14)
            assert np.array_equal(py,d['probability_drop']) and np.array_equal(ly,d['logit_drop'])
            run=p.parent.parent.name;parts=run.split('-');dataset=parts[1];model=parts[2];method=parts[3]
            records.append({'run_id':run,'dataset':dataset,'model_key':model,'method':method,'sample_id':sample,'label':label,
                'min_probability_drop':float(py.min()),'max_probability_drop':float(py.max()),'mean_probability_drop':float(py.mean()),
                'mean_abs_probability_drop':float(np.abs(py).mean()),'negative_cells':int((py<0).sum()),'image_forwards':197,
                'grid_path':path.relative_to(ROOT).as_posix(),'grid_sha256':receipt['data_sha256']})
            grid_cells.extend({'run_id':run,'dataset':dataset,'model_key':model,'method':method,'sample_id':sample,'row':i,'col':j,'probability_drop':float(py[i,j])} for i in range(14) for j in range(14))
        actual_calls+=receipt['image_forwards'];grid_bytes+=path.stat().st_size
    assert len(records)==320 and actual_calls==63040
    csv_write(report/'A3_grid_inventory.csv',records)
    csv_write(report/'A3_gallery_source_SERVER_ONLY.csv',grid_cells)
    bygroup=defaultdict(list)
    for row in grid_cells:bygroup[(row['run_id'],row['row'],row['col'])].append(row['probability_drop'])
    mean_maps=[{'run_id':run,'row':i,'col':j,'n_images':len(v),'mean_probability_drop':float(np.mean(v)),'sample_SD':float(np.std(v,ddof=1))} for (run,i,j),v in bygroup.items()]
    csv_write(report/'A3_mean_response_maps.csv',mean_maps)
    inventory=[];history={}
    specs=[('P1',read(ROOT/'records/checkpoint_inventory.json')['files']),('P2',read(ROOT/'records/p2/checkpoint_inventory.json')['P2_files']),('P3',read(ROOT/'records/p3/checkpoint_inventory.json')['items'])]
    for phase,items in specs:
        for item in items:
            p=ROOT/item['path'];assert p.is_file() and p.stat().st_size==item['size_bytes']
        history[phase]={'count':len(items),'bytes':sum(x['size_bytes'] for x in items),'size_and_presence_match':True}
    from p4_core import jobs
    for job in jobs():
        directory=ROOT/'runs'/job['run_id'];summary=read(directory/'summary.json')
        for role,key in [('best','best_checkpoint'),('last','last_checkpoint')]:
            desc=summary[key];p=directory/desc['path'];assert p.is_file() and p.stat().st_size==desc['size_bytes']
            inventory.append({'run_id':job['run_id'],'stage':job['stage'],'role':role,'path':p.relative_to(ROOT).as_posix(),
                              'sha256_from_original_transaction':desc['sha256'],'size_bytes':desc['size_bytes']})
    csv_write(report/'checkpoint_inventory.csv',inventory)
    times=[]
    for job in jobs():
        if job['stage']!='P4C-profile':continue
        for p in sorted((ROOT/'runs'/job['run_id']/'timings').glob('*.json')):
            row=read(p);row=dict(run_id=job['run_id'],**row);times.append(row)
    assert len(times)==24
    assert Counter(t['measurement_mode'] for t in times)=={'C0_isolated_epoch1':8,'C0_planned_parallel_epochs2_3':16}
    csv_write(report/'C0_isolated_parallel_timings.csv',times)
    cseal=ROOT/'records/p4/C_training_sealed.json';first_test=min(a['started_unix'] for a in ledger['attempts'].values() if a['kind']=='P4C-test')
    assert cseal.stat().st_mtime<first_test
    result={'schema':'P4_delivery_server_supplement_v1','status':'passed','checked_unix':time.time(),
       'auditor_source_sha256':file_hash(Path(__file__)),'C_official_test_ID_and_label_alignment_verified':80,
       'C_seal_mtime_unix':cseal.stat().st_mtime,'first_C_test_attempt_unix':first_test,'test_attempts_start_after_seal':True,
       'A3_grids_hash_and_numerics_recomputed':320,'A3_actual_image_forwards':actual_calls,'A3_raw_NPZ_bytes':grid_bytes,
       'C_raw_prediction_files':test_count,'C_raw_prediction_bytes':test_bytes,'P4_checkpoints':len(inventory),'P4_checkpoint_bytes':sum(x['size_bytes'] for x in inventory),
       'historical_checkpoint_inventories':history,'binary_model_hashes_repeated':False,'raw_prediction_or_checkpoint_downloaded':False,
       'C0_isolated_epochs_verified':8,'C0_planned_parallel_epochs_verified':16,
       'project':size(ROOT),'environment':size(Path('/root/miniconda3/envs/tl_cls_v1')),
       'filesystem_free_bytes':__import__('shutil').disk_usage(ROOT).free,'exit_and_atomic_reserve_bytes':14*2**30,
       'metadata_deviation':'C evaluation manifest selected_by text also says validation rule on secondary_last; actual roles and checkpoint paths use last.e050; preserve original and explain in delivery report',
       'gallery_source_SERVER_ONLY_sha256':file_hash(report/'A3_gallery_source_SERVER_ONLY.csv')}
    result['original_operational_seal_ledger_binding']={'matches':ledger_seal_match,
        'expected_raw_sha256':seal['gpu_ledger_sha256'],'current_raw_sha256':ledger_raw_sha,
        'original_sealed_ledger_bytes_available':False,'original_seal_preserved':True,
        'observed_current_total_equals_original_audit_total':True,
        'cause_from_execution_path':'P4 supervisor calls Budget.snapshot after p4_audit writes seal; snapshot writes updated_monotonic even when all attempts finalized',
        'limitation':'cannot compare original ledger bytes; old raw binding does not pass; corrected delivery seal must supersede this binding, never silently overwrite original'}
    assert result['filesystem_free_bytes']>14*2**30
    write(STAGE/'delivery_server_supplement.json',result)
    print(json.dumps({k:result[k] for k in ['status','A3_grids_hash_and_numerics_recomputed','C_official_test_ID_and_label_alignment_verified','P4_checkpoints','P4_checkpoint_bytes','C_raw_prediction_bytes']}))


if __name__=='__main__':main()
