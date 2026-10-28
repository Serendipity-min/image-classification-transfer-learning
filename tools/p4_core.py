"""P4 独立登记与预算；历史科学模块和台账只读。"""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(os.environ.get('TLCLS_ROOT','/root/user_data/tl_cls_v1')).resolve()
sys.path.insert(0,str(ROOT/'src'))
from tlcls.p2_core import read,write,canonical,file_hash,value_hash,process_alive,process_ticks
from tlcls.p3_budget import Budget

STAGE=ROOT/'records/p4'
CAP=72000
EXIT=60


@contextmanager
def lock(path,blocking=True):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as file:
        fcntl.flock(file,fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def budget():
    return Budget(STAGE/'gpu_ledger.json',CAP)


def storage():
    free=shutil.disk_usage(ROOT).free
    value={'checked_unix':time.time(),'filesystem_free_bytes':free,'required_free_bytes':14*2**30,
           'historical_checkpoints_deleted':False,'artificial_project_cap':False}
    write(STAGE/'storage_latest.json',value)
    if free<14*2**30:raise RuntimeError('physical_storage_exit_reserve')
    return value


def require_scope():
    a=read(STAGE/'authorization.json');f=read(STAGE/'protocol_frozen.json');identity=read(STAGE/'execution_identity.json')
    assert a['approval_mode']=='explicit_user_PI_batch_adoption' and a['total_cap_GPU_hours']==20
    assert f['execution_authorized'] and f['Flowers_primary_and_grids_frozen_before_new_B_A_results']
    assert file_hash(STAGE/'protocol_frozen.json')==identity['protocol_frozen_sha256']
    for name,digest in identity['files'].items():assert file_hash(ROOT/name)==digest, 'P4 execution identity changed'
    for name,digest in identity['historical_read_only_files'].items():assert file_hash(ROOT/name)==digest,'Historical evidence changed'
    return a,f,identity


def jobs():
    return read(STAGE/'protocol_frozen.json')['design_matrix']['training_jobs']


def configuration(job):
    config=dict(job)
    config.update(phase='P4',configuration_id=f"{job['dataset']}-{job['model_key']}-{job['method']}",effective_batch_size=128)
    if config['learning_rate'] is None:
        config['learning_rate']=read(STAGE/'C_selected_hparams.json')['configurations'][config['configuration_id']]['learning_rate']
    return config


def select(rows,expected_epochs=15):
    valid=[r for r in rows if r['status']=='completed' and r['completed_epochs']==expected_epochs and r.get('selection_correct_sum') is not None]
    assert valid,'No finite complete LR candidate'
    return min(valid,key=lambda r:(-r['selection_correct_sum'],r['learning_rate']))


def selection_stage():
    selected={}
    for key in sorted({configuration(j)['configuration_id'] for j in jobs() if j['stage']=='P4C-tuning'}):
        candidates=[]
        for job in jobs():
            if job['stage']!='P4C-tuning' or configuration(job)['configuration_id']!=key:continue
            summary=read(ROOT/'runs'/job['run_id']/'summary.json');candidates.append(summary)
        winner=select(candidates)
        selected[key]={'learning_rate':winner['learning_rate'],'run_id':winner['run_id'],'selection_correct_sum':winner['selection_correct_sum'],
                       'validation_n':1020,'score_percent':winner['selection_correct_sum']/(3*1020)*100,'candidates':candidates}
    value={'schema':'P4C_selection_v1','configurations':selected,'test_accessed':False,'tie_rule':'smaller_LR'}
    path=STAGE/'C_selected_hparams.json'
    if path.exists():assert read(path)==value
    else:write(path,value)
    return value


def stage_seconds(snapshot,prefix):
    now=time.monotonic()
    return sum(a['charged_gpu_seconds'] if a['status']=='finalized' else now-a['started_monotonic']
               for a in snapshot['attempts'].values() if a['kind'].startswith(prefix))
