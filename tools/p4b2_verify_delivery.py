"""桌面独立复核回收文本与整数选参；不连接服务器，不读取二进制模型。"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def value_sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,required=True);args=parser.parse_args();snap=args.snapshot.resolve()
    files=read(snap/'snapshot_manifest.json')['files']
    for name,d in files.items():assert sha(snap/name)==d['sha256'] and (snap/name).stat().st_size==d['bytes']
    s=ROOT/'records/p4b2';seal=read(s/'P4B2_sealed.json');identity=read(s/'execution_identity.json')
    bindings={'execution_identity_sha256':s/'execution_identity.json','manifest_sha256':s/'run_manifest.json','CPU_acceptance_sha256':s/'CPU_acceptance.json',
              'GPU_ledger_sha256':s/'gpu_ledger.json','acceptance_sha256':s/'acceptance.json','comparison_sha256':s/'comparison.json',
              'validation_table_sha256':ROOT/'reports/P4B2/P4B2_LR_validation.csv','epoch_curves_sha256':ROOT/'reports/P4B2/P4B2_epoch_curves.csv','checkpoint_inventory_sha256':s/'checkpoint_inventory.json'}
    for key,path in bindings.items():assert sha(path)==seal[key]
    for name,digest in identity['files'].items():assert sha(ROOT/name)==digest
    for name,digest in identity['readonly_baseline'].items():
        if not name.startswith('weights/'):assert sha(ROOT/name)==digest
    ledger=read(s/'gpu_ledger.json');assert len(ledger['attempts'])==2 and all(a['status']=='finalized' and a['returncode']==0 for a in ledger['attempts'].values())
    used=sum(a['charged_gpu_seconds'] for a in ledger['attempts'].values());assert used<=1800
    results=[]
    for job in read(s/'run_manifest.json')['runs']:
        directory=snap/'runs'/job['run_id'];summary=read(directory/'summary.json');scientific=read(directory/'scientific.json')
        assert summary['scientific_config_sha256']==value_sha(scientific) and scientific['configuration']==job
        assert summary['model_identity']['initial_classifier_sha256']==read(s/'run_manifest.json')['head_initial_sha256']
        assert summary['status']=='completed' and summary['completed_epochs']==15 and summary['optimizer_updates']==120
        assert not summary['test_evaluator_constructed'] and sha(directory/'summary.json')==seal['summary_sha256'][job['run_id']]
        assert sha(directory/'epochs.jsonl')==seal['epoch_logs_sha256'][job['run_id']]
        rows=[json.loads(x) for x in (directory/'epochs.jsonl').read_text().splitlines()];assert len(rows)==15
        assert all(r['validation']['samples']==1020 and r['train']['optimizer_updates']==8 for r in rows)
        pointer=read(directory/'current.json');chain=[]
        while pointer:
            path=directory/pointer['path'];assert sha(path)==pointer['sha256'];c=read(path);chain.append(c['epoch_row']);pointer=c['previous_commit']
        assert list(reversed(chain))==rows
        total=sum(r['validation']['correct'] for r in rows[-3:]);assert total==summary['selection_correct_sum']
        for group in read(directory/'parameter_groups.json')['groups']:
            assert group['peak_lr']==job['learning_rate']*(10 if group['name'].startswith('fc.') else 1)
        results.append({'run_id':job['run_id'],'LR':job['learning_rate'],'correct_sum':total,'score_percent':100*total/3060})
    with (ROOT/'reports/P4B2/P4B2_LR_validation.csv').open(encoding='utf8') as f:table=list(csv.DictReader(f))
    assert len(table)==5
    for r in results:
        line=next(x for x in table if x['run_id']==r['run_id']);assert int(line['correct_sum'])==r['correct_sum'] and abs(float(line['score_percent'])-r['score_percent'])<1e-12
    winner=min(table,key=lambda r:(-int(r['correct_sum']),float(r['backbone_LR'])))
    comparison=read(s/'comparison.json');assert float(winner['backbone_LR'])==comparison['selected_LR_in_five_point_diagnostic']
    assert abs((int(winner['correct_sum'])-2502)/3060*100-comparison['delta_FT_early_pp'])<1e-12
    proof={'schema':'P4B2_desktop_independent_acceptance_v1','status':'passed','text_snapshot_files_verified':len(files),'completed_runs':2,'epochs':30,
           'optimizer_updates_each':120,'integer_selection_recomputed':True,'results':results,'GPU_hours_recomputed':used/3600,'cap_GPU_hours':.5,
           'sealed_GPU_ledger_raw_sha256_verified':True,'server_terminal_seal_sha256':sha(s/'P4B2_sealed.json'),'paper_phase_not_executed':True,
           'original_models_and_predictions_not_downloaded':True}
    (s/'desktop_acceptance.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf8',newline='\n')
    print(json.dumps(proof,ensure_ascii=False))


if __name__=='__main__':main()
