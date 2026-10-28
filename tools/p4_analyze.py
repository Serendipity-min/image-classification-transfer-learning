"""只读既有 NPZ 的后续分析，CUDA 隐藏，原始预测不离开服务器。"""
from collections import defaultdict
import csv
import os
import time
os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
from scipy import stats
from p4_core import ROOT,STAGE,read,write,file_hash,require_scope
from p4_statistics import calibration,holm,paired_image_bootstrap,cluster_bootstrap


SEEDS=[17,42,2026,3407,65537]


def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf8',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def old_prediction(run_id):
    manifest=read(ROOT/'records/p3/evaluation_manifest.draft.json')
    item=next(x for x in manifest['evaluations'] if x['run_id']==run_id and x['evaluation_role']=='primary_best_val')
    value=read(ROOT/'records/p3c/results'/item['evaluation_id']/'completed.json');path=ROOT/value['predictions_path']
    assert file_hash(path)==value['predictions_sha256']
    with np.load(path,allow_pickle=False) as d:
        ids=d['sample_id'].astype(str);labels=d['labels'].copy();logits=d['logits'].copy();prediction=d['prediction'].copy()
    assert len(set(ids))==len(ids)==len(labels) and np.isfinite(logits).all() and np.array_equal(prediction,logits.argmax(1))
    assert labels.min()>=0 and labels.max()<logits.shape[1]
    return ids,labels,logits,value


def main():
    require_scope();started=time.monotonic();report=ROOT/'reports/P4';report.mkdir(parents=True,exist_ok=True)
    runs=read(ROOT/'records/p3/run_manifest.json')['runs'];rows=[];reliability=[];paired={};hashes={}
    contrast_configs={'cifar100-vitb16-linear_probe','cifar100-vitb16-full_ft','svhn-r50-adapter','svhn-r50-full_ft'}
    for run in runs:
        ids,labels,logits,receipt=old_prediction(run['run_id']);hashes[run['run_id']]=receipt['predictions_sha256']
        assert len(ids)==(10000 if run['dataset']=='cifar100' else 26032)
        base={'run_id':run['run_id'],'configuration_id':run['configuration_id'],'seed':run['seed']}
        for bins in [15,10,30]:
            result=calibration(logits,labels,bins)
            rows.append(dict(base,bins=bins,**{k:v for k,v in result.items() if k!='reliability'}))
            reliability.extend(dict(base,bins=bins,**b) for b in result['reliability'])
        if run['configuration_id'] in contrast_configs:
            paired[(run['configuration_id'],run['seed'])]=(ids,labels,logits.argmax(1)==labels)
    rules={'schema':'P4_A_input_rules_v1','primary_role_only':True,'n_models':80,'alignment':'sample_id, same labels, same row count',
           'duplicate_audit_sources':{d:file_hash(ROOT/f'records/{d}_duplicates.json') for d in ['cifar100','svhn']},
           'bootstrap_root_seed':20261027,'stream_mapping':{'cifar100':[0,1,2,3],'svhn':[4,5,6,7]},'source_prediction_sha256':hashes}
    path=STAGE/'A_input_rules.json'
    if path.exists():assert read(path)==rules
    else:write(path,rules)
    children=np.random.SeedSequence(20261027).spawn(8);contrasts=[];mcnemar=[]
    for number,(group,method) in enumerate([('cifar100-vitb16','linear_probe'),('svhn-r50','adapter')]):
        reference=paired[(group+'-full_ft',SEEDS[0])];ids,labels,_=reference;differences=[]
        for seed in SEEDS:
            a=paired[(group+'-'+method,seed)];b=paired[(group+'-full_ft',seed)]
            def align(value):
                order={identifier:i for i,identifier in enumerate(value[0])};assert set(order)==set(ids)
                positions=np.array([order[x] for x in ids]);assert np.array_equal(value[1][positions],labels)
                return value[2][positions]
            aa,bb=align(a),align(b);d=aa.astype(np.int8)-bb.astype(np.int8);differences.append(d)
            discordant_A=int((aa&~bb).sum());discordant_B=int((~aa&bb).sum());total=discordant_A+discordant_B
            p=float(stats.binomtest(discordant_A,total,.5).pvalue) if total else 1.
            mcnemar.append({'comparison':group+'-'+method+'-minus-full_ft','seed':seed,'A_only_correct':discordant_A,'FT_only_correct':discordant_B,
                            'n':len(ids),'p_exact_two_sided':p,'Holm_over_10':None})
        d=np.asarray(differences);dataset=group.split('-')[0]
        clusters=[[i] for i in range(len(ids))];lookup={s:i for i,s in enumerate(ids)};excluded=[];merged=[]
        for entry in read(ROOT/f'records/{dataset}_duplicates.json')['groups']:
            test=[lookup[x['sample_id']] for x in entry['samples'] if x['sample_id'] in lookup]
            if any(x['assigned_split']=='train' for x in entry['samples']):excluded.extend(test)
            if len(test)>1:merged.append(test)
        in_cluster={i for group_ids in merged for i in group_ids};clusters=[x for x in clusters if x[0] not in in_cluster]+merged
        keep=np.ones(len(ids),dtype=bool);keep[excluded]=False
        contrasts.append({'comparison':group+'-'+method+'-minus-full_ft','paired_seed_differences_pp':(d.mean(1)*100).tolist(),
            'fixed_five_model_image_bootstrap':paired_image_bootstrap(d,labels,np.random.default_rng(children[number*4])),
            'crossed_seed_image_bootstrap':paired_image_bootstrap(d,labels,np.random.default_rng(children[number*4+1]),seed_rng=np.random.default_rng(children[number*4+2])),
            'duplicate_clusters':{'n_images':len(ids),'n_clusters':len(clusters),'repeated_clusters':len(merged)},
            'cluster_bootstrap':cluster_bootstrap(d,clusters,np.random.default_rng(children[number*4+3])) if merged else {'status':'NOT_APPLICABLE_no_test_duplicates'},
            'training_pixel_match_exclusion':{'excluded_sample_ids':[str(ids[i]) for i in sorted(set(excluded))],
                'retained_n':int(keep.sum()),'paired_difference_pp':(d[:,keep].mean(1)*100).tolist(),
                'mean_difference_pp':float(d[:,keep].mean()*100),'full_test_remains_primary':True}})
    for row,p in zip(mcnemar,holm([r['p_exact_two_sided'] for r in mcnemar])):row['Holm_over_10']=p
    csv_write(report/'A2_calibration_by_seed.csv',rows);csv_write(report/'A2_reliability_bins.csv',reliability);csv_write(report/'A1_McNemar.csv',mcnemar)
    write(STAGE/'A1_bootstrap.json',{'status':'completed','contrasts':contrasts,'n_resamples':10000,'GPU_seconds':0,'old_t_Holm_replaced':False})
    import resource
    write(STAGE/'A1_A2_acceptance.json',{'status':'completed','primary_models':80,'calibration_rows':len(rows),'mcnemar_tests':10,
          'wall_seconds':time.monotonic()-started,'CPU_peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
          'GPU_seconds':0,'CUDA_VISIBLE_DEVICES_hidden':True,'original_prediction_binaries_remain_server':True,'history_modified':False})
    print('A1/A2 CPU analysis completed: 80 models, shared paired images, 10 McNemar, ECE15/10/30.')


if __name__=='__main__':main()
