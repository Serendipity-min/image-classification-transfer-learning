"""Recompute only the source-extracted key claims; no remote/model access."""
from pathlib import Path
import json, math
import numpy as np
from scipy import stats
ROOT=Path(__file__).resolve().parent
source_commit='2b19bedca2b13b6ee3d2ebd2e0b8ba244722ca6d'
pairs={
 'cifar100-vitb16-LP-minus-FT':[-5.510000000000005,-5.890000000000001,-6.050000000000011,-5.709999999999994,-5.540000000000006],
 'svhn-r50-Adapter-minus-FT':[0.10755992624463317,0.22664413030116748,0.27274124154885726,0.28810694529809666,0.6107867240319678]
}
rows=[]
for name, raw in pairs.items():
 d=np.asarray(raw); n=d.size; m=float(d.mean()); sd=float(d.std(ddof=1));h=float(stats.t.ppf(.975,n-1)*sd/math.sqrt(n))
 rows.append(dict(contrast=name,paired_differences_pp=raw,mean_pp=m,sd_pp=sd,ci95_pp=[m-h,m+h],p_two_sided=float(stats.ttest_1samp(d,0).pvalue),wilcoxon_exact_p=float(stats.wilcoxon(d,method='exact').pvalue)))
prev=0
for rank,i in enumerate(sorted(range(len(rows)),key=lambda i:rows[i]['p_two_sided'])):
 prev=max(prev,min(1,(len(rows)-rank)*rows[i]['p_two_sided']));rows[i]['p_holm']=prev
seconds=sum([50.09319852106273,139.4002377539873,114.16073771938682,119.50692279823124])
report={
 'source_commit':source_commit,
 'scope':'key claims recomputed from exact values read through GitHub connector; not a fresh audit of all 400 NPZ/model binaries',
 'primary_contrasts':rows,
 'LP_best_epochs_in_seed_order':[3,7,3,35,3],
 'LP_epoch3_count':3,
 'critical_boundary_LR_configurations_count':4,
 'critical_boundary_LR_configurations':['cifar100-vitb16-linear_probe','cifar100-vitb16-full_ft','svhn-r50-full_ft','svhn-r50-adapter'],
 'B_timing_estimate':{
   'basis':'sum of four P2 complete-epoch medians, excluding new unmeasured fixed overhead',
   'four_config_sum_epoch_seconds':seconds,
   'eight_new_15epoch_gpu_hours':seconds*30/3600,
   'four_anchor_3epoch_gpu_hours':seconds*3/3600,
   'combined_25percent_margin_gpu_hours':seconds*33/3600*1.25,
   'proposed_workstream_cap_gpu_hours':6.0
 },
 'marginal_iid_binomial_SE_pp':{
   'svhn_r50_adapter':100*math.sqrt(.965742163491088*(1-.965742163491088)/26032),
   'cifar100_vit_adapter':100*math.sqrt(.93588*(1-.93588)/10000),
   'warning':'marginal SEs are not SEs of paired accuracy differences; dependence and duplicates are not accounted for'
 },
 'new_training_executed':False,
 'new_model_inference_executed':False
}
(ROOT/'04_关键复核.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=False,indent=2))
