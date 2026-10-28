"""P4-A3 验证遮挡与条件式 Flowers 测试；不更新模型。"""
import argparse
import time
from p4_core import ROOT,STAGE,require_scope,jobs,configuration,budget,read,write,file_hash,value_hash
from p4_atomic import publish,completed


def flowers_test_gate(item):
    a,_,_=require_scope();assert a['conditional_flowers_test_authorized']
    seal=read(STAGE/'C_training_sealed.json');manifest=read(STAGE/'C_evaluation_manifest.json')
    assert seal['evaluation_manifest_sha256']==file_hash(STAGE/'C_evaluation_manifest.json')
    assert seal['selection_sha256']==file_hash(STAGE/'C_selected_hparams.json')
    assert item in manifest['evaluations'] and len(manifest['evaluations'])==80
    for run_id,digest in seal['summary_sha256'].items():
        path=ROOT/'runs'/run_id/'summary.json';assert file_hash(path)==digest and read(path)['status']=='completed'
    assert len(seal['summary_sha256'])==40
    return seal


def infer_test(item,device,attempt):
    flowers_test_gate(item)
    directory=STAGE/'C_results'/item['evaluation_id'];identity=value_hash(item)
    if completed(directory,identity):return
    import torch
    import numpy as np
    from tlcls.common import setup
    from tlcls.data import loader
    from p4_data_models import build_flowers,FlowerData
    from p3c_analyze import independent_metrics
    from p4_statistics import calibration
    setup(item['seed']);torch.cuda.set_device(device);target=torch.device('cuda',device)
    checkpoint=ROOT/item['checkpoint'];assert file_hash(checkpoint)==item['checkpoint_sha256']
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False);config=configuration(next(j for j in jobs() if j['run_id']==item['run_id']))
    assert payload['config']==config and payload['scientific_config_sha256']==item['scientific_config_sha256']
    model,_=build_flowers(config,target);model.load_state_dict(payload['model'],strict=True);del payload;model.eval()
    dataset=FlowerData(config['model_key'],'test',config['seed'],allow_test=True);stream,_=loader(dataset,config['seed'],False)
    torch.cuda.reset_peak_memory_stats(target);logits=[];labels=[];ids=[];started=time.monotonic()
    with torch.no_grad():
        for images,y,sample in stream:
            with torch.autocast('cuda',dtype=torch.bfloat16):scores=model(images.to(target,non_blocking=True))
            scores=scores.float();assert torch.isfinite(scores).all()
            logits.append(scores.cpu().numpy());labels.append(y.numpy());ids.extend(sample)
    x=np.concatenate(logits);y=np.concatenate(labels);assert len(y)==6149 and len(set(ids))==6149
    metric=independent_metrics(x,y)
    value=publish(directory,identity,lambda f:np.savez_compressed(f,logits=x,labels=y,prediction=x.argmax(1),sample_id=np.asarray(ids)),
        {'item':item,'metrics':metric,'attempt_id':attempt,'test_used_for_selection':False,'completed_unix':time.time(),
         'inference_seconds':time.monotonic()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated(target),
         'peak_reserved_bytes':torch.cuda.max_memory_reserved(target)})
    if item['role']=='primary_best_val':
        write(directory/'calibration.json',{'configuration_id':config['configuration_id'],'seed':config['seed'],
              'source_prediction_sha256':value['data_sha256'],'bins':{str(b):calibration(x,y,b) for b in [15,10,30]}})


def occlusion(item,device,attempt):
    require_scope()
    import torch
    import numpy as np
    from tlcls.common import setup
    from tlcls.data import TrainValData
    from tlcls.models import build_model
    setup(42);torch.cuda.set_device(device);target=torch.device('cuda',device)
    primary=read(ROOT/'records/p3/evaluation_manifest.draft.json')
    source=next(x for x in primary['evaluations'] if x['run_id']==item['run_id'] and x['evaluation_role']=='primary_best_val')
    model_descriptor=next(x for x in read(ROOT/'records/p3/run_manifest.json')['runs'] if x['run_id']==item['run_id'])
    checkpoint=ROOT/source['checkpoint'];assert file_hash(checkpoint)==source['checkpoint_sha256']
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model,_=build_model(model_descriptor,target);model.load_state_dict(payload['model'],strict=True);del payload;model.eval()
    before={n:b.detach().cpu().clone() for n,b in model.named_buffers() if 'running_' in n or 'num_batches_tracked' in n}
    dataset=TrainValData(model_descriptor['dataset'],model_descriptor['model_key'],'validation',42)
    pool=read(STAGE/'A3_display_pool.json')['datasets'][model_descriptor['dataset']]
    images=[];count=0
    torch.cuda.reset_peak_memory_stats(target)
    for sample in pool:
        directory=STAGE/'A3_results'/item['run_id']/str(sample['raw_index'])
        identity=value_hash({'source':source,'sample':sample,'pool_sha256':file_hash(STAGE/'A3_display_pool.json'),'patch':16,'stride':16})
        prior=completed(directory,identity)
        if prior:images.append(prior);continue
        offset=dataset.indices.index(sample['raw_index']);image,label,sample_id=dataset[offset]
        assert sample_id==sample['sample_id'] and label==sample['label']
        stack=image.unsqueeze(0).repeat(197,1,1,1)
        for k in range(196):
            row,col=divmod(k,14);stack[k+1,:,row*16:(row+1)*16,col*16:(col+1)*16]=0
        scores=[]
        with torch.no_grad():
            for start in range(0,197,128):
                with torch.autocast('cuda',dtype=torch.bfloat16):v=model(stack[start:start+128].to(target))
                scores.append(v.float().cpu())
        logits=torch.cat(scores).numpy();assert np.isfinite(logits).all()
        shifted=logits.astype(np.float64)-logits.max(1,keepdims=True);p=np.exp(shifted);p/=p.sum(1,keepdims=True)
        probability_map=(p[0,label]-p[1:,label]).reshape(14,14);logit_map=(logits[0,label]-logits[1:,label]).reshape(14,14)
        images.append(publish(directory,identity,lambda f:np.savez_compressed(f,probability_drop=probability_map,logit_drop=logit_map,
            original_logits=logits[0],masked_logits=logits[1:],sample_id=np.asarray(sample_id),label=np.asarray(label)),
            {'attempt_id':attempt,'sample_id':sample_id,'checkpoint_sha256':source['checkpoint_sha256'],'image_forwards':197,'role':'validation_occlusion_only'}))
        count+=197
    after=dict(model.named_buffers());assert all(torch.equal(v,after[n].detach().cpu()) for n,v in before.items())
    write(STAGE/'A3_models'/f"{item['run_id']}.json",{'status':'completed','images':len(images),'total_required_forwards':3940,
          'new_image_forwards_this_attempt':count,'BN_buffers_unchanged':True,'pool_sha256':file_hash(STAGE/'A3_display_pool.json'),
          'source_checkpoint_sha256':source['checkpoint_sha256'],'grids':[v['data_path'] for v in images],
          'peak_allocated_bytes':torch.cuda.max_memory_allocated(target),'peak_reserved_bytes':torch.cuda.max_memory_reserved(target)})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['A3','C3'],required=True);parser.add_argument('--id',required=True);parser.add_argument('--device',type=int,required=True);args=parser.parse_args()
    import os
    attempt=os.environ['TLCLS_ATTEMPT_ID'];lease=budget().snapshot()['attempts'][attempt]
    assert lease['status']=='running' and lease['device']==args.device and lease['run_id']==args.id
    if args.mode=='C3':
        item=next(x for x in read(STAGE/'C_evaluation_manifest.json')['evaluations'] if x['evaluation_id']==args.id)
        infer_test(item,args.device,attempt)
    else:
        assert args.id in [x['run_id'] for x in read(STAGE/'A3_models_manifest.json')['models']]
        occlusion({'run_id':args.id},args.device,attempt)


if __name__=='__main__':main()
