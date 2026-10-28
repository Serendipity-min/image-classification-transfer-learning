"""新增 Flowers 适配层，旧数据和模型函数保持原样。"""
import math
import numpy as np
from PIL import Image
import torch
from torch import nn
import timm
from safetensors.torch import load_file
from torchvision.datasets import Flowers102
from torchvision.transforms import functional as TF,InterpolationMode
from p4_core import ROOT,STAGE,read
from tlcls.common import rng_stream,derive_seed,tensor_hash
from tlcls.models import head_of,insert_adapters,apply_strategy


def build_flowers(config,device):
    protocol=read(ROOT/'configs/protocol.json');key=config['model_key'];method=config['method'];seed=config['seed']
    kwargs={'num_classes':102,'drop_rate':0.,'drop_path_rate':0.}
    if key=='vitb16':kwargs.update(attn_drop_rate=0.,proj_drop_rate=0.,global_pool='token')
    with rng_stream(derive_seed('model',seed,key)):
        model=timm.create_model(protocol['models'][key]['model_id'],pretrained=False,**kwargs)
    if method!='scratch':
        state=load_file(str(ROOT/'weights'/key/'model.safetensors'));classifier='fc' if key=='r50' else 'head'
        assert state[classifier+'.weight'].shape[0]==(1000 if key=='r50' else 21843)
        del state[classifier+'.weight'],state[classifier+'.bias']
        result=model.load_state_dict(state,strict=False)
        assert set(result.missing_keys)=={classifier+'.weight',classifier+'.bias'} and not result.unexpected_keys
        del state
    head=head_of(model,key)
    with rng_stream(derive_seed('head',seed,'flowers102',key)):
        nn.init.trunc_normal_(head.weight,std=.02);nn.init.zeros_(head.bias)
    info={'initial_classifier_sha256':tensor_hash(head.state_dict().items()),'target_classes':102,'constructor_kwargs':kwargs,
          'pretrained_loaded':method!='scratch','pretraining_dataset':protocol['models'][key]['pretraining_dataset'] if method!='scratch' else None}
    if method=='adapter':insert_adapters(model,key,seed)
    apply_strategy(model,key,method)
    return model.to(device),info


class FlowerData(torch.utils.data.Dataset):
    def __init__(self,key,split,seed,allow_test=False):
        assert split in ['train','validation'] or (split=='test' and allow_test)
        self.dataset='flowers102';self.model_key=key;self.split=split;self.seed=seed
        self.raw=Flowers102(str(ROOT/'data'),split='val' if split=='validation' else split,download=False)
        self.normalization=read(ROOT/'configs/protocol.json')['models'][key]
        manifest=read(STAGE/'flowers_integrity.json');assert manifest['status']=='passed'
        self.ids=manifest['splits'][split]['sample_ids']
        assert len(self.raw)==len(self.ids) and set(self.raw._labels)==set(range(102))

    def __len__(self):return len(self.raw)

    def __getitem__(self,position):
        epoch,offset=position if isinstance(position,tuple) else (0,position)
        image,label=self.raw[offset];sample=self.ids[offset]
        image=TF.resize(image,256,interpolation=InterpolationMode.BICUBIC,antialias=True)
        if self.split=='train':
            rng=np.random.default_rng(derive_seed(self.seed,epoch,sample,0))
            top=int(rng.integers(0,image.height-224+1));left=int(rng.integers(0,image.width-224+1))
            image=TF.crop(image,top,left,224,224)
            if rng.random()<.5:image=TF.hflip(image)
        else:image=TF.center_crop(image,[224,224])
        value=TF.normalize(TF.to_tensor(image),self.normalization['mean'],self.normalization['std'])
        return value,int(label),sample
