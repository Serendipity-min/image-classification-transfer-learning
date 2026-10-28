"""核验官方 Flowers 文件与 RGB 像素重复；不构造任何测试模型。"""
from collections import Counter,defaultdict
import hashlib
import tarfile
from p4_core import ROOT,STAGE,read,write,file_hash,require_scope


def main():
    require_scope()
    from scipy.io import loadmat
    from PIL import Image
    import numpy as np
    folder=ROOT/'data/flowers-102';archive=folder/'102flowers.tgz'
    files={'102flowers.tgz':'52808999861908f626f3c1f4e79d11fa','imagelabels.mat':'e0620be6f572b9609742df49c70aed4d','setid.mat':'a5357ecc9cb78c4bef273ce3793fc85c'}
    descriptors=[]
    for name,md5 in files.items():
        p=folder/name;assert hashlib.md5(p.read_bytes()).hexdigest()==md5
        descriptors.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'official_md5':md5,'sha256':file_hash(p)})
    if not (folder/'jpg').exists():
        with tarfile.open(archive) as tar:
            members=tar.getmembers()
            assert all((folder/m.name).resolve().is_relative_to(folder.resolve()) and (m.isdir() or m.isfile()) for m in members)
            tar.extractall(folder,members=members)
    labels=loadmat(folder/'imagelabels.mat',squeeze_me=True)['labels']-1
    indices=loadmat(folder/'setid.mat',squeeze_me=True);splits={};membership={}
    for name,key,size in [('train','trnid',1020),('validation','valid',1020),('test','tstid',6149)]:
        ids=indices[key].tolist();counts=Counter(int(labels[i-1]) for i in ids)
        assert len(ids)==size and len(set(ids))==size and set(counts)==set(range(102))
        if name!='test':assert set(counts.values())=={10}
        for i in ids:assert i not in membership;membership[i]=name
        splits[name]={'n':size,'sample_ids':[f'flowers102/image_{i:05d}' for i in ids],'image_ids':ids,'class_counts':dict(counts)}
    assert set(membership)==set(range(1,8190)) and labels.min()==0 and labels.max()==101
    pixels=defaultdict(list);sizes=Counter()
    for i in sorted(membership):
        p=folder/'jpg'/f'image_{i:05d}.jpg'
        with Image.open(p) as source:
            image=source.convert('RGB');array=np.asarray(image);sizes[(image.width,image.height)]+=1
            digest=hashlib.sha256(str(array.shape).encode()+array.tobytes()).hexdigest()
        pixels[digest].append({'sample_id':f'flowers102/image_{i:05d}','split':membership[i],'label':int(labels[i-1])})
    result={'schema':'P4_Flowers_integrity_v1','status':'passed','files':descriptors,'splits':splits,'labels_are_zero_based':True,
            'images_decoded':8189,'size_distribution':[{'width':w,'height':h,'n':n} for (w,h),n in sorted(sizes.items())],
            'pixel_duplicate_groups':[{'sha256':h,'samples':items} for h,items in pixels.items() if len(items)>1],
            'test_model_inference_accessed':False,'pretraining_overlap':'UNKNOWN','project_history':'no Flowers data or P4 run directory existed at preflight; this does not audit unrelated projects'}
    path=STAGE/'flowers_integrity.json'
    if path.exists():assert read(path)==result
    else:write(path,result)
    print('Flowers official integrity PASS: 1020/1020/6149, 8189 RGB decoded; no model prediction.')


if __name__=='__main__':main()
