"""P4 的 CPU 概率指标与保持 seed×image 配对的统计。"""
import numpy as np
from scipy import stats


def calibration(logits,labels,bins):
    x=logits.astype(np.float64);maximum=x.max(1);den=maximum+np.log(np.exp(x-maximum[:,None]).sum(1))
    probability=np.exp(x-den[:,None]);prediction=x.argmax(1);correct=prediction==labels
    confidence=probability.max(1);which=np.minimum((confidence*bins).astype(int),bins-1)
    n=len(labels);reliability=[];ece=0.
    for index in range(bins):
        mask=which==index;count=int(mask.sum());conf=float(confidence[mask].mean()) if count else None;acc=float(correct[mask].mean()) if count else None
        if count:ece+=count/n*abs(conf-acc)
        reliability.append({'bin':index,'lower':index/bins,'upper':(index+1)/bins,'n':count,'confidence':conf,'accuracy':acc})
    brier=float((np.square(probability).sum(1)-2*probability[np.arange(n),labels]+1).mean())
    return {'n':n,'correct':int(correct.sum()),'top1_percent':100*float(correct.mean()),'nll':float((den-x[np.arange(n),labels]).mean()),
            'brier_no_class_division':brier,'ece':ece,'reliability':reliability}


def holm(values):
    order=np.argsort(values);result=np.zeros(len(values));previous=0.
    for rank,index in enumerate(order):
        previous=max(previous,min(1.,(len(values)-rank)*values[index]));result[index]=previous
    return result.tolist()


def paired_image_bootstrap(d,labels,rng,n=10000,seed_rng=None):
    assert d.shape[0]==5 and d.shape[1]==len(labels)
    # 每类压缩相同五维差值模式；多项抽样等价于共同图像索引的有放回抽样。
    totals=np.zeros((n,5),dtype=np.float64)
    for label in np.unique(labels):
        block=d[:,labels==label].T;patterns,counts=np.unique(block,axis=0,return_counts=True)
        weights=rng.multinomial(len(block),counts/len(block),size=n)
        totals+=weights@patterns
    if seed_rng is None:estimate=totals.mean(1)/len(labels)*100
    else:
        seed_counts=seed_rng.multinomial(5,np.ones(5)/5,size=n)
        estimate=(totals*seed_counts).sum(1)/(5*len(labels))*100
    return {'mean_difference_pp':float(d.mean()*100),'CI95_percentile_pp':np.quantile(estimate,[.025,.975]).tolist(),
            'replicates':n,'algorithm':'classwise multinomial over five-seed correctness-difference patterns; equivalent paired-image distribution',
            'class_weighting':'original image frequencies','seed_pairs_resampled':seed_rng is not None}


def cluster_bootstrap(d,clusters,rng,n=10000):
    assert sorted(i for group in clusters for i in group)==list(range(d.shape[1]))
    sums=np.array([d[:,group].sum(1) for group in clusters]);sizes=np.array([len(group) for group in clusters]);out=[]
    for _ in range(n):
        index=rng.integers(0,len(clusters),len(clusters));out.append(float(sums[index].sum()/(5*sizes[index].sum())*100))
    return {'CI95_percentile_pp':np.quantile(out,[.025,.975]).tolist(),'replicates':n,'clusters':len(clusters),
            'unit':'whole exact-pixel cluster; all member rows retained; row-count-weighted ratio'}


def primary_noninferiority(d,margin=-.5):
    d=np.asarray(d,dtype=float)
    if len(d)!=5:return {'status':'incomplete','n':len(d),'noninferiority_supported':None}
    mean=float(d.mean());sd=float(d.std(ddof=1));half=float(stats.t.ppf(.975,4))*sd/np.sqrt(5)
    return {'status':'complete','n':5,'mean_difference_pp':mean,'sample_SD_pp':sd,'CI95_pp':[mean-half,mean+half],
            'margin_pp':margin,'noninferiority_supported':bool(mean-half>margin),'family_size':1}
