"""冻结测试证据作图，保留所有五 seed 和所有预算前缀。"""
import argparse
import csv
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figures/P3C'
METHODS = ['scratch','full_ft','linear_probe','adapter']
LABELS = ['Scratch','Full FT','Linear Probe','Adapter']
COLORS = ['#777777','#356B9A','#C17C3A','#408579']
GROUPS = ['cifar100-r50','cifar100-vitb16','svhn-r50','svhn-vitb16']
TITLES = ['CIFAR-100 / R50 (IN1K)','CIFAR-100 / ViT (IN21K)','SVHN / R50 (IN1K)','SVHN / ViT (IN21K)']
SEEDS = [17,42,2026,3407,65537]


def load(name):
    with (ROOT/'reports'/name).open(encoding='utf8') as file:
        return list(csv.DictReader(file))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--qa-scripts',type=Path,required=True);args=parser.parse_args()
    sys.path.insert(0,str(args.qa_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['DejaVu Sans'],
        'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
        'legend.fontsize':8,'legend.frameon':False,'axes.spines.right':False,'axes.spines.top':False,
        'svg.fonttype':'none','pdf.fonttype':42,'axes.linewidth':.7,'lines.linewidth':1.2})
    rows=load('P3_C_逐角色测试指标.csv')

    def save(fig,name):
        fig.canvas.draw()
        require_matplotlib_panel_alignment(fig,json_out=OUT/(name+'.alignment.json'),
            tolerance_pt=1.5,gutter_tolerance_pt=1.5,strict=True)
        fig.savefig(OUT/(name+'.pdf'),facecolor='white')
        fig.savefig(OUT/(name+'.svg'),facecolor='white')
        svg=OUT/(name+'.svg')
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8',newline='\n')
        fig.savefig(OUT/(name+'.png'),dpi=600,facecolor='white')
        plt.close(fig)

    def grid():
        fig,axes=plt.subplots(2,2,figsize=(7.1,5.8))
        fig.subplots_adjust(left=.10,right=.98,bottom=.14,top=.89,wspace=.32,hspace=.51)
        for i,ax in enumerate(axes.flat):
            ax.set_title(chr(97+i)+'  '+TITLES[i],loc='left',pad=11)
            ax.grid(axis='y',alpha=.13,linewidth=.5);ax.set_ylim(0,100)
            ax.set_ylabel('Official test Top-1 (%)')
        return fig,list(axes.flat)

    fig,axes=grid()
    for ax,group in zip(axes,GROUPS):
        for i,(method,color) in enumerate(zip(METHODS,COLORS)):
            subset=[r for r in rows if r['configuration_id']==group+'-'+method and r['role']=='primary_best_val']
            values=np.array([float(next(r['top1_percent'] for r in subset if int(r['seed'])==s)) for s in SEEDS])
            ax.scatter(i+np.linspace(-.12,.12,5),values,color=color,s=15,zorder=3,alpha=.85)
            ax.errorbar(i,values.mean(),yerr=values.std(ddof=1),fmt='_',markersize=16,color='black',capsize=3,linewidth=.9,zorder=4)
        ax.set_xticks(range(4),LABELS,rotation=18,ha='right',rotation_mode='anchor');ax.set_xlim(-.5,3.5)
    fig.text(.5,.045,'All five seeds; black marks = mean ± sample SD. Validation-selected checkpoints.',ha='center',fontsize=8)
    save(fig,'primary_test_results')

    differences=load('P3_C_主比较差值.csv');tests=load('P3_C_逐种子配对统计.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.8))
    fig.subplots_adjust(left=.13,right=.98,bottom=.25,top=.82,wspace=.48)
    for i,(ax,test) in enumerate(zip(axes,tests)):
        values=np.array([float(next(r['difference_pp'] for r in differences if r['comparison']==test['comparison'] and int(r['seed'])==s)) for s in SEEDS])
        ax.axvline(0,color='#888888',linewidth=.8,linestyle='--')
        if i==0:ax.axvline(-.5,color='#C17C3A',linewidth=.8,linestyle=':')
        ax.scatter(values,np.arange(5),s=22,color=COLORS[2 if i==0 else 3],zorder=3)
        mean=float(test['mean_difference_pp']);lo=float(test['CI95_low_pp']);hi=float(test['CI95_high_pp'])
        ax.errorbar(mean,5.4,xerr=np.array([[mean-lo],[hi-mean]]),fmt='D',color='black',markersize=4,capsize=3)
        ax.set_yticks([0,1,2,3,4,5.4],[str(s) for s in SEEDS]+['Mean / 95% CI'])
        ax.set_ylim(6.1,-.7);ax.set_xlim((-6.5,.5) if i==0 else (-.15,.75))
        ax.set_xlabel(('LP − Full FT' if i==0 else 'Adapter − Full FT')+' (pp)')
        ax.set_title(chr(97+i)+'  '+TITLES[1 if i==0 else 2],loc='left',pad=12)
    fig.text(.5,.115,'Paired seeds (n = 5); two-sided t 95% CI. Dashed line: zero; dotted line: −0.5 pp margin.',ha='center',fontsize=7)
    fig.text(.5,.055,'Holm-adjusted p: 1.24 × 10⁻⁶ (LP below FT); 0.0227 (Adapter above FT).',ha='center',fontsize=8)
    save(fig,'preregistered_paired_contrasts')

    fig,axes=grid()
    for ax,group in zip(axes,GROUPS):
        for method,label,color in zip(METHODS,LABELS,COLORS):
            data=np.array([[float(next(r['top1_percent'] for r in rows if r['configuration_id']==group+'-'+method and r['role']=='budget_prefix' and int(r['seed'])==s and float(r['budget_fraction'])==fraction)) for fraction in [.25,.5,1.]] for s in SEEDS])
            x=np.array([.25,.5,1.]);mean=data.mean(0);sd=data.std(0,ddof=1)
            ax.errorbar(x,mean,yerr=sd,color=color,label=label,marker='o',markersize=3,capsize=2,linewidth=1)
        ax.set_xticks([.25,.5,1.],['0.25B','0.5B','B']);ax.set_xlim(.18,1.07)
        ax.set_xlabel('Frozen common training-time budget')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',ncol=4,bbox_to_anchor=(.5,.995))
    fig.text(.5,.052,'Mean ± sample SD, five seeds; all 240 prefixes shown in aggregates. Exploratory.',ha='center',fontsize=8)
    fig.text(.5,.025,'Repeated checkpoints from the same trajectories; prefixes are not independent experiments.',ha='center',fontsize=8)
    save(fig,'fixed_budget_test_prefixes')


if __name__=='__main__':
    main()
