"""Summarize the partition experiment and plot absolute overlap and layer shifts."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]

def main():
    d=pd.read_csv(ROOT/'results/partition/partition_contrasts.csv')
    keys=['checkpoint','pooling','language_pair']
    cols=[f'{metric}_{mode}' for metric in ('l9','l12','shift') for mode in ('shared','swapped','independent')]
    s=d.groupby(keys)[cols].mean().reset_index()
    fixed=pd.read_csv(ROOT/'results/partition/layer_shifts.csv')
    fixed=fixed[fixed.seed==-1].groupby(keys)[['l9','l12','shift']].mean().add_suffix('_fixed').reset_index()
    s=s.merge(fixed,on=keys,validate='one_to_one')
    s.to_csv(ROOT/'results/partition_summary.csv',index=False)
    for metric, ylabel, filename in [('l12','L12 10-NN overlap (%)','absolute_cls_overlap'),('shift','L12 − L9 overlap (pp)','cls_layer_shifts')]:
        data=s[s.pooling=='cls'].set_index(['checkpoint','language_pair'])
        labels=[(c,l) for c in ['infonce_01_e4','sigreg_01_e4'] for l in ['AR-EN','ZH-EN','FR-EN']]
        x=np.arange(len(labels)); fig,ax=plt.subplots(figsize=(10,4.5))
        for i,mode in enumerate(['fixed','shared','swapped','independent']):
            ax.bar(x+(i-1.5)*.19,[data.loc[k,f'{metric}_{mode}'] for k in labels],.19,label=mode.capitalize())
        ax.set_xticks(x,[('InfoNCE' if c.startswith('infonce') else 'SIGReg')+'\n'+l for c,l in labels]); ax.set_ylabel(ylabel)
        ax.axhline(0,color='0.4',linewidth=.7); ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,1.14),frameon=False)
        ax.spines[['top','right']].set_visible(False);fig.tight_layout()
        for ext in ('png','svg'): fig.savefig(ROOT/'figures'/f'{filename}.{ext}',dpi=300,bbox_inches='tight')
        plt.close(fig)
if __name__=='__main__': main()
