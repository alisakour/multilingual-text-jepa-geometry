"""Plot equal-weight language averages for the five evaluated checkpoints."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
d=pd.read_csv(ROOT/'results/length_control_comparison.csv')
ids=['infonce_01_e1','sigreg_01_e1','infonce_01_e4','sigreg_01_e4','infonce_1_e4']
labels=['InfoNCE 0.1\nE1','SIGReg 0.1\nE1','InfoNCE 0.1\nE4','SIGReg 0.1\nE4','InfoNCE 1.0\nE4']
fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained',sharex=True)
for i,pool in enumerate(['cls','mean']):
    s=d[d.pooling==pool].groupby('checkpoint').mean(numeric_only=True).reindex(ids)
    for j,(metric,ylabel) in enumerate([('knn_pct','Neighbor overlap shift (pp)'),('rank_src','Source effective-rank shift')]):
        ax=axs[i,j]
        for k,(suffix,label) in enumerate([('mlama','mLAMA'),('64','NTREX (64)'),('256','NTREX (256)')]):
            ax.bar(np.arange(len(ids))+(k-1)*.24,s[f'{metric}_shift_{suffix}'],width=.23,label=label)
        ax.axhline(0,color='0.35',linewidth=.8)
        ax.set_title(('CLS' if pool=='cls' else 'Mean pooling')+' | '+('Overlap' if j==0 else 'Effective rank'))
        ax.set_ylabel(ylabel); ax.set_xticks(range(len(ids)),labels); ax.grid(axis='y',alpha=.2); ax.set_axisbelow(True)
axs[0,0].legend(frameon=False,fontsize=9)
fig.suptitle('L12 minus L9: mLAMA and NTREX-128',fontsize=15)
for ext in ['png','svg']: fig.savefig(ROOT/'figures'/f'cross_dataset_shifts.{ext}',dpi=180)
