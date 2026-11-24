"""Generate manuscript figure from archived CSVs (Matplotlib + NumPy)."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
P=Path(__file__).resolve().parents[1]
def read(n):
    with (P/'results'/n).open(newline='') as f: return list(csv.DictReader(f))
a=read('JEPA_Neighbor_Decomposition_Summary.csv'); b=read('JEPA_Within_Relation_Control_Summary.csv')
keys=[(pool,lang) for pool in ['cls','mean'] for lang in ['AR-EN','ZH-EN','FR-EN']]
def vec(rows,key): return np.array([float(next(r[key] for r in rows if (r['pooling'],r['language_pair'])==k)) for k in keys])
y=np.arange(6)+np.array([0,0,0,.45,.45,.45]); labels=['AR–EN','ZH–EN','FR–EN']*2
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(1,2,figsize=(7.15,3.45),sharey=True)
fig.subplots_adjust(left=.17,right=.98,bottom=.18,top=.76,wspace=.24)
ax=axs[0]; wi=vec(a,'objective_contrast_within_pp'); ac=vec(a,'objective_contrast_across_pp')
ax.barh(y,wi,height=.58,color='#32658a',label='Same relation'); ax.barh(y,ac,left=wi,height=.58,color='#b9c6ce',edgecolor='#55646d',linewidth=.4,hatch='///',label='Different relation')
ax.set_title('(a) Global overlap decomposition',loc='left',fontweight='bold',pad=36)
ax.legend(loc='lower left',bbox_to_anchor=(0,1.015),frameon=False,ncol=1,fontsize=7.1,borderaxespad=0)
ax.set_xlim(-28,1.8); ax.set_xlabel('SIGReg − InfoNCE shift (pp)'); ax.xaxis.set_major_locator(MultipleLocator(5))
ax.set_yticks(y,labels); ax.invert_yaxis()
ax.text(-.45,.83,'CLS',transform=ax.transAxes,va='center',ha='left',fontweight='bold',fontsize=8); ax.text(-.45,.24,'Mean',transform=ax.transAxes,va='center',ha='left',fontweight='bold',fontsize=8)
ax=axs[1]; s=vec(b,'sigreg_example_weighted_shift_pp'); t=vec(b,'infonce_example_weighted_shift_pp')
ax.barh(y-.16,s,height=.29,color='#b75b3d',label='SIGReg'); ax.barh(y+.16,t,height=.29,color='#32658a',label='InfoNCE')
ax.set_title('(b) Same-relation candidates only',loc='left',fontweight='bold',pad=36)
ax.legend(loc='lower left',bbox_to_anchor=(0,1.015),frameon=False,ncol=1,fontsize=7.1,borderaxespad=0)
ax.set_xlim(-19,19); ax.xaxis.set_major_locator(MultipleLocator(10)); ax.set_xlabel('L12 − L9 overlap (pp)')
for ax in axs:
    ax.axvline(0,color='#333333',lw=.8); ax.axhline(2.72,color='#aaaaaa',lw=.6,ls=':'); ax.set_axisbelow(True); ax.grid(axis='x',color='#e5e5e5',lw=.55); ax.spines['left'].set_visible(False); ax.tick_params(axis='y',length=0); ax.spines['bottom'].set_color('#888888')
for ext in ['pdf','png']:
    fig.savefig(P/'figures'/f'JEPA_Relation_Neighborhoods.{ext}',dpi=350,facecolor='white',bbox_inches='tight',pad_inches=.08)
plt.close(fig)
