"""Verify archived metrics, contrasts, assignments and provenance without a GPU."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['checkpoint', 'seed', 'condition', 'complement', 'pooling', 'language_pair']
GROUP = ['checkpoint', 'seed', 'pooling', 'language_pair']

def read(stage, name):
    return pd.read_csv(ROOT / 'results' / stage / (name + '.csv'))

def compare(actual, expected, keys, columns):
    a = actual[keys + columns].sort_values(keys).reset_index(drop=True)
    b = expected[keys + columns].sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[keys], b[keys], check_dtype=False, check_names=False)
    np.testing.assert_allclose(a[columns], b[columns], atol=1e-9, rtol=1e-10)

def verify():
    for r in pd.read_csv(ROOT / 'provenance/result_file_hashes.csv').itertuples():
        assert hashlib.sha256((ROOT / r.path).read_bytes()).hexdigest() == r.sha256, r.path
    pilot = read('pilot', 'metrics')
    pk = ['checkpoint', 'condition', 'pooling', 'language_pair']
    assert len(pilot) == 120 and not pilot.duplicated(pk + ['layer']).any()
    p = pilot[pilot.layer == 9].merge(pilot[pilot.layer == 12], on=pk, suffixes=('_l9','_l12'), validate='one_to_one')
    for c in ['knn_pct','rank_src','rank_en']:
        p[c+'_shift'] = p[c+'_l12'] - p[c+'_l9']
    compare(p, read('pilot','layer_shifts'), pk, ['knn_pct_shift','rank_src_shift','rank_en_shift'])
    cs = []
    for key, g in p.groupby(['checkpoint','pooling','language_pair']):
        q = g.set_index('condition')
        row = dict(zip(['checkpoint','pooling','language_pair'], key))
        for c in ['knn_pct_shift','rank_src_shift','rank_en_shift','knn_pct_l12']:
            f = q.loc[['fixed_a','fixed_b'],c].mean(); m = q.loc[['mixed_a','mixed_b'],c].mean()
            row.update({c+'_original':q.loc['original',c],c+'_fixed':f,c+'_mixed':m,c+'_mixed_minus_fixed':m-f})
        cs.append(row)
    cs = pd.DataFrame(cs); k = ['checkpoint','pooling','language_pair']
    assert len(cs) == 12
    compare(cs,read('pilot','diversity_contrasts'),k,[c for c in cs if c not in k])
    for stage,n in [('assignment',528),('partition',768)]:
        m = read(stage,'metrics')
        assert len(m)==n and not m.duplicated(KEYS+['layer']).any()
        assert m.knn_pct.between(0,100).all()
        w = m.pivot(index=KEYS,columns='layer',values='knn_pct').reset_index().rename(columns={9:'l9',12:'l12'})
        w['shift']=w.l12-w.l9
        compare(w,read(stage,'layer_shifts'),KEYS,['l9','l12','shift'])
        g = w[w.seed!=-1].groupby(GROUP+['condition'])[['l9','l12','shift']].mean().reset_index()
        pv = g.pivot(index=GROUP,columns='condition',values=['l9','l12','shift'])
        r = pv.index.to_frame(index=False).reset_index(drop=True)
        for c in ['l9','l12','shift']:
            for mode in ['shared','independent'] + (['swapped'] if stage=='partition' else []):
                r[c+'_'+mode]=pv[c,mode].to_numpy()
            for a,b in [('shared','independent')] + ([('shared','swapped'),('swapped','independent')] if stage=='partition' else []):
                r[c+'_'+a+'_minus_'+b]=(pv[c,a]-pv[c,b]).to_numpy()
        assert len(r)==60
        fname='partition_contrasts' if stage=='partition' else 'shared_vs_independent'
        compare(r,read(stage,fname),GROUP,[c for c in r if c not in GROUP])
        a = read(stage,'assignment_audit')
        assert (a.loc[a.condition=='shared','agreement']==1).all()
        if stage=='partition': assert (a.loc[a.condition=='swapped','agreement']==0).all()
        assert ((a.n_template_b_en*2-a.n).abs()<=1).all()
        assert ((a.n_template_b_src*2-a.n).abs()<=1).all()
        # Reconstruct pilot checks from the archived metric rows.
        q=pd.concat([m[m.seed==-1],m[(m.seed==42)&(m.condition=='shared')].assign(condition=lambda x:x.complement.map({0:'mixed_a',1:'mixed_b'}))])
        z=q.merge(pilot,on=pk+['layer'],suffixes=('_new','_pilot'),validate='one_to_one')
        assert len(z)==96 and (z.knn_pct_new-z.knn_pct_pilot).abs().max()<=0.1
    a=read('assignment','metrics'); b=read('partition','metrics'); b=b[b.condition!='swapped']
    x=a.merge(b,on=KEYS+['layer'],suffixes=('_a','_b'),validate='one_to_one')
    assert len(x)==528 and (x.knn_pct_a-x.knn_pct_b).abs().max()<=0.1
    for stage in ['pilot','assignment','partition']:
        assert (read(stage,'token_length_audit').n_over_64==0).all()
        pd.testing.assert_frame_equal(read(stage,'checkpoint_manifest'),pd.read_csv(ROOT/'provenance/checkpoint_manifest.csv'))
        pd.testing.assert_frame_equal(read(stage,'template_bank'),pd.read_csv(ROOT/'provenance/template_bank.csv'))
        for name in ['infonce_01_e4','sigreg_01_e4']:
            loading=json.loads((ROOT/'results'/stage/(name+'_loading.json')).read_text())
            assert not any(loading.get(k) for k in ['missing_keys','mismatched_keys','error_msgs'])
    keys = ['checkpoint', 'pooling', 'language_pair']
    cols = [f'{metric}_{mode}' for metric in ('l9','l12','shift') for mode in ('shared','swapped','independent')]
    summary = read('partition','partition_contrasts').groupby(keys)[cols].mean().reset_index()
    fixed = read('partition','layer_shifts')
    fixed = fixed[fixed.seed == -1].groupby(keys)[['l9','l12','shift']].mean().add_suffix('_fixed').reset_index()
    summary = summary.merge(fixed, on=keys, validate='one_to_one')
    assert len(summary) == 12
    compare(summary, pd.read_csv(ROOT/'results/partition_summary.csv'), keys, [c for c in summary if c not in keys])
    print('PASS: archived file hashes, 1,416 metric rows, layer shifts, contrasts, assignment balance, reproduction and loading checks')

if __name__=='__main__': verify()
