"""Check archived NTREX outputs and optionally compare fresh GPU results."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY = ['checkpoint', 'epoch', 'lambda_metadata', 'pooling', 'language_pair']
METRIC_KEY = KEY + ['layer']
VALUES = ['knn_pct', 'rank_src', 'rank_en']
MANIFEST = pd.read_csv(ROOT / 'provenance/checkpoint_manifest.csv')
IDS = MANIFEST.checkpoint.tolist()
EXPECTED = {(c,p,l,lang) for c in IDS for p in ['cls','mean'] for l in [9,12] for lang in ['AR-EN','ZH-EN','FR-EN']}


def compare(a, b, keys, columns=None, atol=1e-9, rtol=1e-10):
    assert not a.duplicated(keys).any() and not b.duplicated(keys).any()
    a = a.sort_values(keys).reset_index(drop=True)
    b = b.sort_values(keys).reset_index(drop=True)
    if columns is not None:
        a, b = a[keys+columns], b[keys+columns]
    pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,atol=atol,rtol=rtol)


def shifts(m):
    a = m[m.layer==9].set_index(KEY)[VALUES]
    b = m[m.layer==12].set_index(KEY)[VALUES]
    assert a.index.is_unique and b.index.is_unique and set(a.index)==set(b.index)
    s = a.join(b,lsuffix='_l9',rsuffix='_l12')
    for v in VALUES:
        s[v+'_shift'] = s[v+'_l12']-s[v+'_l9']
    return s.reset_index()


def check_run(folder, length):
    meta = json.loads((folder/'evaluation_metadata.json').read_text())
    assert meta['max_length']==length and meta['complete_checkpoints']==5 and set(meta['completed_ids'])==set(IDS)
    assert meta['k']==10 and meta['layers']==[9,12]
    m = pd.read_csv(folder/'ntrex_metrics.csv')
    assert len(m)==60 and set(zip(m.checkpoint,m.pooling,m.layer,m.language_pair))==EXPECTED
    assert (m.n_examples==1997).all() and np.isfinite(m[VALUES]).all().all()
    assert m.knn_pct.between(0,100).all() and (m[['rank_src','rank_en']]>=1).all().all()
    for row in MANIFEST.to_dict('records'):
        c=row['checkpoint']; subset=m[m.checkpoint==c]
        for k,v in row.items(): assert (subset[k]==v).all(), (c,k)
        compare(subset,pd.read_csv(folder/f'{c}_metrics.csv'),METRIC_KEY)
        loading=json.loads((folder/f'{c}_loading.json').read_text())
        assert not any(loading.get(k) for k in ('missing_keys','mismatched_keys','error_msgs'))
    s=shifts(m)
    compare(s,pd.read_csv(folder/'ntrex_layer_shifts.csv'),KEY)
    prior=shifts(pd.read_csv(ROOT/'provenance/mlama_reference.csv'))
    cross=s.merge(prior,on=KEY,suffixes=('_ntrex','_mlama'),validate='one_to_one')
    compare(cross,pd.read_csv(folder/'cross_dataset_comparisons.csv'),KEY)
    lens=pd.read_csv(folder/'token_length_audit.csv')
    assert len(lens)==20 and not lens.duplicated(['checkpoint','language']).any()
    assert set(zip(lens.checkpoint,lens.language))=={(c,l) for c in IDS for l in ['en','ar','zh','fr']}
    assert (lens.n_examples==1997).all() and lens[f'n_over_{length}'].between(0,1997).all()
    np.testing.assert_allclose(lens[f'pct_over_{length}'],100*lens[f'n_over_{length}']/1997,atol=1e-9)
    assert ((lens.max_tokens>length)==(lens[f'n_over_{length}']>0)).all()
    compare(lens,pd.concat([pd.read_csv(folder/f'{c}_lengths.csv') for c in IDS],ignore_index=True),['checkpoint','language'])
    if length==256: assert (lens.n_over_256==0).all()
    print(f'PASS: {length} tokens: 60 metric records, 30 layer shifts, 20 length records, checkpoint provenance and cross-dataset comparison')
    return m,s,meta


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reproduced',type=Path)
    args=parser.parse_args()
    archived={l:check_run(ROOT/'results'/f'max_length_{l}',l) for l in [64,256]}
    assert archived[64][2]['dataset_files']==archived[256][2]['dataset_files']
    prior=shifts(pd.read_csv(ROOT/'provenance/mlama_reference.csv'))
    cols=[v+'_shift' for v in VALUES]
    merged=archived[64][1][KEY+cols].merge(archived[256][1][KEY+cols],on=KEY,suffixes=('_64','_256'),validate='one_to_one').merge(prior[KEY+cols].rename(columns={v:v+'_mlama' for v in cols}),on=KEY,validate='one_to_one')
    compare(merged,pd.read_csv(ROOT/'results/length_control_comparison.csv'),KEY, [v+'_'+s for v in cols for s in ['64','256','mlama']])
    print('PASS: all cross-length comparisons; maximum absolute overlap-shift difference = %.4f pp'%abs(merged.knn_pct_shift_64-merged.knn_pct_shift_256).max())
    if args.reproduced:
        for length in [64,256]:
            m,s,meta=check_run(args.reproduced/f'max_length_{length}',length)
            assert meta['dataset_files']==archived[length][2]['dataset_files']
            x=m.merge(archived[length][0],on=METRIC_KEY,suffixes=('_new','_saved'),validate='one_to_one')
            np.testing.assert_allclose(x.knn_pct_new,x.knn_pct_saved,atol=0.1,rtol=0)
            for v in ['rank_src','rank_en']: np.testing.assert_allclose(x[v+'_new'],x[v+'_saved'],rtol=.005,atol=0)
            print(f'PASS: fresh {length}-token metrics match archived tolerances (0.1 pp overlap; 0.5% rank)')

if __name__=='__main__': main()
