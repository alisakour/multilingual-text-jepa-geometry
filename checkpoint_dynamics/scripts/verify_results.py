"""Validate archived checkpoint metrics and derived summaries without model downloads."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
m = pd.read_csv(ROOT / 'results/checkpoint_metrics.csv')
s = pd.read_csv(ROOT / 'results/layer_shifts.csv')
manifest = pd.read_csv(ROOT / 'provenance/checkpoint_manifest.csv')
keys = ['checkpoint', 'pooling', 'language_pair', 'layer']
idx = ['checkpoint', 'epoch', 'recorded_steps', 'lambda_metadata', 'pooling', 'language_pair']
assert len(m) == 216 and len(s) == 108 and len(manifest) == 18
assert not m.duplicated(keys).any() and not s.duplicated(idx).any()
assert manifest.checkpoint.is_unique and manifest.weights_sha256.is_unique
expected = {(p, l, k) for p in ('cls', 'mean') for l in (9, 12) for k in ('AR-EN', 'FR-EN', 'ZH-EN')}
assert set(m.checkpoint) == set(manifest.checkpoint)
for name, g in m.groupby('checkpoint'):
    assert len(g) == 12 and g.n_examples.eq(1725).all()
    assert set(g[['pooling', 'layer', 'language_pair']].itertuples(index=False, name=None)) == expected
    spec = manifest.set_index('checkpoint').loc[name]
    for field in spec.index:
        assert g[field].eq(spec[field]).all(), (name, field)
assert np.isfinite(m[['knn_pct', 'rank_src', 'rank_en']]).all().all()
wide = m.pivot(index=idx, columns='layer', values=['knn_pct', 'rank_src', 'rank_en'])
rebuilt = pd.DataFrame(index=wide.index)
for metric in ('knn_pct', 'rank_src', 'rank_en'):
    for l in (9, 12):
        rebuilt[f'{metric}_l{l}'] = wide[metric, l]
    rebuilt[f'{metric}_shift'] = wide[metric, 12] - wide[metric, 9]
np.testing.assert_allclose(rebuilt.sort_index(), s.set_index(idx).sort_index()[rebuilt.columns], atol=1e-10, rtol=0)
s['run'] = s.checkpoint.str.rsplit('_e', n=1).str[0]
summary_idx = ['run', 'epoch', 'recorded_steps', 'pooling']
summary = pd.read_csv(ROOT / 'results/trajectory_summary.csv').set_index(summary_idx).sort_index()
np.testing.assert_allclose(summary, s.groupby(summary_idx)[summary.columns.tolist()].mean().sort_index(), atol=1e-10, rtol=0)
join = s[s.run.eq('infonce_01')].merge(s[s.run.eq('sigreg_01')], on=['epoch', 'pooling', 'language_pair'], suffixes=('_infonce','_sigreg'), validate='one_to_one')
comp = pd.read_csv(ROOT / 'results/epoch_matched_comparisons.csv').set_index(['epoch','pooling','language_pair']).sort_index()
join = join.set_index(['epoch','pooling','language_pair']).sort_index()
for metric in ('knn_pct_shift','rank_src_shift','rank_en_shift'):
    for run in ('infonce','sigreg'):
        np.testing.assert_allclose(comp[f'{metric}_{run}'], join[f'{metric}_{run}'], atol=1e-10, rtol=0)
    np.testing.assert_allclose(comp[f'{metric}_contrast'], join[f'{metric}_infonce'] - join[f'{metric}_sigreg'], atol=1e-10, rtol=0)
repro = pd.read_csv(ROOT / 'provenance/reproduction_check.csv')
assert len(repro) == 24 and repro.passes.eq(True).all()
ali = pd.read_csv(ROOT / 'provenance/infonce_lambda1_metrics.csv')
shared = m.merge(ali, on=keys, suffixes=('_combined','_ali'), validate='one_to_one')
assert len(shared) == 48
for k in ('revision','weights_sha256'):
    assert shared[f'{k}_combined'].eq(shared[f'{k}_ali']).all()
assert (shared.knn_pct_combined-shared.knn_pct_ali).abs().max() <= .1
for k in ('rank_src','rank_en'):
    assert ((shared[k+'_combined']-shared[k+'_ali']).abs()/shared[k+'_combined']).max() <= .005
print('18 checkpoints, 216 metric rows, 108 layer shifts, 30 epoch-matched comparisons: verified.')
print('Original reproduction checks: 24/24 passed. Shared InfoNCE results: consistent.')
