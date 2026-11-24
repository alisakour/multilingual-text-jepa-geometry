"""Verify the archived CSVs, without access to the original embedding caches."""
from pathlib import Path
import csv
import numpy as np
R=Path(__file__).resolve().parents[1]/"results"
def read(n):
    with (R/n).open(newline='',encoding='utf-8-sig') as f: return list(csv.DictReader(f))
def close(a,b):
    assert abs(float(a)-float(b)) < 3e-5, (a,b)
def weighted(rows, key):
    return np.average([float(r[key]) for r in rows],weights=[int(r['n_examples']) for r in rows])
def grouped(prefix, mapping, extra=()):
    s=read(prefix+'_Summary.csv'); d=read(prefix+'_By_Relation.csv')
    assert len(s)==(12 if extra else 6) and len(d)==25*len(s)
    for row in s:
        group=[r for r in d if all(r[k]==row[k] for k in ('pooling','language_pair',*extra))]
        assert len(group)==25 and len({r['relation'] for r in group})==25
        assert sum(int(r['n_examples']) for r in group)==1725
        for a,b in mapping.items(): close(row[a],weighted(group,b))
    return s,d
s,d=grouped('JEPA_Objective_Comparison',{k:k for k in ['sigreg_l9_pct','sigreg_l12_pct','sigreg_shift_pp','infonce_l9_pct','infonce_l12_pct','infonce_shift_pp','sigreg_minus_infonce_shift_pp']},('metric',))
ns,nd=grouped('JEPA_Neighbor_Decomposition',{**{f'{m}_{k}_l12_minus_l9_pp':f'{m}_{k}_shift_pp' for m in ['sigreg','infonce'] for k in ['total','within','across']},**{f'objective_contrast_{k}_pp':f'objective_contrast_{k}_pp' for k in ['total','within','across']}})
ws,wd=grouped('JEPA_Within_Relation_Control',{'sigreg_example_weighted_shift_pp':'sigreg_within_only_shift_pp','infonce_example_weighted_shift_pp':'infonce_within_only_shift_pp','objective_example_weighted_contrast_pp':'sigreg_minus_infonce_shift_pp','sigreg_chance_adjusted_shift_pp':'sigreg_chance_adjusted_shift_pp','infonce_chance_adjusted_shift_pp':'infonce_chance_adjusted_shift_pp','objective_chance_adjusted_contrast_pp':'objective_chance_adjusted_contrast_pp'})
for r in ns:
    for m in ('sigreg','infonce'):
        close(r[f'{m}_total_l12_minus_l9_pp'],float(r[f'{m}_within_l12_minus_l9_pp'])+float(r[f'{m}_across_l12_minus_l9_pp']))
    close(r['objective_contrast_total_pp'],float(r['objective_contrast_within_pp'])+float(r['objective_contrast_across_pp']))
    orig=next(t for t in s if t['pooling']==r['pooling'] and t['language_pair']==r['language_pair'] and t['metric']=='global_10nn')
    close(orig['sigreg_shift_pp'],r['sigreg_total_l12_minus_l9_pp']); close(orig['infonce_shift_pp'],r['infonce_total_l12_minus_l9_pp'])
    assert float(r['sigreg_across_l12_minus_l9_pp'])>0
for r in ws:
    g=[x for x in wd if x['pooling']==r['pooling'] and x['language_pair']==r['language_pair']]
    v=np.array([float(x['sigreg_minus_infonce_shift_pp']) for x in g]); a=np.array([float(x['objective_chance_adjusted_contrast_pp']) for x in g])
    close(v.mean(),r['objective_relation_macro_contrast_pp']); close(a.mean(),r['objective_relation_macro_chance_adjusted_contrast_pp'])
    assert int(r['relations_sigreg_worse_than_infonce'])==int((v<0).sum()) and int(r['relations_tied'])==int((v==0).sum())
    assert weighted([x for x in g if x['relation'] not in ('P530','P190')],'sigreg_minus_infonce_shift_pp')<0
    for x in g:
        scale=1-10/(int(x['n_examples'])-1)
        close(float(x['sigreg_minus_infonce_shift_pp'])/scale,x['objective_chance_adjusted_contrast_pp'])
assert sum(float(x['sigreg_minus_infonce_shift_pp'])<0 for x in wd)==136
print('PASS: 12 matched, 6 decomposition and 6 restricted-pool summary rows agree with relation-level data; contrast, sign, macro and exclusion checks passed.')
print('Validation scope: archived CSV consistency.')
