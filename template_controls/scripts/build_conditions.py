"""Construct matched mLAMA template conditions from the original benchmark."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
LANGS=('en','ar','zh','fr')
ORIGINAL_SHA='4a8fe28fb1a9ddb73e3431a60a200d80cbf31e0f4dc69967a83469495345404f'
# Original template and two alternatives per relation and language.
BANK={
'P190':{
'en':('[X] and [Y] are twin cities .','[X] and [Y] are sister cities.','A sister-city partnership links [X] and [Y].'),
'ar':('[X] و [Y] مدينتان توأمتان.','[X] و[Y] تربطهما علاقة توأمة.','هناك توأمة بين مدينتي [X] و[Y].'),
'zh':('[X]和[Y]是双子城。','[X]与[Y]互为姐妹城市。','[X]与[Y]之间存在姐妹城市关系。'),
'fr':('[X] et [Y] sont des villes jumelles.','[X] et [Y] sont deux villes jumelées.','Un jumelage unit [X] et [Y].')},
'P47':{
'en':('[X] shares border with [Y] .','[X] and [Y] share a border.','There is a shared border between [X] and [Y].'),
'ar':('[X] تشترك في الحدود مع [Y].','[X] و[Y] لهما حدود مشتركة.','هناك حدود مشتركة بين [X] و[Y].'),
'zh':('[X]与[Y]共享边界。','[X]和[Y]有共同边界。','[X]与[Y]之间有共同的边界。'),
'fr':('[X] partage la frontière avec [Y].','[X] et [Y] ont une frontière commune.','Il existe une frontière commune entre [X] et [Y].')},
'P530':{
'en':('[X] maintains diplomatic relations with [Y] .','[X] and [Y] have diplomatic relations.','Diplomatic relations exist between [X] and [Y].'),
'ar':('[X] تقيم علاقات دبلوماسية مع [Y].','[X] و[Y] لديهما علاقات دبلوماسية.','توجد علاقات دبلوماسية بين [X] و[Y].'),
'zh':('[X]与[Y]建交。','[X]和[Y]建立了外交关系。','[X]与[Y]之间已建立外交关系。'),
'fr':('[X] entretient des relations diplomatiques avec [Y].','[X] et [Y] ont des relations diplomatiques.','Des relations diplomatiques existent entre [X] et [Y].')}
}
CONDITIONS=('original','fixed_a','fixed_b','mixed_a','mixed_b')
def render(t,x,y):
    return t.replace('[X]',x).replace('[Y]',y)
def parse(s,t,r,l,sid,oid):
    p,rest=t.split('[X]');m,z=rest.split('[Y]')
    assert s.startswith(p) and s.endswith(z),(r,l,s)
    body=s[len(p):len(s)-len(z) if z else len(s)]
    parts=body.split(m)
    if len(parts)!=2:
        # Havana (Q1563) is independently observed as 哈瓦那 in other P190 rows.
        assert (r,l,sid,oid,s)==('P190','zh','Q3935','Q1563','温得和克和哈瓦那是双子城。')
        parts=body.rsplit(m,1)
    x,y=parts
    assert x and y and render(t,x,y)==s
    return x,y

def build(source,output):
    assert hashlib.sha256(source.read_bytes()).hexdigest()==ORIGINAL_SHA
    d=pd.read_csv(source);assert len(d)==1725 and d.relation.nunique()==25
    d=d[d.relation.isin(BANK)].copy();d.insert(0,'fact_id',d.index);d=d.reset_index(drop=True)
    assert len(d)==858 and not d.duplicated(['sub_qid','obj_qid','relation']).any()
    # A single seeded assignment is shared across languages and checkpoints.
    # Reverse it in mixed_b, so every fact sees each alternative once.
    rng=np.random.default_rng(42);assign=np.zeros(len(d),dtype=int)
    for _,g in d.groupby('relation',sort=True):
        idx=rng.permutation(g.index.to_numpy());assign[idx[len(idx)//2:]]=1
    records=[]
    for i,r in d.iterrows():
        for condition in CONDITIONS:
            v={'original':0,'fixed_a':1,'fixed_b':2,'mixed_a':1+assign[i],'mixed_b':2-assign[i]}[condition]
            rec={k:r[k] for k in ['fact_id','sub_qid','obj_qid','relation','domain_category','in_size_matched_cultural_eval']};rec.update(condition=condition,template_id=int(v))
            for l in LANGS:
                x,y=parse(r[f'stmt_{l}'],BANK[r.relation][l][0],r.relation,l,r.sub_qid,r.obj_qid)
                rec[f'sub_{l}']=x;rec[f'obj_{l}']=y;rec[f'stmt_{l}']=render(BANK[r.relation][l][v],x,y)
            records.append(rec)
    result=pd.DataFrame(records);assert len(result)==4290
    for condition,g in result.groupby('condition'):
        assert g.fact_id.tolist()==d.fact_id.tolist()
        for l in LANGS:
            if g[f'stmt_{l}'].duplicated().any():
                raise ValueError(f'Duplicate text: {condition}, {l}')
    orig=result[result.condition=='original']
    for l in LANGS: assert orig[f'stmt_{l}'].tolist()==d[f'stmt_{l}'].tolist()
    for _,g in result.groupby('fact_id'):
        assert set(g[g.condition.str.startswith('mixed')].template_id)=={1,2}
        for l in LANGS:
            assert g[f'sub_{l}'].nunique()==1 and g[f'obj_{l}'].nunique()==1
    output.mkdir(parents=True,exist_ok=True)
    result.to_csv(output/'template_conditions.csv',index=False)
    pd.DataFrame([{'relation':r,'language':l,'template_id':i,'template':t} for r,ls in BANK.items() for l,ts in ls.items() for i,t in enumerate(ts)]).to_csv(output/'template_bank.csv',index=False)
    result.groupby(['relation','condition','template_id']).size().rename('n').reset_index().to_csv(output/'assignment_counts.csv',index=False)
    preview_ids=d.groupby('relation').head(3).fact_id
    result[result.fact_id.isin(preview_ids)].to_csv(output/'review_examples.csv',index=False)
    (output/'design_metadata.json').write_text(json.dumps({'source_sha256':ORIGINAL_SHA,'seed':42,'n_facts':858,'relations':d.relation.value_counts().to_dict(),'conditions':list(CONDITIONS),'semantic_review':'Arabic reviewed; independent review of English, Chinese and French not documented'},indent=2))
    print('Verified',len(result),'rows across',len(CONDITIONS),'conditions')
    return result
if __name__=='__main__':
    build(ROOT/'inputs'/'mlama_quad_parallel_benchmark.csv',ROOT/'reproduced'/'data')
