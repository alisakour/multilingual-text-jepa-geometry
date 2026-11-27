"""Evaluate fixed and mixed template conditions with pinned mLAMA encoders."""
import gc, hashlib, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download
from geometry_reference import extract_language, neighbors, overlap, effective_rank, sha256, atomic_csv
from build_conditions import build, CONDITIONS, LANGS
ROOT=Path(__file__).resolve().parents[1]
SELECTED=['infonce_01_e4','sigreg_01_e4']

def summarize(m):
    keys=['checkpoint','condition','pooling','language_pair']
    a=m[m.layer==9].merge(m[m.layer==12],on=keys,suffixes=('_l9','_l12'),validate='one_to_one')
    for metric in ['knn_pct','rank_src','rank_en']:
        a[metric+'_shift']=a[metric+'_l12']-a[metric+'_l9']
    contrast=[]
    for key,g in a.groupby(['checkpoint','pooling','language_pair']):
        q=g.set_index('condition')
        if set(q.index)!=set(CONDITIONS): continue
        row=dict(zip(['checkpoint','pooling','language_pair'],key))
        for metric in ['knn_pct_shift','rank_src_shift','rank_en_shift','knn_pct_l12']:
            fixed=q.loc[['fixed_a','fixed_b'],metric].mean();mixed=q.loc[['mixed_a','mixed_b'],metric].mean()
            row.update({metric+'_original':q.loc['original',metric],metric+'_fixed':fixed,metric+'_mixed':mixed,metric+'_mixed_minus_fixed':mixed-fixed})
        contrast.append(row)
    return a,pd.DataFrame(contrast)

def run():
    if not torch.cuda.is_available(): raise RuntimeError('Select a GPU runtime in Colab.')
    d=build(ROOT/'inputs'/'mlama_quad_parallel_benchmark.csv',ROOT/'reproduced'/'data')
    manifest=pd.read_csv(ROOT/'provenance'/'checkpoint_manifest.csv').set_index('checkpoint').loc[SELECTED].reset_index()
    out=ROOT/'reproduced'/'pilot';out.mkdir(parents=True,exist_ok=True)
    frames=[];audits=[]
    for spec in manifest.to_dict('records'):
        name=spec['checkpoint'];print('Evaluating',name,flush=True)
        snapshot=Path(snapshot_download(repo_id=spec['repo'],revision=spec['revision'],allow_patterns=['*.json','*.model','model.safetensors']))
        assert sha256(snapshot/'model.safetensors')==spec['weights_sha256']
        cfg=json.loads((snapshot/'dual_objective_config.json').read_text())
        assert float(cfg['lambda_alignment'])==spec['lambda_metadata'] and cfg['alignment_loss_type']==name.split('_')[0]
        tok=AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
        # Audit all conditions before extraction; preserve the original 64-token protocol.
        for condition in CONDITIONS:
            g=d[d.condition==condition]
            for lang in LANGS:
                lengths=np.array([len(x) for x in tok(g[f'stmt_{lang}'].tolist(),add_special_tokens=True,truncation=False)['input_ids']])
                audits.append(dict(checkpoint=name,condition=condition,language=lang,n=len(lengths),mean=float(lengths.mean()),max=int(lengths.max()),n_over_64=int((lengths>64).sum())))
        atomic_csv(pd.DataFrame(audits),out/'token_length_audit.csv')
        if any(x['n_over_64'] for x in audits): raise RuntimeError('Token truncation detected; inspect audit before changing protocol.')
        model,loading=AutoModel.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.float32,output_loading_info=True,add_pooling_layer=False)
        (out/(name+'_loading.json')).write_text(json.dumps(loading,default=str,indent=2))
        assert not any(loading.get(k) for k in ('missing_keys','mismatched_keys','error_msgs')) and model.config.num_hidden_layers==12
        model=model.to('cuda').eval()
        rows=[]
        try:
            for condition in CONDITIONS:
                g=d[d.condition==condition];print(' ',condition,flush=True)
                reps={lang:extract_language(model,tok,g[f'stmt_{lang}'].tolist()) for lang in LANGS}
                for pool in ('cls','mean'):
                    for layer in (9,12):
                        en=reps['en'][pool,layer];nn=neighbors(en);er=effective_rank(en)
                        for lang in LANGS[1:]:
                            src=reps[lang][pool,layer]
                            rows.append(dict(**spec,condition=condition,pooling=pool,layer=layer,language_pair=lang.upper()+'-EN',n_examples=len(g),knn_pct=overlap(neighbors(src),nn),rank_src=effective_rank(src),rank_en=er))
                del reps
        finally:
            del model,tok;gc.collect();torch.cuda.empty_cache()
        frame=pd.DataFrame(rows);assert len(frame)==60 and not frame.duplicated(['checkpoint','condition','pooling','layer','language_pair']).any()
        frames.append(frame);metrics=pd.concat(frames,ignore_index=True);atomic_csv(metrics,out/'metrics.csv')
        shifts,contrasts=summarize(metrics);atomic_csv(shifts,out/'layer_shifts.csv');atomic_csv(contrasts,out/'diversity_contrasts.csv')
    assert len(metrics)==120 and len(shifts)==60 and len(contrasts)==12
    for f in ['template_bank.csv','assignment_counts.csv','design_metadata.json']:
        shutil.copy2(ROOT/'reproduced'/'data'/f,out/f)
    atomic_csv(manifest,out/'checkpoint_manifest.csv')
    shutil.make_archive(str(ROOT/'reproduced'/'jepa_template_pilot_results'),'zip',out)
    print('Completed 120 metrics, 60 layer shifts, 12 diversity contrasts.',flush=True)
if __name__=='__main__':run()
