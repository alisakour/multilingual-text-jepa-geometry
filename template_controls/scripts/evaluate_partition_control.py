"""Test cross-language template assignment effects on matched mLAMA facts."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")
import gc, json, shutil
from pathlib import Path
import numpy as np
import pandas as pd
from build_conditions import build, LANGS
ROOT=Path(__file__).resolve().parents[1]
SEEDS=(42,17,101,202,303)
SELECTED=("infonce_01_e4","sigreg_01_e4")
POOLS=("cls","mean")
LAYERS=(9,12)

def balanced(relations, rng):
    a=np.zeros(len(relations),dtype=np.int8)
    for rel in sorted(set(relations)):
        idx=rng.permutation(np.flatnonzero(relations==rel))
        a[idx[len(idx)//2:]]=1
    return a

def assignments(relations,seed):
    a=balanced(relations,np.random.default_rng(seed))
    independent={"en":a}
    for i,lang in enumerate(LANGS[1:],1):
        independent[lang]=balanced(relations,np.random.default_rng(np.random.SeedSequence([seed,i])))
    return {"shared":{l:a for l in LANGS},"independent":independent,"swapped":{l:(a if l=="en" else a^1) for l in LANGS}}

def select(a,b,assignment):
    return np.where(assignment[:,None]==0,a,b)

def score(reps, relations, name, neighbors, overlap):
    rows=[]; audit=[]
    for pool in POOLS:
        for layer in LAYERS:
            key=(pool,layer)
            for variant in ("fixed_a","fixed_b"):
                nn={l:neighbors(reps[l,variant][key]) for l in LANGS}
                for l in LANGS[1:]:
                    rows.append(dict(checkpoint=name,seed=-1,condition=variant,complement=0,pooling=pool,layer=layer,language_pair=l.upper()+"-EN",knn_pct=overlap(nn[l],nn["en"])))
            for seed in SEEDS:
                for mode,bylang in assignments(relations,seed).items():
                    for complement in (0,1):
                        nn={l:neighbors(select(reps[l,"fixed_a"][key],reps[l,"fixed_b"][key],bylang[l]^complement)) for l in LANGS}
                        for l in LANGS[1:]:
                            rows.append(dict(checkpoint=name,seed=seed,condition=mode,complement=complement,pooling=pool,layer=layer,language_pair=l.upper()+"-EN",knn_pct=overlap(nn[l],nn["en"])))
    for seed in SEEDS:
        for mode,bylang in assignments(relations,seed).items():
            for l in LANGS[1:]:
                for rel in sorted(set(relations)):
                    mask=relations==rel
                    audit.append(dict(checkpoint=name,seed=seed,condition=mode,language=l,relation=rel,n=int(mask.sum()),n_template_b_en=int(bylang["en"][mask].sum()),n_template_b_src=int(bylang[l][mask].sum()),agreement=float((bylang[l][mask]==bylang["en"][mask]).mean())))
    return pd.DataFrame(rows),pd.DataFrame(audit)

def summarize(m):
    keys=["checkpoint","seed","condition","complement","pooling","language_pair"]
    wide=m.pivot(index=keys,columns="layer",values="knn_pct").reset_index().rename(columns={9:"l9",12:"l12"})
    wide["shift"]=wide.l12-wide.l9
    base=wide[wide.seed==-1].groupby(["checkpoint","pooling","language_pair"])[["l9","l12","shift"]].mean().add_suffix("_fixed")
    mix=wide[wide.seed!=-1].groupby(["checkpoint","seed","condition","pooling","language_pair"])[["l9","l12","shift"]].mean().reset_index()
    mix=mix.merge(base.reset_index(),on=["checkpoint","pooling","language_pair"],validate="many_to_one")
    for col in ("l9","l12","shift"):
        mix[col+"_minus_fixed"]=mix[col]-mix[col+"_fixed"]
    a=mix[mix.condition=="shared"].merge(mix[mix.condition=="independent"],on=["checkpoint","seed","pooling","language_pair"],suffixes=("_shared","_independent"),validate="one_to_one")
    for col in ("l9","l12","shift"):
        a[col+"_shared_minus_independent"]=a[col+"_shared"]-a[col+"_independent"]
    return wide,mix,a

def verify_reference(m):
    fixed=m[m.seed==-1].copy(); shared=m[(m.seed==42)&(m.condition=="shared")].copy()
    shared["condition"]=shared.complement.map({0:"mixed_a",1:"mixed_b"})
    check=pd.concat([fixed,shared],ignore_index=True)
    keys=["checkpoint","condition","pooling","layer","language_pair"]
    ref=pd.read_csv(ROOT/"results"/"pilot"/"metrics.csv")
    x=check.merge(ref[keys+["knn_pct"]],on=keys,suffixes=("_new","_pilot"),validate="one_to_one")
    assert len(x)==len(check)
    x["error_pp"]=x.knn_pct_new-x.knn_pct_pilot
    x["passes"]=x.error_pp.abs()<=0.1
    return x

def run():
    import torch
    from transformers import AutoModel,AutoTokenizer
    from huggingface_hub import snapshot_download
    from geometry_reference import extract_language,neighbors,overlap,sha256,atomic_csv
    if not torch.cuda.is_available(): raise RuntimeError("CUDA GPU required")
    torch.set_num_threads(2)
    data=build(ROOT/"inputs"/"mlama_quad_parallel_benchmark.csv",ROOT/"reproduced"/"data")
    tables={v:data[data.condition==v].reset_index(drop=True) for v in ("fixed_a","fixed_b")}
    assert tables["fixed_a"].fact_id.equals(tables["fixed_b"].fact_id)
    rel=tables["fixed_a"].relation.to_numpy()
    # Confirm seed 42 and complementary assignment match the original pilot.
    assert np.array_equal(assignments(rel,42)["shared"]["en"],data[data.condition=="mixed_a"].template_id.to_numpy()-1)
    manifest=pd.read_csv(ROOT/"provenance"/"checkpoint_manifest.csv").set_index("checkpoint").loc[list(SELECTED)].reset_index()
    out=ROOT/"reproduced"/"partition";out.mkdir(parents=True,exist_ok=True)
    frames=[]; audits=[]; lengths=[]
    for spec in manifest.to_dict("records"):
        name=spec["checkpoint"]; print("Extracting",name,flush=True)
        snap=Path(snapshot_download(repo_id=spec["repo"],revision=spec["revision"],allow_patterns=["*.json","*.model","model.safetensors"]))
        assert sha256(snap/"model.safetensors")==spec["weights_sha256"]
        cfg=json.loads((snap/"dual_objective_config.json").read_text())
        assert float(cfg["lambda_alignment"])==spec["lambda_metadata"] and cfg["alignment_loss_type"]==name.split("_")[0]
        tok=AutoTokenizer.from_pretrained(snap,local_files_only=True,trust_remote_code=False)
        model,loading=AutoModel.from_pretrained(snap,local_files_only=True,trust_remote_code=False,use_safetensors=True,dtype=torch.float32,output_loading_info=True,add_pooling_layer=False)
        (out/(name+"_loading.json")).write_text(json.dumps(loading,default=str,indent=2))
        assert not any(loading.get(k) for k in ("missing_keys","mismatched_keys","error_msgs")) and model.config.num_hidden_layers==12
        model=model.to("cuda").eval(); reps={}
        try:
            for variant,g in tables.items():
                for l in LANGS:
                    texts=g[f"stmt_{l}"].tolist()
                    lens=np.array([len(x) for x in tok(texts,add_special_tokens=True,truncation=False)["input_ids"]])
                    lengths.append(dict(checkpoint=name,condition=variant,language=l,n=len(lens),max=int(lens.max()),n_over_64=int((lens>64).sum())))
                    assert lens.max()<=64
                    reps[l,variant]=extract_language(model,tok,texts)
        finally:
            del model,tok;gc.collect();torch.cuda.empty_cache()
        print("Scoring cached representations",name,flush=True)
        m,a=score(reps,rel,name,neighbors,overlap);del reps
        frames.append(m);audits.append(a)
        allm=pd.concat(frames,ignore_index=True)
        assert not allm.duplicated(["checkpoint","seed","condition","complement","pooling","layer","language_pair"]).any()
        verify=verify_reference(allm);atomic_csv(verify,out/"pilot_reproduction.csv")
        atomic_csv(allm,out/"metrics.csv")
        if not verify.passes.all(): raise RuntimeError("Pilot overlap reproduction exceeded 0.1 pp; inspect output before interpreting")
        wide,mix,contrast=summarize(allm)
        atomic_csv(wide,out/"layer_shifts.csv");atomic_csv(mix,out/"mixed_vs_fixed.csv");atomic_csv(contrast,out/"shared_vs_independent.csv")
        atomic_csv(pd.concat(audits,ignore_index=True),out/"assignment_audit.csv")
        atomic_csv(pd.DataFrame(lengths),out/"token_length_audit.csv")
        print("Completed",name,flush=True)
    assert len(allm)==768 and len(contrast)==60 and len(verify)==96
    keys=["checkpoint","seed","condition","complement","pooling","layer","language_pair"]
    ref=pd.read_csv(ROOT/"results"/"assignment"/"metrics.csv")
    observed=allm[allm.condition!="swapped"]
    checked=observed.merge(ref,on=keys,suffixes=("_new","_previous"),validate="one_to_one")
    assert len(checked)==len(ref)==len(observed)==528
    checked["error_pp"]=checked.knn_pct_new-checked.knn_pct_previous
    checked["passes"]=checked.error_pp.abs()<=0.1
    atomic_csv(checked,out/"assignment_reproduction.csv")
    if not checked.passes.all(): raise RuntimeError("Assignment reproduction exceeded 0.1 pp")
    ck=["checkpoint","seed","pooling","language_pair"]
    columns=["l9","l12","shift"]
    piv=mix.pivot(index=ck,columns="condition",values=columns)
    result=piv.index.to_frame(index=False).reset_index(drop=True)
    for col in columns:
        for mode in ("shared","swapped","independent"):
            result[col+"_"+mode]=piv[col,mode].to_numpy()
        for a,b in (("shared","swapped"),("swapped","independent"),("shared","independent")):
            result[col+"_"+a+"_minus_"+b]=(piv[col,a]-piv[col,b]).to_numpy()
    assert len(result)==60
    atomic_csv(result,out/"partition_contrasts.csv")
    audit=pd.concat(audits,ignore_index=True)
    assert (audit.loc[audit.condition=="shared","agreement"]==1).all()
    assert (audit.loc[audit.condition=="swapped","agreement"]==0).all()
    atomic_csv(manifest,out/"checkpoint_manifest.csv")
    shutil.copy2(ROOT/"reproduced"/"data/template_bank.csv",out/"template_bank.csv")
    (out/"protocol.json").write_text(json.dumps(dict(seeds=SEEDS,checkpoints=SELECTED,n_facts=len(rel),benchmark_sha256=sha256(ROOT/"inputs"/"mlama_quad_parallel_benchmark.csv"),note="Assignment seeds, not model-training seeds. Overlap only. See provenance/template_review.md for language review status. Swapped assignments retain the shared binary partition with opposite template IDs in source languages."),indent=2))
    shutil.make_archive(str(ROOT/"reproduced"/"jepa_template_partition_results"),"zip",out)
    print("Completed all partition controls",flush=True)
if __name__=="__main__":run()
