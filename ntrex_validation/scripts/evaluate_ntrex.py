"""Evaluate revision-pinned multilingual encoder checkpoints on the NTREX-128 parallel news benchmark."""
import argparse
import gc
import hashlib
import json
import urllib.request
import zipfile
import platform
import transformers
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
LANGS = ("en", "ar", "zh", "fr")
LAYERS = (9, 12)
K, BATCH_SIZE, MAX_LENGTH = 10, 32, 64

KEYS = ["checkpoint", "pooling", "language_pair", "layer"]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def atomic_csv(frame, path):
    tmp = path.with_suffix(".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)

def neighbors(z, k=K):
    z = np.array(z, dtype=np.float64, copy=True)
    z -= z.mean(axis=0, keepdims=True)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)[:, :k]

def overlap(a, b):
    return float(100 * (a[:, :, None] == b[:, None, :]).any(axis=2).mean())

def effective_rank(z):
    z = np.array(z, dtype=np.float64, copy=True)
    z -= z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(z, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))

@torch.inference_mode()
def extract_language(model, tokenizer, texts):
    arrays = {(pool, layer): [] for pool in ("cls", "mean") for layer in LAYERS}
    for i in range(0, len(texts), BATCH_SIZE):
        enc = tokenizer(texts[i:i+BATCH_SIZE], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt").to("cuda")
        states = model(**enc, output_hidden_states=True).hidden_states
        mask = enc.attention_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1e-9)
        for layer in LAYERS:
            hs = states[layer].float()
            arrays["cls", layer].append(hs[:, 0, :].cpu().numpy())
            arrays["mean", layer].append(((hs * mask).sum(dim=1) / denom).cpu().numpy())
    return {key: np.concatenate(values, axis=0).astype(np.float32) for key, values in arrays.items()}

def evaluate_checkpoint(spec, df, out):
    print("Evaluating", spec["checkpoint"], flush=True)
    snapshot = Path(snapshot_download(repo_id=spec["repo"], revision=spec["revision"], allow_patterns=["*.json", "*.model", "model.safetensors"]))
    observed_hash = sha256(snapshot / "model.safetensors")
    if observed_hash != spec["weights_sha256"]:
        raise RuntimeError("Unexpected model weight hash")
    config = json.loads((snapshot / "dual_objective_config.json").read_text())
    if float(config["lambda_alignment"]) != spec["lambda_metadata"] or config["alignment_loss_type"] != ("infonce" if spec["checkpoint"].startswith("infonce") else "sigreg"):
        raise RuntimeError("Unexpected objective metadata")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    model, loading = AutoModel.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, use_safetensors=True, dtype=torch.float32, output_loading_info=True, add_pooling_layer=False)
    (out / (spec["checkpoint"] + "_loading.json")).write_text(json.dumps(loading, default=str, indent=2))
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError("Unexpected missing or mismatched encoder weights: " + str(loading))
    model = model.to("cuda").eval()
    if model.config.num_hidden_layers != 12:
        raise RuntimeError("Expected a 12-layer encoder")
    representations = {}
    lengths = []
    try:
        for lang in LANGS:
            texts = df[f"stmt_{lang}"].tolist()
            lens = [len(ids) for start in range(0, len(texts), 64) for ids in tokenizer(texts[start:start+64], truncation=False, add_special_tokens=True, verbose=False)["input_ids"]]
            lengths.append(dict(checkpoint=spec["checkpoint"], language=lang, n_examples=len(lens), **{f"n_over_{MAX_LENGTH}": sum(n > MAX_LENGTH for n in lens), f"pct_over_{MAX_LENGTH}": 100 * np.mean(np.array(lens) > MAX_LENGTH)}, max_tokens=max(lens), mean_tokens=float(np.mean(lens))))
            representations[lang] = extract_language(model, tokenizer, texts)
            print("  Extracted", lang, flush=True)
    finally:
        del model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    rows = []
    for pool in ("cls", "mean"):
        for layer in LAYERS:
            key = (pool, layer)
            en = representations["en"][key]
            en_nn, en_rank = neighbors(en), effective_rank(en)
            for lang in LANGS[1:]:
                src = representations[lang][key]
                rows.append(dict(**spec, pooling=pool, layer=layer, language_pair=lang.upper()+"-EN", n_examples=len(df), knn_pct=overlap(neighbors(src), en_nn), rank_src=effective_rank(src), rank_en=en_rank))
    result = pd.DataFrame(rows)
    atomic_csv(pd.DataFrame(lengths), out / (spec["checkpoint"] + "_lengths.csv"))
    atomic_csv(result, out / (spec["checkpoint"] + "_metrics.csv"))
    return result


DATA_REV = '468c6b69c7f6a75d31d4743d9daba2af566cc18d'
DATA_FILES = {
    'en': ('newstest2019-src.eng.txt', 'ebc28ab1a5b3561d7e6fee2173441a7d2bfa1edd'),
    'ar': ('newstest2019-ref.arb.txt', 'bfe5e9fd9f1f89363ff5320a83a487cddf8412e5'),
    'zh': ('newstest2019-ref.zho-CN.txt', '20b794983bd59c10adb16714f9a7d880e8a8fba1'),
    'fr': ('newstest2019-ref.fra.txt', '74d0fa25e19f64580ef1d75d6188bdcd354c5b37'),
}

def dataset():
    columns, hashes = {}, {}
    for lang, (filename, expected) in DATA_FILES.items():
        url = f'https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/{DATA_REV}/NTREX-128/{filename}'
        with urllib.request.urlopen(url, timeout=90) as response:
            raw = response.read()
        blob = hashlib.sha1(f'blob {len(raw)}\0'.encode() + raw).hexdigest()
        if blob != expected:
            raise ValueError(f'NTREX source hash mismatch: {filename}')
        texts = raw.decode('utf-8-sig').splitlines()
        if len(texts) != 1997 or any(not t.strip() for t in texts):
            raise ValueError(f'Unexpected line count or empty sentence: {filename}: {len(texts)}')
        columns['stmt_' + lang] = texts
        hashes[lang] = {'file': filename, 'git_blob_sha1': blob, 'sha256': hashlib.sha256(raw).hexdigest(), 'n_examples': len(texts), 'unique_texts': len(set(texts))}
    return pd.DataFrame(columns), hashes

def shifts(metrics):
    index = ['checkpoint','epoch','lambda_metadata','pooling','language_pair']
    l9 = metrics[metrics.layer == 9].set_index(index)
    l12 = metrics[metrics.layer == 12].set_index(index)
    assert l9.index.is_unique and l12.index.is_unique and set(l9.index) == set(l12.index)
    result = l9[['knn_pct','rank_src','rank_en']].join(l12[['knn_pct','rank_src','rank_en']], lsuffix='_l9', rsuffix='_l12')
    for key in ('knn_pct','rank_src','rank_en'):
        result[key+'_shift'] = result[key+'_l12'] - result[key+'_l9']
    return result.reset_index()

def validate_metric(path, spec):
    lengths_path = path.with_name(spec['checkpoint']+'_lengths.csv')
    if not path.exists() or not lengths_path.exists():
        return None
    frame = pd.read_csv(path)
    expected = {(p,l,lang.upper()+'-EN') for p in ('cls','mean') for l in LAYERS for lang in LANGS[1:]}
    if len(frame) != 12 or set(zip(frame.pooling,frame.layer,frame.language_pair)) != expected or not (frame.n_examples == 1997).all():
        raise ValueError(f'Invalid saved metric rows: {path}')
    for key in ('checkpoint','repo','revision','weights_sha256','epoch','lambda_metadata','recorded_steps'):
        if not (frame[key] == spec[key]).all():
            raise ValueError(f'Invalid saved metadata: {key}')
    if not np.isfinite(frame[['knn_pct','rank_src','rank_en']].to_numpy()).all():
        raise ValueError('Nonfinite saved metrics')
    lengths = pd.read_csv(lengths_path)
    if len(lengths)!=4 or set(lengths.language)!=set(LANGS) or not (lengths.n_examples==1997).all() or not (lengths.checkpoint==spec['checkpoint']).all():
        raise ValueError('Invalid saved token-length audit')
    return frame

def publish(out, manifest, metadata):
    frames = []
    for spec in manifest:
        f = validate_metric(out / (spec['checkpoint'] + '_metrics.csv'), spec)
        if f is not None:
            frames.append(f)
    if not frames:
        return
    all_metrics = pd.concat(frames, ignore_index=True)
    atomic_csv(all_metrics, out / 'ntrex_metrics.csv')
    delta = shifts(all_metrics)
    atomic_csv(delta, out / 'ntrex_layer_shifts.csv')
    prior = shifts(pd.read_csv(ROOT / 'provenance' / 'mlama_reference.csv'))
    comparison = delta.merge(prior, on=['checkpoint','epoch','lambda_metadata','pooling','language_pair'], suffixes=('_ntrex','_mlama'), validate='one_to_one')
    atomic_csv(comparison, out / 'cross_dataset_comparisons.csv')
    atomic_csv(pd.concat([pd.read_csv(out/(f.checkpoint.iloc[0]+'_lengths.csv')) for f in frames],ignore_index=True),out/'token_length_audit.csv')
    metadata['complete_checkpoints'] = len(frames)
    metadata['expected_checkpoints'] = len(manifest)
    metadata['completed_ids'] = [f.checkpoint.iloc[0] for f in frames]
    (out/'evaluation_metadata.json').write_text(json.dumps(metadata,indent=2))
    archive = out.parent / (out.name + '.zip')
    tmp = archive.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for file in sorted(out.glob('*')):
            if file.is_file() and file.suffix in ('.csv','.json'):
                z.write(file,file.name)
        z.write(ROOT/'provenance'/'checkpoint_manifest.csv','checkpoint_manifest.csv')
    tmp.replace(archive)
    print(f'Saved {len(frames)}/{len(manifest)} checkpoints: {archive}',flush=True)

def main():
    global MAX_LENGTH
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-length', type=int, choices=[64, 256], default=256)
    parser.add_argument('--output', type=Path, default=None, help='Fresh output directory; defaults to reproduced/max_length_N')
    parser.add_argument('--max-new',type=int,default=5)
    args=parser.parse_args()
    MAX_LENGTH = args.max_length
    if args.max_new < 1:
        parser.error('--max-new must be positive')
    if not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime in Colab.')
    manifest=pd.read_csv(ROOT/'provenance'/'checkpoint_manifest.csv').to_dict('records')
    df, hashes=dataset()
    print(f'NTREX verified: {len(df)} parallel sentences in each of four languages',flush=True)
    out = (args.output or ROOT/'reproduced'/f'max_length_{MAX_LENGTH}').resolve()
    if out == ROOT/'results' or ROOT/'results' in out.parents:
        raise ValueError('Use a separate output directory, not the archived results.')
    out.mkdir(parents=True, exist_ok=True)
    if any(out.glob('*_metrics.csv')) and not (out/'evaluation_metadata.json').exists():
        raise ValueError('Existing metrics without evaluation metadata; use a fresh output directory.')
    metadata={'dataset':'NTREX-128','dataset_repository':'https://github.com/MicrosoftTranslator/NTREX','dataset_revision':DATA_REV,'dataset_license':'CC BY-SA 4.0','dataset_files':hashes,'batch_size':BATCH_SIZE,'max_length':MAX_LENGTH,'k':K,'layers':list(LAYERS),'python':platform.python_version(),'torch':torch.__version__,'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0),'numpy':np.__version__,'pandas':pd.__version__}
    if (out/'evaluation_metadata.json').exists():
        old=json.loads((out/'evaluation_metadata.json').read_text())
        for key in ('dataset_revision','dataset_files','batch_size','max_length','k','layers'):
            if old[key]!=metadata[key]:
                raise ValueError(f'Resume metadata mismatch: {key}')
    count=0
    for spec in manifest:
        path=out/(spec['checkpoint']+'_metrics.csv')
        if validate_metric(path,spec) is not None:
            print('Already completed:',spec['checkpoint'],flush=True)
            continue
        if count>=args.max_new:
            break
        evaluate_checkpoint(spec,df,out)
        count+=1
        publish(out,manifest,metadata)
    publish(out,manifest,metadata)

if __name__=='__main__':
    main()
