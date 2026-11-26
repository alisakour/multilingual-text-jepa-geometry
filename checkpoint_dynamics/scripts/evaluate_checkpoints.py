"""Evaluate revision-pinned multilingual encoder checkpoints on the quad-parallel mLAMA benchmark."""
import argparse
import gc
import hashlib
import json
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
EXPECTED_SHA = "4a8fe28fb1a9ddb73e3431a60a200d80cbf31e0f4dc69967a83469495345404f"
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
    try:
        for lang in LANGS:
            representations[lang] = extract_language(model, tokenizer, df[f"stmt_{lang}"].tolist())
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
    atomic_csv(result, out / (spec["checkpoint"] + "_metrics.csv"))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", help="Repeat to select multiple checkpoint IDs; default: all.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for extraction.")
    if sha256(args.benchmark) != EXPECTED_SHA:
        raise ValueError("Benchmark SHA-256 mismatch.")
    df = pd.read_csv(args.benchmark)
    assert len(df) == 1725 and df.relation.nunique() == 25
    assert not df[["relation"] + [f"stmt_{lang}" for lang in LANGS]].isna().any().any()
    manifest = pd.read_csv(ROOT / "provenance/checkpoint_manifest.csv")
    if args.checkpoint:
        if not set(args.checkpoint) <= set(manifest.checkpoint):
            raise ValueError("Unknown checkpoint ID.")
        manifest = manifest[manifest.checkpoint.isin(args.checkpoint)]
    args.output.mkdir(parents=True, exist_ok=True)
    reference = pd.read_csv(ROOT / "results/checkpoint_metrics.csv")
    frames = []
    for spec in manifest.to_dict("records"):
        result = evaluate_checkpoint(spec, df, args.output)
        assert len(result) == 12 and not result.duplicated(KEYS).any()
        ref = reference.merge(result, on=KEYS, suffixes=("_ref", "_new"), validate="one_to_one")
        assert len(ref) == 12
        ref["knn_error_pp"] = ref.knn_pct_new - ref.knn_pct_ref
        for col in ("rank_src", "rank_en"):
            ref[col + "_relative_error"] = (ref[col + "_new"] - ref[col + "_ref"]) / ref[col + "_ref"]
        ref["passes"] = (ref.knn_error_pp.abs() <= 0.1) & (ref.rank_src_relative_error.abs() <= 0.005) & (ref.rank_en_relative_error.abs() <= 0.005)
        atomic_csv(ref, args.output / (spec["checkpoint"] + "_comparison.csv"))
        frames.append(result)
        atomic_csv(pd.concat(frames, ignore_index=True), args.output / "checkpoint_metrics.csv")
        print(f"{spec['checkpoint']}: {int(ref.passes.sum())}/12 numerical checks passed", flush=True)
    
if __name__ == "__main__":
    main()
