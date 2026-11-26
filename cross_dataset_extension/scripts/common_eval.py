"""Shared extraction, geometry, and checkpoint utilities for NTREX & MASSIVE."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer

LANGS = ("en", "ar", "zh", "fr")
LAYERS = (9, 12)
K_NN = 10
BATCH_SIZE = 32

# Complete checkpoint registry across 5-lang trajectories, 15-lang models, and baselines
CHECKPOINT_SPECS: List[Dict[str, Any]] = [
    # 5-Lang OPUS100 InfoNCE (lambda = 0.1)
    {
        "checkpoint": "infonce_01_e1",
        "group": "5lang_infonce_01",
        "repo": "maktzgls/bert-jepa-infonce-opus100-5lang-infonce",
        "revision": "5d8a2bc46a2a09b8f0b68d1b1990ba14a5c3fe98",
        "lambda_metadata": 0.1,
        "epoch": 1,
        "recorded_steps": 781,
    },
    {
        "checkpoint": "infonce_01_e4",
        "group": "5lang_infonce_01",
        "repo": "maktzgls/bert-jepa-infonce-opus100-5lang-infonce",
        "revision": "6cdb7c65d6a636c73033cd72b096bf0fa5e3bdec",
        "lambda_metadata": 0.1,
        "epoch": 4,
        "recorded_steps": 3124,
    },
    {
        "checkpoint": "infonce_01_e5",
        "group": "5lang_infonce_01",
        "repo": "maktzgls/bert-jepa-infonce-opus100-5lang-infonce",
        "revision": "ca1fe2239c9e22ff746e4e634ec33b15d617414f",
        "lambda_metadata": 0.1,
        "epoch": 5,
        "recorded_steps": 3905,
    },
    # 5-Lang OPUS100 InfoNCE (lambda = 1.0)
    {
        "checkpoint": "infonce_1_e1",
        "group": "5lang_infonce_1",
        "repo": "maktzgls/bert-jepa-infonce-opus100-5lang-infonce",
        "revision": "c3fb7c6a7fc2c36f4648b33c91e2d464e6e237bb",
        "lambda_metadata": 1.0,
        "epoch": 1,
        "recorded_steps": 781,
    },
    {
        "checkpoint": "infonce_1_e4",
        "group": "5lang_infonce_1",
        "repo": "maktzgls/bert-jepa-infonce-opus100-5lang-infonce",
        "revision": "b9c8111a3f6dbe584103d1f589cce955fae503bc",
        "lambda_metadata": 1.0,
        "epoch": 4,
        "recorded_steps": 3124,
    },
    # 5-Lang OPUS100 SIGReg (lambda = 0.1)
    {
        "checkpoint": "sigreg_01_e1",
        "group": "5lang_sigreg_01",
        "repo": "maktzgls/bert-jepa-sigreg-opus100-5lang-sigreg",
        "revision": "42bca7e62db0d8991d38e291e27a2eae567d5358",
        "lambda_metadata": 0.1,
        "epoch": 1,
        "recorded_steps": 3125,
    },
    {
        "checkpoint": "sigreg_01_e4",
        "group": "5lang_sigreg_01",
        "repo": "maktzgls/bert-jepa-sigreg-opus100-5lang-sigreg",
        "revision": "b2848e5015557beb65746e3ddc5106b62730f411",
        "lambda_metadata": 0.1,
        "epoch": 4,
        "recorded_steps": 12500,
    },
    {
        "checkpoint": "sigreg_01_e10",
        "group": "5lang_sigreg_01",
        "repo": "maktzgls/bert-jepa-sigreg-opus100-5lang-sigreg",
        "revision": "f9b40cbb84bf959313155d16fcdaff62176ca3a3",
        "lambda_metadata": 0.1,
        "epoch": 10,
        "recorded_steps": 31250,
    },
    # 15-Lang InfoNCE Factorial (FLORES & OPUS100) + Baselines
    {
        "checkpoint": "flores_mono15",
        "group": "15lang_flores",
        "repo": "maktzgls/bert-jepa-infonce-monolingual-15lang-flores-flores-15lang-infonce-monolingual",
        "revision": "main",
        "lambda_metadata": 1.0,
        "epoch": 50,
        "recorded_steps": 21950,
    },
    {
        "checkpoint": "flores_biling15",
        "group": "15lang_flores",
        "repo": "maktzgls/bert-jepa-infonce-bilingual-15lang-flores-flores-15lang-infonce-bilingual",
        "revision": "main",
        "lambda_metadata": 1.0,
        "epoch": 50,
        "recorded_steps": 21950,
    },
    {
        "checkpoint": "opus_mono15",
        "group": "15lang_opus",
        "repo": "maktzgls/bert-jepa-infonce-monolingual-15lang-opus100-15lang-infonce-monolingual",
        "revision": "main",
        "lambda_metadata": 1.0,
        "epoch": 10,
        "recorded_steps": 21870,
    },
    {
        "checkpoint": "opus_biling15",
        "group": "15lang_opus",
        "repo": "maktzgls/bert-jepa-infonce-bilingual-15lang-opus100-15lang-infonce-bilingual",
        "revision": "main",
        "lambda_metadata": 1.0,
        "epoch": 10,
        "recorded_steps": 21870,
    },
    {
        "checkpoint": "xlmr_base",
        "group": "baseline_mlm",
        "repo": "FacebookAI/xlm-roberta-base",
        "revision": "main",
        "lambda_metadata": np.nan,
        "epoch": 0,
        "recorded_steps": 0,
    },
    {
        "checkpoint": "me5_base",
        "group": "supervised_upper_bound",
        "repo": "intfloat/multilingual-e5-base",
        "revision": "main",
        "lambda_metadata": np.nan,
        "epoch": 0,
        "recorded_steps": 0,
    },
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def top_neighbors(z: np.ndarray, k: int = K_NN) -> np.ndarray:
    z = np.array(z, dtype=np.float64, copy=True)
    z -= z.mean(axis=0, keepdims=True)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)[:, :k]


def mutual_knn_pct(nn_src: np.ndarray, nn_en: np.ndarray) -> float:
    return float(100.0 * (nn_src[:, :, None] == nn_en[:, None, :]).any(axis=2).mean())


def effective_rank(z: np.ndarray) -> float:
    z = np.array(z, dtype=np.float64, copy=True)
    z -= z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(z, compute_uv=False)
    p = s / (s.sum() + 1e-12)
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))


def shared_parts_by_label(
    nn_src: np.ndarray, nn_en: np.ndarray, labels: np.ndarray
) -> Tuple[float, float, float]:
    """Return (total_pct, same_label_pct, diff_label_pct) matching Chaewon's decomposition."""
    k = nn_src.shape[1]
    shared = (nn_src[:, :, None] == nn_en[:, None, :]).any(axis=2)
    same_lbl = labels[nn_src] == labels[:, None]
    same_pct = float(100.0 * (shared & same_lbl).sum(axis=1).mean() / k)
    diff_pct = float(100.0 * (shared & ~same_lbl).sum(axis=1).mean() / k)
    return same_pct + diff_pct, same_pct, diff_pct


def _load_model_clean(snapshot: Path):
    kwargs_base = dict(
        local_files_only=True,
        trust_remote_code=False,
        use_safetensors=True,
        output_loading_info=True,
    )
    for dtype_kw in ({"dtype": torch.float32}, {"torch_dtype": torch.float32}):
        for extra_kw in ({"add_pooling_layer": False}, {}):
            try:
                return AutoModel.from_pretrained(snapshot, **kwargs_base, **dtype_kw, **extra_kw)
            except TypeError:
                continue
    raise RuntimeError(f"Failed to load model from {snapshot}")


@torch.inference_mode()
def extract_layers_and_lengths(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    df: pd.DataFrame,
    checkpoint_name: str,
    max_length: int,
    batch_size: int = BATCH_SIZE,
    device: str = "cuda",
) -> Tuple[Dict[str, Dict[Tuple[str, int], np.ndarray]], pd.DataFrame]:
    reps: Dict[str, Dict[Tuple[str, int], np.ndarray]] = {}
    length_rows: List[Dict[str, Any]] = []

    for lang in LANGS:
        texts = df[f"stmt_{lang}"].tolist()
        lens = [
            len(ids)
            for start in range(0, len(texts), 64)
            for ids in tokenizer(
                texts[start : start + 64],
                truncation=False,
                add_special_tokens=True,
            )["input_ids"]
        ]
        n_over = sum(n > max_length for n in lens)
        length_rows.append(
            {
                "checkpoint": checkpoint_name,
                "language": lang,
                "n_examples": len(lens),
                f"n_over_{max_length}": n_over,
                f"pct_over_{max_length}": 100.0 * n_over / len(lens),
                "max_tokens": max(lens),
                "mean_tokens": float(np.mean(lens)),
            }
        )

        arrays: Dict[Tuple[str, int], List[np.ndarray]] = {
            (pool, layer): [] for pool in ("cls", "mean") for layer in LAYERS
        }
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i : i + batch_size],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            states = model(**enc, output_hidden_states=True).hidden_states
            mask = enc.attention_mask.unsqueeze(-1).float()
            denom = mask.sum(dim=1).clamp(min=1e-9)
            for layer in LAYERS:
                hs = states[layer].float()
                arrays[("cls", layer)].append(hs[:, 0, :].cpu().numpy())
                arrays[("mean", layer)].append(((hs * mask).sum(dim=1) / denom).cpu().numpy())

        reps[lang] = {
            key: np.concatenate(chunks, axis=0).astype(np.float32)
            for key, chunks in arrays.items()
        }

    return reps, pd.DataFrame(length_rows)


def evaluate_single_checkpoint(
    spec: Dict[str, Any],
    df: pd.DataFrame,
    out_dir: Path,
    max_length: int,
    batch_size: int = BATCH_SIZE,
    device: str = "cuda",
    group_labels: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Download, verify, extract L9/L12 representations, and compute metrics for one checkpoint."""
    ckpt = spec["checkpoint"]
    print(f"Evaluating {ckpt} ({spec['repo']} @ {spec['revision'][:8]})...", flush=True)

    snapshot = Path(
        snapshot_download(
            repo_id=spec["repo"],
            revision=spec["revision"],
            allow_patterns=["*.json", "*.model", "*.txt", "model.safetensors"],
        )
    )
    weights_sha256 = sha256_file(snapshot / "model.safetensors")

    dual_cfg_path = snapshot / "dual_objective_config.json"
    if dual_cfg_path.exists() and pd.notna(spec["lambda_metadata"]):
        dual_cfg = json.loads(dual_cfg_path.read_text())
        if abs(float(dual_cfg["lambda_alignment"]) - float(spec["lambda_metadata"])) > 1e-6:
            raise RuntimeError(f"Lambda mismatch for {ckpt}: {dual_cfg} vs {spec['lambda_metadata']}")

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
    model, loading = _load_model_clean(snapshot)
    (out_dir / f"{ckpt}_loading.json").write_text(json.dumps(loading, default=str, indent=2))
    if loading.get("missing_keys") or loading.get("mismatched_keys") or loading.get("error_msgs"):
        raise RuntimeError(f"Weight loading error for {ckpt}: {loading}")

    model = model.to(device).eval()
    try:
        reps, len_df = extract_layers_and_lengths(
            model, tokenizer, df, ckpt, max_length=max_length, batch_size=batch_size, device=device
        )
    finally:
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metric_rows: List[Dict[str, Any]] = []
    decomp_rows: List[Dict[str, Any]] = []

    for pool in ("cls", "mean"):
        for layer in LAYERS:
            key = (pool, layer)
            en = reps["en"][key]
            en_nn = top_neighbors(en, k=K_NN)
            en_rank = effective_rank(en)

            for lang in LANGS[1:]:
                src = reps[lang][key]
                src_nn = top_neighbors(src, k=K_NN)
                knn_val = mutual_knn_pct(src_nn, en_nn)
                src_rank = effective_rank(src)

                metric_rows.append(
                    {
                        "checkpoint": ckpt,
                        "group": spec["group"],
                        "repo": spec["repo"],
                        "revision": spec["revision"],
                        "lambda_metadata": spec["lambda_metadata"],
                        "epoch": spec["epoch"],
                        "recorded_steps": spec["recorded_steps"],
                        "weights_sha256": weights_sha256,
                        "pooling": pool,
                        "layer": layer,
                        "language_pair": f"{lang.upper()}-EN",
                        "n_examples": len(df),
                        "knn_pct": knn_val,
                        "rank_src": src_rank,
                        "rank_en": en_rank,
                    }
                )

                if group_labels is not None:
                    tot, within, across = shared_parts_by_label(src_nn, en_nn, group_labels)
                    decomp_rows.append(
                        {
                            "checkpoint": ckpt,
                            "group": spec["group"],
                            "pooling": pool,
                            "layer": layer,
                            "language_pair": f"{lang.upper()}-EN",
                            "total_10nn_pct": tot,
                            "within_label_pct": within,
                            "across_label_pct": across,
                        }
                    )

    metrics_df = pd.DataFrame(metric_rows)
    atomic_csv(len_df, out_dir / f"{ckpt}_lengths.csv")
    atomic_csv(metrics_df, out_dir / f"{ckpt}_metrics.csv")

    decomp_df = pd.DataFrame(decomp_rows) if decomp_rows else None
    if decomp_df is not None:
        atomic_csv(decomp_df, out_dir / f"{ckpt}_decomp.csv")

    return metrics_df, decomp_df


def compute_layer_shifts(metrics_df: pd.DataFrame) -> pd.DataFrame:
    idx_cols = ["checkpoint", "pooling", "language_pair"]
    l9 = metrics_df[metrics_df["layer"] == 9].set_index(idx_cols)
    l12 = metrics_df[metrics_df["layer"] == 12].set_index(idx_cols)
    meta_cols = ["group", "epoch", "lambda_metadata"]
    val_cols = ["knn_pct", "rank_src", "rank_en"]
    joined = l9[meta_cols + val_cols].join(l12[val_cols], lsuffix="_l9", rsuffix="_l12")
    for col in val_cols:
        joined[f"{col}_shift"] = joined[f"{col}_l12"] - joined[f"{col}_l9"]
    return joined.reset_index()
