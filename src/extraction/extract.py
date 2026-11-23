"""Multi-layer representation extraction ([CLS] and MeanPool) with local .npz caching."""

from __future__ import annotations

import gc
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModel, AutoTokenizer
from transformers import logging as hf_logging

LANGS = ("en", "ar", "zh", "fr")


def load_models_registry(config_path: Path | str = "configs/models.yaml") -> List[Dict[str, Any]]:
    """Load the ordered list of model configurations from configs/models.yaml."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        raise FileNotFoundError(f"Model registry not found: {cfg_file}")
    with open(cfg_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("models", []))


def repo_cache_path(hf_repo: str, cache_dir: Path | str) -> Path:
    """Return a deterministic filesystem path for caching a Hugging Face checkpoint's layers."""
    safe_name = re.sub(r"[^a-zA-Z0-9._-]+", "__", hf_repo.strip())
    return Path(cache_dir) / f"{safe_name}.npz"


def _load_hf_model(repo: str, device: str) -> AutoModel:
    """Load an encoder in float32 cleanly across both transformers v4 and v5."""
    try:
        mod = AutoModel.from_pretrained(repo, dtype=torch.float32)
    except TypeError:
        mod = AutoModel.from_pretrained(repo, torch_dtype=torch.float32)
    return mod.to(device).eval()


@torch.no_grad()
def extract_all_layers_both_poolings(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    texts: Sequence[str],
    device: str,
    batch_size: int = 128,
    max_length: int = 64,
) -> Dict[str, np.ndarray]:
    """Extract [CLS] and attention-masked MeanPool representations across all hidden states.

    Returns dict with keys 'cls' and 'mean', each of shape [num_layers + 1, N, hidden_dim] (float32).
    """
    num_states = model.config.num_hidden_layers + 1
    cls_layers: List[List[np.ndarray]] = [[] for _ in range(num_states)]
    mean_layers: List[List[np.ndarray]] = [[] for _ in range(num_states)]

    for i in range(0, len(texts), batch_size):
        batch = list(texts[i : i + batch_size])
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        out = model(**enc, output_hidden_states=True)

        mask = enc.attention_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1e-9)

        for l_idx, hs in enumerate(out.hidden_states):
            hs_f = hs.float()
            cls_layers[l_idx].append(hs_f[:, 0, :].cpu().numpy())
            mean_vec = ((hs_f * mask).sum(dim=1) / denom).cpu().numpy()
            mean_layers[l_idx].append(mean_vec)

    return {
        "cls": np.stack([np.vstack(cls_layers[l]) for l in range(num_states)], axis=0).astype(np.float32),
        "mean": np.stack([np.vstack(mean_layers[l]) for l in range(num_states)], axis=0).astype(np.float32),
    }


def extract_and_cache_all_models(
    models_registry: Sequence[Dict[str, Any]],
    df_benchmark: pd.DataFrame,
    cache_dir: Path | str = "data/cache",
    device: str = "auto",
    batch_size: int = 128,
    max_length: int = 64,
    overwrite: bool = False,
) -> None:
    """Extract and cache required poolings for every unique Hugging Face repo in the registry."""
    hf_logging.set_verbosity_error()
    resolved_device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    # Group required pooling modes per unique HF repo (e.g., XLM-R needs both cls & mean)
    repo_poolings: Dict[str, Set[str]] = defaultdict(set)
    ordered_repos: List[str] = []
    for cfg in models_registry:
        repo = cfg["hf_repo"]
        if repo not in repo_poolings:
            ordered_repos.append(repo)
        repo_poolings[repo].add(cfg["pooling"])

    print(f"Extraction device: {resolved_device} | Unique checkpoints: {len(ordered_repos)} | N={len(df_benchmark)}")

    for repo in ordered_repos:
        out_npz = repo_cache_path(repo, cache_path)
        needed_keys = {f"{lang}_{pool}" for lang in LANGS for pool in repo_poolings[repo]}

        if not overwrite and out_npz.exists():
            with np.load(out_npz) as existing:
                if needed_keys.issubset(set(existing.files)):
                    print(f"[Cached] Skipping {repo} -> {out_npz}")
                    continue

        print(f"Extracting representations: {repo} (poolings={sorted(repo_poolings[repo])})")
        tok = AutoTokenizer.from_pretrained(repo)
        mod = _load_hf_model(repo, resolved_device)

        arrays_to_save: Dict[str, np.ndarray] = {}
        for lang in LANGS:
            pooled = extract_all_layers_both_poolings(
                mod,
                tok,
                df_benchmark[f"stmt_{lang}"].tolist(),
                device=resolved_device,
                batch_size=batch_size,
                max_length=max_length,
            )
            for pool in repo_poolings[repo]:
                arrays_to_save[f"{lang}_{pool}"] = pooled[pool]

        np.savez(out_npz, **arrays_to_save)
        print(f"Saved cache -> {out_npz}")

        del mod, tok, arrays_to_save
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_cached_layers(
    cfg: Dict[str, Any],
    cache_dir: Path | str = "data/cache",
) -> Dict[str, np.ndarray]:
    """Load cached representations for one model config as {lang: array[num_layers+1, N, D]}."""
    npz_file = repo_cache_path(cfg["hf_repo"], cache_dir)
    if not npz_file.exists():
        raise FileNotFoundError(
            f"Missing cached representations for '{cfg['name']}': {npz_file}. "
            "Run scripts/run_extraction.py first."
        )
    pool = cfg["pooling"]
    with np.load(npz_file) as data:
        return {lang: data[f"{lang}_{pool}"] for lang in LANGS}
