"""Representation extraction and geometry metrics for the template controls."""
import hashlib
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
LANGS = ("en", "ar", "zh", "fr")
LAYERS = (9, 12)
K, BATCH_SIZE, MAX_LENGTH = 10, 32, 64

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
