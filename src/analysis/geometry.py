"""Geometric and spectral representation metrics (Mutual k-NN, 5-fold Procrustes, Erank)."""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
from scipy.linalg import orthogonal_procrustes


def compute_mutual_knn(Z1: np.ndarray, Z2: np.ndarray, k: int = 10) -> float:
    """Compute mean mutual k-NN overlap (%) after centering and L2 normalization."""
    Z1_c = Z1 - Z1.mean(axis=0, keepdims=True)
    Z2_c = Z2 - Z2.mean(axis=0, keepdims=True)
    Z1_n = Z1_c / (np.linalg.norm(Z1_c, axis=1, keepdims=True) + 1e-8)
    Z2_n = Z2_c / (np.linalg.norm(Z2_c, axis=1, keepdims=True) + 1e-8)

    sim1 = Z1_n @ Z1_n.T
    sim2 = Z2_n @ Z2_n.T
    np.fill_diagonal(sim1, -np.inf)
    np.fill_diagonal(sim2, -np.inf)

    nn1 = np.argsort(-sim1, axis=1)[:, :k]
    nn2 = np.argsort(-sim2, axis=1)[:, :k]
    overlaps = [len(set(nn1[i]).intersection(set(nn2[i]))) / float(k) for i in range(len(Z1))]
    return float(np.mean(overlaps)) * 100.0


def compute_effective_rank(Z: np.ndarray) -> float:
    """Compute effective spectral rank exp(H(sigma / sum(sigma))) in float64."""
    Z_f64 = Z.astype(np.float64)
    Z_c = Z_f64 - Z_f64.mean(axis=0, keepdims=True)
    _, S, _ = np.linalg.svd(Z_c, full_matrices=False)
    p = S / (np.sum(S) + 1e-12)
    p = p[p > 1e-12]
    return float(np.exp(-np.sum(p * np.log(p))))


def fit_apply_relation_debiasing(
    Z_tr: np.ndarray,
    rel_tr: np.ndarray,
    Z_te: Optional[np.ndarray] = None,
    rel_te: Optional[np.ndarray] = None,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Subtract relation-specific centroids fitted strictly on Z_tr (prevents template leakage)."""
    Z_tr_out = np.zeros_like(Z_tr, dtype=np.float64)
    rel_means: Dict[object, np.ndarray] = {}
    global_mean = Z_tr.mean(axis=0, keepdims=True)

    for r in np.unique(rel_tr):
        mask_tr = rel_tr == r
        mu_r = Z_tr[mask_tr].mean(axis=0, keepdims=True) if np.sum(mask_tr) > 1 else global_mean
        rel_means[r] = mu_r
        Z_tr_out[mask_tr] = Z_tr[mask_tr] - mu_r

    if Z_te is None:
        return Z_tr_out

    Z_te_out = np.zeros_like(Z_te, dtype=np.float64)
    for r in np.unique(rel_te):
        mask_te = rel_te == r
        mu_r = rel_means.get(r, global_mean)
        Z_te_out[mask_te] = Z_te[mask_te] - mu_r

    return Z_tr_out, Z_te_out


def compute_instance_level_metrics(
    Z_src: np.ndarray,
    Z_tgt: np.ndarray,
    k: int = 10,
    n_folds: int = 5,
    seed: int = 42,
    relations: Optional[np.ndarray] = None,
) -> Dict[str, Union[np.ndarray, float]]:
    """Compute per-instance Mutual k-NN, 5-fold Procrustes alignment, Top-1/Top-5, and Erank."""
    n = len(Z_src)
    Z_src_f = Z_src.astype(np.float64)
    Z_tgt_f = Z_tgt.astype(np.float64)

    if relations is not None:
        Z1_knn_base = fit_apply_relation_debiasing(Z_src_f, relations)
        Z2_knn_base = fit_apply_relation_debiasing(Z_tgt_f, relations)
    else:
        Z1_knn_base = Z_src_f
        Z2_knn_base = Z_tgt_f

    Z1_c = Z1_knn_base - Z1_knn_base.mean(axis=0, keepdims=True)
    Z2_c = Z2_knn_base - Z2_knn_base.mean(axis=0, keepdims=True)
    Z1_n = Z1_c / (np.linalg.norm(Z1_c, axis=1, keepdims=True) + 1e-8)
    Z2_n = Z2_c / (np.linalg.norm(Z2_c, axis=1, keepdims=True) + 1e-8)

    sim1 = Z1_n @ Z1_n.T
    sim2 = Z2_n @ Z2_n.T
    np.fill_diagonal(sim1, -np.inf)
    np.fill_diagonal(sim2, -np.inf)
    nn1 = np.argsort(-sim1, axis=1)[:, :k]
    nn2 = np.argsort(-sim2, axis=1)[:, :k]
    knn_i = np.array(
        [(len(set(nn1[i]).intersection(set(nn2[i]))) / float(k)) * 100.0 for i in range(n)],
        dtype=float,
    )

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, n_folds)

    unrot_cos_i = np.zeros(n, dtype=float)
    rot_cos_i = np.zeros(n, dtype=float)
    unrot_top1_i = np.zeros(n, dtype=float)
    rot_top1_i = np.zeros(n, dtype=float)
    rot_top5_i = np.zeros(n, dtype=float)

    for te_idx in folds:
        tr_mask = np.ones(n, dtype=bool)
        tr_mask[te_idx] = False
        tr_idx = np.where(tr_mask)[0]

        if relations is not None:
            Z1_tr_raw, Z1_te_raw = fit_apply_relation_debiasing(
                Z_src_f[tr_idx], relations[tr_idx], Z_src_f[te_idx], relations[te_idx]
            )
            Z2_tr_raw, Z2_te_raw = fit_apply_relation_debiasing(
                Z_tgt_f[tr_idx], relations[tr_idx], Z_tgt_f[te_idx], relations[te_idx]
            )
        else:
            Z1_tr_raw, Z1_te_raw = Z_src_f[tr_idx], Z_src_f[te_idx]
            Z2_tr_raw, Z2_te_raw = Z_tgt_f[tr_idx], Z_tgt_f[te_idx]

        mu1_tr = Z1_tr_raw.mean(axis=0, keepdims=True)
        mu2_tr = Z2_tr_raw.mean(axis=0, keepdims=True)

        Z1_tr = Z1_tr_raw - mu1_tr
        Z2_tr = Z2_tr_raw - mu2_tr
        Z1_tr = Z1_tr / (np.linalg.norm(Z1_tr, axis=1, keepdims=True) + 1e-8)
        Z2_tr = Z2_tr / (np.linalg.norm(Z2_tr, axis=1, keepdims=True) + 1e-8)

        Z1_te = Z1_te_raw - mu1_tr
        Z2_te = Z2_te_raw - mu2_tr
        Z1_te = Z1_te / (np.linalg.norm(Z1_te, axis=1, keepdims=True) + 1e-8)
        Z2_te = Z2_te / (np.linalg.norm(Z2_te, axis=1, keepdims=True) + 1e-8)

        unrot_cos_i[te_idx] = np.sum(Z1_te * Z2_te, axis=1)
        sims_unrot = Z1_te @ Z2_te.T
        unrot_top1_i[te_idx] = (
            np.argmax(sims_unrot, axis=1) == np.arange(len(te_idx))
        ).astype(float) * 100.0

        R_fold, _ = orthogonal_procrustes(Z1_tr, Z2_tr)
        Z1_te_rot = Z1_te @ R_fold
        rot_cos_i[te_idx] = np.sum(Z1_te_rot * Z2_te, axis=1)

        sims_rot = Z1_te_rot @ Z2_te.T
        ranks_rot = np.argsort(-sims_rot, axis=1)
        true_te = np.arange(len(te_idx))[:, None]
        rot_top1_i[te_idx] = (ranks_rot[:, 0] == np.arange(len(te_idx))).astype(float) * 100.0
        rot_top5_i[te_idx] = np.any(ranks_rot[:, :5] == true_te, axis=1).astype(float) * 100.0

    return {
        "knn_i": knn_i,
        "unrot_cos_i": unrot_cos_i,
        "rot_cos_i": rot_cos_i,
        "rot_gain_i": rot_cos_i - unrot_cos_i,
        "unrot_top1_i": unrot_top1_i,
        "rot_top1_i": rot_top1_i,
        "rot_top5_i": rot_top5_i,
        "erank_src": compute_effective_rank(Z1_knn_base),
        "erank_tgt": compute_effective_rank(Z2_knn_base),
    }
