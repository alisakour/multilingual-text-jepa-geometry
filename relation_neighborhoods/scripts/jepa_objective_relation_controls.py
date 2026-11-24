#!/usr/bin/env python3
"""Matched SIGReg vs InfoNCE relation-level diagnostics for multilingual BERT-JEPA.

Required inputs:
    1) mlama_quad_parallel_benchmark.csv
    2) maktzgls__bert-jepa-sigreg-opus100-5lang-sigreg.npz
    3) maktzgls__bert-jepa-infonce-opus100-5lang-infonce.npz

This script compares L9 and L12 across AR/EN, ZH/EN, FR/EN and CLS/mean pooling.
For each example it computes two metrics:
  - global_10nn: exactly the repository's global, centered, L2-normalized
    mutual 10-nearest-neighbor overlap on the full benchmark.
  - cross_relation_10nn: overlap after excluding candidates with the
    same relation label as the query.

In each case, the example scores are then averaged per relation (25 rows).
The matched objective difference is SIGReg (L12-L9) minus InfoNCE (L12-L9).
Negative values indicate that SIGReg declined more (or improved less).

Requires NumPy. Runs on CPU using cached representations.
Benchmark and cache rows must correspond in order.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

import numpy as np

EXPECTED_SIGREG = {
    ("cls", "ar"): -11.91, ("cls", "zh"): -6.42, ("cls", "fr"): -12.81,
    ("mean", "ar"): -16.85, ("mean", "zh"): -12.22, ("mean", "fr"): -20.79,
}
EXPECTED_INFO_CLS = {("cls", "ar"): 5.30, ("cls", "zh"): 5.33, ("cls", "fr"): 11.89}


def read_relations(filename: Path) -> np.ndarray:
    with filename.open("r", encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "relation" not in r.fieldnames:
            raise ValueError("Benchmark CSV needs a 'relation' column")
        labels = [line["relation"].strip() for line in r]
    if not labels or any(not v for v in labels):
        raise ValueError("Benchmark has missing/empty relation labels")
    return np.asarray(labels, dtype=str)


def top_neighbors(z: np.ndarray, relations: np.ndarray, k: int = 10):
    """Return global and cross-relation k-NN indices for all N rows."""
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2:
        raise ValueError(f"Expected 2D representations; got shape {z.shape}")
    n = z.shape[0]
    if n != len(relations) or n <= k:
        raise ValueError("N mismatch or too few examples")
    z -= z.mean(axis=0, keepdims=True)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)
    # Match the sorting convention in src/analysis/geometry.py.
    global_neighbors = np.argsort(-sim, axis=1)[:, :k]
    same_relation = relations[:, None] == relations[None, :]
    sim[same_relation] = -np.inf
    if np.any((~same_relation).sum(axis=1) < k):
        raise ValueError("Not enough cross-relation candidates")
    cross_neighbors = np.argsort(-sim, axis=1)[:, :k]
    return global_neighbors, cross_neighbors


def overlap_pct(src_neighbors: np.ndarray, tgt_neighbors: np.ndarray) -> np.ndarray:
    if src_neighbors.shape != tgt_neighbors.shape:
        raise ValueError("Neighbor shapes don't match")
    k = src_neighbors.shape[1]
    matching = (src_neighbors[:, :, None] == tgt_neighbors[:, None, :]).any(axis=2)
    return 100.0 * matching.sum(axis=1, dtype=np.float64) / k


def evaluate_cache(filename: Path, relations: np.ndarray):
    n = len(relations)
    observed = {}
    with np.load(filename, allow_pickle=False) as bundle:
        keys_needed = {f"{lang}_{pool}" for lang in ("en", "ar", "zh", "fr") for pool in ("cls", "mean")}
        missing = keys_needed - set(bundle.files)
        if missing:
            raise ValueError(f"Missing cache keys in {filename.name}: {sorted(missing)}")
        print(f"Reading {filename.name} ({len(bundle.files)} arrays)", flush=True)
        for pool in ("cls", "mean"):
            en_states = bundle[f"en_{pool}"]
            if en_states.ndim != 3 or en_states.shape[0] <= 12 or en_states.shape[1] != n:
                raise ValueError(f"Unexpected {pool} EN array shape {en_states.shape}; expected (>=13, {n}, D)")
            for layer in (9, 12):
                en_gl, en_cr = top_neighbors(en_states[layer], relations)
                for lang in ("ar", "zh", "fr"):
                    states = bundle[f"{lang}_{pool}"]
                    if states.shape != en_states.shape:
                        raise ValueError(f"{lang}_{pool} has shape {states.shape}, expected {en_states.shape}")
                    src_gl, src_cr = top_neighbors(states[layer], relations)
                    observed[(pool, lang, layer, "global_10nn")] = overlap_pct(src_gl, en_gl)
                    observed[(pool, lang, layer, "cross_relation_10nn")] = overlap_pct(src_cr, en_cr)
                    print(f"  {pool:4s} {lang.upper()}-EN L{layer}: computed global and cross-relation scores", flush=True)
    return observed


def csv_save(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError(f"No data for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def weighted_rel_mean(values: np.ndarray, inv: np.ndarray, weights: np.ndarray) -> np.ndarray:
    out = np.zeros(len(weights), dtype=np.float64)
    for j, weight in enumerate(weights):
        out[j] = values[inv == j].mean()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", type=Path, required=True)
    ap.add_argument("--sigreg", type=Path, required=True)
    ap.add_argument("--infonce", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("jepa_matched_results"))
    ap.add_argument("--skip-reference-checks", action="store_true", help="Disable Table 2 reference validation for non-reference inputs.")
    a = ap.parse_args()
    for p in (a.benchmark, a.sigreg, a.infonce):
        if not p.is_file():
            ap.error(f"Cannot find file: {p}")
    rel = read_relations(a.benchmark)
    unique, inv, counts = np.unique(rel, return_inverse=True, return_counts=True)
    print(f"Loaded benchmark: N={len(rel)}, relations={len(unique)}", flush=True)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    res = {
        "SIGReg": evaluate_cache(a.sigreg, rel),
        "InfoNCE": evaluate_cache(a.infonce, rel),
    }
    summary_rows, detail_rows = [], []
    failures = 0
    for pool in ("cls", "mean"):
        for lang in ("ar", "zh", "fr"):
            for metric in ("global_10nn", "cross_relation_10nn"):
                sig_l9 = res["SIGReg"][(pool, lang, 9, metric)]
                sig_l12 = res["SIGReg"][(pool, lang, 12, metric)]
                inf_l9 = res["InfoNCE"][(pool, lang, 9, metric)]
                inf_l12 = res["InfoNCE"][(pool, lang, 12, metric)]
                sig_delta, inf_delta = sig_l12-sig_l9, inf_l12-inf_l9
                contrast = sig_delta-inf_delta
                sig_by = weighted_rel_mean(sig_delta, inv, counts)
                inf_by = weighted_rel_mean(inf_delta, inv, counts)
                contrast_by = sig_by-inf_by
                sig_m = float(sig_delta.mean())
                inf_m = float(inf_delta.mean())
                if metric == "global_10nn":
                    expected_s = EXPECTED_SIGREG[(pool, lang)]
                    is_sig_match = abs(sig_m-expected_s) < 0.15
                    if not is_sig_match:
                        failures += 1
                    expected_i = EXPECTED_INFO_CLS.get((pool, lang))
                    is_info_match = "" if expected_i is None else abs(inf_m-expected_i) < 0.15
                    if expected_i is not None and not is_info_match:
                        failures += 1
                    print(f"CHECK {pool} {lang.upper()} global SIGReg {sig_m:+.2f} (expected {expected_s:+.2f}, match {is_sig_match}); InfoNCE {inf_m:+.2f} (expected {expected_i if expected_i is not None else 'not specified'})")
                else:
                    expected_s, expected_i, is_sig_match, is_info_match = "", "", "", ""
                # Sensitivity of weighted contrast to removing any *relation's scores*.
                # The neighbor graphs remain fixed.
                keep_deltas = [float(contrast[inv != rid].mean()) for rid in range(len(unique))]
                summary_rows.append({
                    "pooling": pool, "language_pair": lang.upper()+"-EN", "metric":metric,
                    "n_examples": len(rel), "n_relations": len(unique),
                    "sigreg_l9_pct":round(float(sig_l9.mean()),5),
                    "sigreg_l12_pct":round(float(sig_l12.mean()),5),
                    "sigreg_shift_pp": round(sig_m,5),
                    "infonce_l9_pct":round(float(inf_l9.mean()),5),
                    "infonce_l12_pct":round(float(inf_l12.mean()),5),
                    "infonce_shift_pp": round(inf_m,5),
                    "sigreg_minus_infonce_shift_pp": round(float(contrast.mean()),5),
                    "relations_with_sigreg_drop": int(np.sum(sig_by < -1e-9)),
                    "relations_with_infonce_drop": int(np.sum(inf_by < -1e-9)),
                    "relations_sigreg_worse_than_infonce": int(np.sum(contrast_by < -1e-9)),
                    "relation_median_objective_contrast_pp":round(float(np.median(contrast_by)),5),
                    "leave_one_relation_out_min_contrast_pp":round(float(min(keep_deltas)),5),
                    "leave_one_relation_out_max_contrast_pp":round(float(max(keep_deltas)),5),
                    "expected_sigreg_readme_shift_pp":expected_s,
                    "matches_sigreg_readme":is_sig_match,
                    "expected_infonce_readme_shift_pp":expected_i,
                    "matches_infonce_readme":is_info_match,
                })
                for j, r in enumerate(unique):
                    ids = inv==j
                    detail_rows.append({
                        "pooling":pool,"language_pair":lang.upper()+"-EN","metric":metric,
                        "relation":r,"n_examples":int(counts[j]),
                        "sigreg_l9_pct":round(float(sig_l9[ids].mean()),5),
                        "sigreg_l12_pct":round(float(sig_l12[ids].mean()),5),
                        "sigreg_shift_pp":round(float(sig_by[j]),5),
                        "infonce_l9_pct":round(float(inf_l9[ids].mean()),5),
                        "infonce_l12_pct":round(float(inf_l12[ids].mean()),5),
                        "infonce_shift_pp":round(float(inf_by[j]),5),
                        "sigreg_minus_infonce_shift_pp":round(float(contrast_by[j]),5),
                    })
    p1 = a.out_dir / "JEPA_Objective_Comparison_Summary.csv"
    p2 = a.out_dir / "JEPA_Objective_Comparison_By_Relation.csv"
    csv_save(p1, summary_rows)
    csv_save(p2, detail_rows)
    print(f"\nSaved {p1.resolve()}")
    print(f"Saved {p2.resolve()}")
    print("cross_relation_10nn uses a restricted candidate pool.")
    print("The leave-one-relation-out check holds the full-pool neighbor graph fixed.")
    if failures and not a.skip_reference_checks:
        print(f"WARNING: {failures} known Table 2 global shifts were not reproduced. Check row alignment or input caches.",file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
