#!/usr/bin/env python3
"""Within-relation-only 10-NN control for the matched SIGReg and InfoNCE caches.

All nearest neighbors are selected from the query's own relation (excluding
self), rather than from the full benchmark. This changes the candidate pool relative to Table 2.

Centering and normalization are done over the full benchmark at each layer,
as in the source repo, *before* the candidate sets are restricted.

The script reports raw mutual 10-NN overlap and a relation-size chance-
adjusted overlap: (overlap_fraction - k/(n_relation-1)) /
(1 - k/(n_relation-1)). The reference assumes independent, uniformly sampled neighbor sets
within a relation.

Usage:
  py jepa_within_relation_control.py --benchmark mlama_quad_parallel_benchmark.csv \
      --sigreg maktzgls__bert-jepa-sigreg-opus100-5lang-sigreg.npz \
      --infonce maktzgls__bert-jepa-infonce-opus100-5lang-infonce.npz

Requires NumPy, Python 3.9+. No network or GPU needed.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

POOLS = ("cls", "mean")
LANGS = ("ar", "zh", "fr")
LAYERS = (9, 12)


def read_relation(path: Path) -> np.ndarray:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if "relation" not in (reader.fieldnames or []):
            raise ValueError("Expected 'relation' column in benchmark CSV")
        arr = [r["relation"].strip() for r in reader]
    if not arr or any(not r for r in arr):
        raise ValueError("No relations or empty relation labels")
    return np.asarray(arr)


def read_array(cache, key: str, n: int) -> np.ndarray:
    if key not in cache.files:
        raise ValueError(f"Cache lacks '{key}'")
    arr = cache[key]
    if arr.ndim != 3 or arr.shape[0] <= 12 or arr.shape[1] != n:
        raise ValueError(f"Unexpected shape {key}: {arr.shape}; expected [>=13,{n},D]")
    return arr


def within_neighbors(z: np.ndarray, groups: list[np.ndarray], k: int) -> np.ndarray:
    """[n,k] integer indices: rank neighbors only among same-relation rows."""
    z = np.asarray(z, dtype=np.float64)
    z = z - z.mean(axis=0, keepdims=True)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    result = np.empty((len(z), k), dtype=np.int32)
    for ix in groups:
        subset = z[ix]
        sims = subset @ subset.T
        np.fill_diagonal(sims, -np.inf)
        loc = np.argsort(-sims, axis=1)[:, :k]
        result[ix] = ix[loc]
    return result


def paired_overlap(nn_src: np.ndarray, nn_en: np.ndarray, k: int) -> np.ndarray:
    return (100.0 / k) * (nn_src[:, :, None] == nn_en[:, None, :]).any(axis=2).sum(axis=1)


def evaluate(path: Path, n: int, groups: list[np.ndarray], k: int):
    ret = {}
    with np.load(path, allow_pickle=False) as data:
        print(f"Reading {path.name} ({len(data.files)} arrays)", flush=True)
        for pooling in POOLS:
            en = read_array(data, f"en_{pooling}", n)
            en_nn = {layer: within_neighbors(en[layer], groups, k) for layer in LAYERS}
            del en
            for lang in LANGS:
                src = read_array(data, f"{lang}_{pooling}", n)
                scores = {}
                for layer in LAYERS:
                    src_nn = within_neighbors(src[layer], groups, k)
                    scores[layer] = paired_overlap(src_nn, en_nn[layer], k)
                del src
                ret[(pooling, lang)] = scores[12] - scores[9]
                print(f" {pooling:4s} {lang.upper()}-EN: within-relation-only L12-L9 {ret[(pooling,lang)].mean():+.2f} pp", flush=True)
    return ret


def write_csv(path: Path, rows: list[dict]):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def rnd(x):
    return round(float(x), 5)


def self_test():
    rng = np.random.default_rng(11)
    groups = [np.arange(0,16), np.arange(16,34)]
    z = rng.normal(size=(34,16))
    nn = within_neighbors(z,groups,10)
    assert np.allclose(paired_overlap(nn,nn,10), 100)
    assert np.all((nn[:16] < 16) & (nn[:16] >= 0))
    assert np.all((nn[16:] >= 16) & (nn[16:] < 34))
    assert all(i not in nn[i] for i in range(34))
    assert np.all((paired_overlap(nn,nn[:,::-1],10) - 100) == 0)
    print("Synthetic self-test passed.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", type=Path)
    p.add_argument("--sigreg", type=Path)
    p.add_argument("--infonce", type=Path)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--out-dir", type=Path, default=Path("jepa_within_relation_results"))
    p.add_argument("--self-test", action="store_true")
    a = p.parse_args()
    if a.self_test:
        self_test()
        return
    if not (a.benchmark and a.sigreg and a.infonce):
        p.error("--benchmark, --sigreg, and --infonce are required")
    for filepath in (a.benchmark, a.sigreg, a.infonce):
        if not filepath.is_file():
            p.error(f"Missing input file: {filepath}")
    relations = read_relation(a.benchmark)
    uniq, inv = np.unique(relations, return_inverse=True)
    groups = [np.where(inv == j)[0] for j in range(len(uniq))]
    minimum = min(len(g) for g in groups)
    if a.k < 1 or a.k >= minimum:
        p.error(f"--k must be between 1 and {minimum-1}; smallest relation has {minimum} rows")
    n = len(relations)
    print(f"N={n}, relations={len(uniq)}, minimum relation size={minimum}, k={a.k}")
    sig = evaluate(a.sigreg, n, groups, a.k)
    info = evaluate(a.infonce, n, groups, a.k)
    chance_by_row = np.zeros(n, dtype=float)
    for g in groups:
        chance_by_row[g] = a.k / (len(g)-1)
    adjustment_denominator = 1.0 - chance_by_row
    # Since each relation has >= k+1 rows, denominator > 0.
    if np.any(adjustment_denominator <= 0):
        p.error("A relation is too small for chance-adjusted overlap")
    summary, details = [], []
    for pooling in POOLS:
        for lang in LANGS:
            pair = f"{lang.upper()}-EN"
            s = sig[(pooling,lang)]; t = info[(pooling,lang)]
            contrast = s-t
            # Baseline k/(n_relation-1) cancels in L12-L9 raw differences;
            # chance adjustment divides the layer difference by 1-baseline.
            sa = s/adjustment_denominator; ta = t/adjustment_denominator
            ca = sa-ta
            byrel = []
            for name,g in zip(uniq,groups):
                row={
                    "pooling":pooling,"language_pair":pair,"relation":str(name),"n_examples":len(g),
                    "k":a.k,"random_neighbor_overlap_pct":rnd(100*a.k/(len(g)-1)),
                    "sigreg_within_only_shift_pp":rnd(s[g].mean()),
                    "infonce_within_only_shift_pp":rnd(t[g].mean()),
                    "sigreg_minus_infonce_shift_pp":rnd(contrast[g].mean()),
                    "sigreg_chance_adjusted_shift_pp":rnd(sa[g].mean()),
                    "infonce_chance_adjusted_shift_pp":rnd(ta[g].mean()),
                    "objective_chance_adjusted_contrast_pp":rnd(ca[g].mean()),
                }
                details.append(row)
                byrel.append((float(contrast[g].mean()), float(ca[g].mean())))
            summary.append({
                "pooling":pooling,"language_pair":pair,"n_examples":n,"n_relations":len(uniq),"k":a.k,
                "sigreg_example_weighted_shift_pp":rnd(s.mean()),
                "infonce_example_weighted_shift_pp":rnd(t.mean()),
                "objective_example_weighted_contrast_pp":rnd(contrast.mean()),
                "sigreg_chance_adjusted_shift_pp":rnd(sa.mean()),
                "infonce_chance_adjusted_shift_pp":rnd(ta.mean()),
                "objective_chance_adjusted_contrast_pp":rnd(ca.mean()),
                "objective_relation_macro_contrast_pp":rnd(np.mean([x[0] for x in byrel])),
                "objective_relation_macro_chance_adjusted_contrast_pp":rnd(np.mean([x[1] for x in byrel])),
                "relations_sigreg_worse_than_infonce":sum(x[0]<-1e-9 for x in byrel),
                "relations_tied":sum(abs(x[0])<=1e-9 for x in byrel),
            })
    a.out_dir.mkdir(parents=True,exist_ok=True)
    path_s=a.out_dir/"JEPA_Within_Relation_Control_Summary.csv"
    path_r=a.out_dir/"JEPA_Within_Relation_Control_By_Relation.csv"
    write_csv(path_s,summary); write_csv(path_r,details)
    print(f"Saved {path_s.resolve()}\nSaved {path_r.resolve()}")
    print("Candidate pool: same relation as the query.")
    print("Chance reference: independent uniform neighbor sets.")

if __name__=="__main__":
    main()
