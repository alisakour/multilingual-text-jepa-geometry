#!/usr/bin/env python3
"""Decompose BERT-JEPA global mutual 10-NN alignment into within-relation and
across-relation shared-neighbor contributions.

Select neighbors from the full benchmark as in Table 2. Partition each
shared neighbor by whether its relation matches the query's relation.
The two contributions sum to the original mutual 10-NN score. Unlike
cross_relation_10nn, this partition does not alter candidate selection.

Inputs: benchmark CSV and SIGReg/InfoNCE NPZ caches.
Outputs: aggregate and relation-level CSVs. Requires NumPy; runs on CPU.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import numpy as np

K = 10
LANGS = ("ar", "zh", "fr")
POOLS = ("cls", "mean")
MODELS = ("SIGReg", "InfoNCE")
EXPECTED_SIGREG = {
    ("cls", "ar"): -11.91, ("cls", "zh"): -6.42, ("cls", "fr"): -12.81,
    ("mean", "ar"): -16.85, ("mean", "zh"): -12.22, ("mean", "fr"): -20.79,
}
EXPECTED_INFO_CLS = {( "cls", "ar"): 5.30, ("cls", "zh"): 5.33, ("cls", "fr"): 11.89}


def read_relations(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        if not r.fieldnames or "relation" not in r.fieldnames:
            raise ValueError("Benchmark CSV requires a 'relation' column")
        vals = [line["relation"].strip() for line in r]
    if not vals or any(not v for v in vals):
        raise ValueError("Empty benchmark or empty relation label")
    return np.asarray(vals)


def neighbors(z: np.ndarray, k: int = K) -> np.ndarray:
    """Same centering, L2 normalization and sort as original geometry.py."""
    z = np.asarray(z, dtype=np.float64).copy()
    if z.ndim != 2 or len(z) <= k:
        raise ValueError(f"Unexpected embedding shape {z.shape}")
    z -= z.mean(axis=0, keepdims=True)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    sim = z @ z.T
    np.fill_diagonal(sim, -np.inf)
    return np.argsort(-sim, axis=1)[:, :k]


def shared_parts(src_nn: np.ndarray, en_nn: np.ndarray, rel: np.ndarray):
    """Global overlap and its additive relation-specific contributions in pp."""
    k = src_nn.shape[1]
    shared = (src_nn[:, :, None] == en_nn[:, None, :]).any(axis=2)
    same_relation = rel[src_nn] == rel[:, None]
    same_overlap = 100.0 * (shared & same_relation).sum(axis=1) / k
    across_overlap = 100.0 * (shared & ~same_relation).sum(axis=1) / k
    src_same_fraction = 100.0 * same_relation.sum(axis=1) / k
    en_same_fraction = 100.0 * (rel[en_nn] == rel[:, None]).sum(axis=1) / k
    return {
        "total": same_overlap + across_overlap,
        "within": same_overlap,
        "across": across_overlap,
        "src_same_share": src_same_fraction,
        "en_same_share": en_same_fraction,
    }


def common_nn_pct(nn_l9: np.ndarray, nn_l12: np.ndarray):
    k = nn_l9.shape[1]
    return 100.0 * (nn_l9[:, :, None] == nn_l12[:, None, :]).any(axis=2).sum(axis=1) / k


def eval_cache(path: Path, rel: np.ndarray):
    results = {}
    n = len(rel)
    with np.load(path, allow_pickle=False) as cache:
        required = {f"{lang}_{pool}" for pool in POOLS for lang in ("en", *LANGS)}
        missing = required - set(cache.files)
        if missing:
            raise ValueError(f"{path.name} missing arrays {sorted(missing)}")
        print(f"Reading {path.name}", flush=True)
        for pool in POOLS:
            en_states = cache[f"en_{pool}"]
            if en_states.ndim != 3 or en_states.shape[0] <= 12 or en_states.shape[1] != n:
                raise ValueError(f"Bad shape for en_{pool}: {en_states.shape}")
            en_nn = {layer: neighbors(en_states[layer]) for layer in (9, 12)}
            en_stability = common_nn_pct(en_nn[9], en_nn[12])
            del en_states
            for lang in LANGS:
                states = cache[f"{lang}_{pool}"]
                if states.ndim != 3 or states.shape[0] <= 12 or states.shape[1] != n:
                    raise ValueError(f"Bad shape for {lang}_{pool}: {states.shape}")
                src_nn = {layer: neighbors(states[layer]) for layer in (9, 12)}
                del states
                parts9 = shared_parts(src_nn[9], en_nn[9], rel)
                parts12 = shared_parts(src_nn[12], en_nn[12], rel)
                delta = {key: parts12[key] - parts9[key] for key in parts9}
                delta["src_nn_stability"] = common_nn_pct(src_nn[9], src_nn[12])
                delta["en_nn_stability"] = en_stability
                if not np.allclose(delta["total"], delta["within"] + delta["across"], atol=1e-9):
                    raise AssertionError("Component sum mismatch")
                results[(pool, lang)] = delta
                print(f"  {pool:4s} {lang.upper()}-EN global L12-L9: "
                      f"{delta['total'].mean():+.2f} pp", flush=True)
    return results


def save_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError("No output rows")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def val(n: float):
    return round(float(n), 5)


def self_test():
    rng = np.random.default_rng(42)
    rel = np.array(["P1"] * 10 + ["P2"] * 10 + ["P3"] * 10)
    x = rng.normal(size=(30, 12))
    y = x + 0.01 * rng.normal(size=(30, 12))
    p = shared_parts(neighbors(x), neighbors(y), rel)
    assert np.allclose(p["within"] + p["across"], p["total"])
    assert np.all((p["src_same_share"] >= 0) & (p["src_same_share"] <= 100))
    assert np.allclose(common_nn_pct(neighbors(x), neighbors(x)), 100)
    print("Synthetic self-test passed.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark", type=Path)
    ap.add_argument("--sigreg", type=Path)
    ap.add_argument("--infonce", type=Path)
    ap.add_argument("--out-dir", type=Path, default=Path("jepa_neighbor_results"))
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not all((args.benchmark, args.sigreg, args.infonce)):
        ap.error("--benchmark, --sigreg and --infonce are required")
    for path in (args.benchmark, args.sigreg, args.infonce):
        if not path.is_file():
            ap.error(f"File not found: {path}")
    rel = read_relations(args.benchmark)
    unique, inv = np.unique(rel, return_inverse=True)
    print(f"Benchmark N={len(rel)}, relations={len(unique)}")
    r = {"SIGReg": eval_cache(args.sigreg, rel), "InfoNCE": eval_cache(args.infonce, rel)}
    summary, detail = [], []
    failures = []
    for pool in POOLS:
        for lang in LANGS:
            pair = f"{lang.upper()}-EN"
            s = r["SIGReg"][(pool, lang)]
            t = r["InfoNCE"][(pool, lang)]
            sig = float(s["total"].mean())
            inf = float(t["total"].mean())
            expected_s = EXPECTED_SIGREG[(pool, lang)]
            expected_i = EXPECTED_INFO_CLS.get((pool, lang))
            if abs(sig - expected_s) > 0.15:
                failures.append(f"SIGReg {pool} {pair}: {sig:+.3f} vs {expected_s:+.2f}")
            if expected_i is not None and abs(inf - expected_i) > 0.15:
                failures.append(f"InfoNCE {pool} {pair}: {inf:+.3f} vs {expected_i:+.2f}")
            row = {"pooling": pool, "language_pair": pair, "n_examples": len(rel), "n_relations": len(unique)}
            for name, x in (("sigreg", s), ("infonce", t)):
                for part in ("total", "within", "across", "src_same_share", "en_same_share",
                             "src_nn_stability", "en_nn_stability"):
                    row[f"{name}_{part}_{'l12_minus_l9_pp' if part not in ('src_nn_stability','en_nn_stability') else 'l9_l12_overlap_pct'}"] = val(x[part].mean())
            for part in ("total", "within", "across"):
                row[f"objective_contrast_{part}_pp"] = val((s[part]-t[part]).mean())
            summary.append(row)
            for j, relation in enumerate(unique):
                mask = inv == j
                dr = {"pooling": pool, "language_pair": pair, "relation": relation,
                      "n_examples": int(mask.sum())}
                for name, x in (("sigreg", s), ("infonce", t)):
                    for part in ("total", "within", "across"):
                        dr[f"{name}_{part}_shift_pp"] = val(x[part][mask].mean())
                for part in ("total", "within", "across"):
                    dr[f"objective_contrast_{part}_pp"] = val((s[part]-t[part])[mask].mean())
                detail.append(dr)
    if failures:
        raise ValueError("Table 2 reference validation failed: " + "; ".join(failures))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    a = args.out_dir / "JEPA_Neighbor_Decomposition_Summary.csv"
    b = args.out_dir / "JEPA_Neighbor_Decomposition_By_Relation.csv"
    save_csv(a, summary)
    save_csv(b, detail)
    print("Table 2 consistency checks passed.")
    print(f"Saved {a.resolve()}")
    print(f"Saved {b.resolve()}")
    print("Within/across components partition the full-pool 10-NN overlap.")


if __name__ == "__main__":
    main()
