#!/usr/bin/env python3
"""Check whether SIGReg's L9->L12 mutual 10-NN change is broad across relations.

Compute each example's mutual 10-NN overlap over the full benchmark,
using the metric from Table 2. Aggregate the resulting scores by relation.

Requires Python 3.9+ and numpy. Example:
  py -m pip install numpy
  py check_jepa_relations.py --benchmark mlama_quad_parallel_benchmark.csv \
       --cache maktzgls__bert-jepa-sigreg-opus100-5lang-sigreg.npz

Outputs two small CSVs:
  jepa_sigreg_relation_shifts.csv
  jepa_sigreg_overall_checks.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

EXPECTED_PUBLISHED = {
    ("cls", "ar"): -11.91,
    ("cls", "zh"): -6.42,
    ("cls", "fr"): -12.81,
    ("mean", "ar"): -16.85,
    ("mean", "zh"): -12.22,
    ("mean", "fr"): -20.79,
}


def read_relations(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "relation" not in reader.fieldnames:
            raise ValueError("Benchmark CSV is missing the required 'relation' column.")
        relations = [row["relation"] for row in reader]
    if not relations or any(not rel for rel in relations):
        raise ValueError("Benchmark contains no rows or contains an empty relation.")
    return np.array(relations)


def global_mutual_10nn_per_example(z_lang: np.ndarray, z_en: np.ndarray, k: int = 10) -> np.ndarray:
    """Match repository geometry.py: centered, normalized, global 10-NN overlap."""
    n = len(z_lang)
    if z_en.shape != z_lang.shape:
        raise ValueError(f"Language/English representation shape mismatch: {z_lang.shape}, {z_en.shape}")
    if n <= k:
        raise ValueError(f"Need more than {k} benchmark rows for mutual {k}-NN, got {n}.")

    def neighbors(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        z = z - z.mean(axis=0, keepdims=True)
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)
        sims = z @ z.T
        np.fill_diagonal(sims, -np.inf)
        return np.argsort(-sims, axis=1)[:, :k]

    nn_lang = neighbors(z_lang)
    nn_en = neighbors(z_en)
    return np.array([
        len(set(nn_lang[i]).intersection(nn_en[i])) / float(k) * 100.0
        for i in range(n)
    ], dtype=np.float64)


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--benchmark", required=True, type=Path, help="mlama_quad_parallel_benchmark.csv")
    p.add_argument("--cache", required=True, type=Path, help="SIGReg .npz cache file from Google Drive")
    p.add_argument("--out-dir", type=Path, default=Path("."), help="Directory for the small CSV outputs")
    args = p.parse_args()

    if not args.benchmark.is_file():
        p.error(f"Benchmark file not found: {args.benchmark}")
    if not args.cache.is_file():
        p.error(f"Cache file not found: {args.cache}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    relations = read_relations(args.benchmark)
    unique_rel, rel_indices = np.unique(relations, return_inverse=True)
    if len(unique_rel) != 25:
        print(f"NOTE: Expected 25 unique relations from the README, found {len(unique_rel)}.")

    detail_rows = []
    overall_rows = []
    with np.load(args.cache, allow_pickle=False) as data:
        print(f"Loaded cache: {args.cache.name}")
        print(f"Found keys: {', '.join(data.files)}")
        for pooling in ("cls", "mean"):
            english_key = f"en_{pooling}"
            if english_key not in data.files:
                print(f"Skipping {pooling}: missing {english_key}")
                continue
            en = data[english_key]
            if en.ndim != 3 or en.shape[0] <= 12 or en.shape[1] != len(relations):
                raise ValueError(f"{english_key} expected shape (>=13, {len(relations)}, hidden), got {en.shape}")

            for lang in ("ar", "zh", "fr"):
                key = f"{lang}_{pooling}"
                if key not in data.files:
                    print(f"Skipping {key}: not found in cache")
                    continue
                z = data[key]
                if z.shape != en.shape:
                    raise ValueError(f"Cache keys {english_key}/{key} have different shapes: {en.shape} vs {z.shape}")

                score_l9 = global_mutual_10nn_per_example(z[9], en[9])
                score_l12 = global_mutual_10nn_per_example(z[12], en[12])
                delta = score_l12 - score_l9
                pub = EXPECTED_PUBLISHED[(pooling, lang)]
                observed = float(delta.mean())
                matches = abs(observed - pub) < 0.15
                overall_rows.append({
                    "model": "SIGReg 5-language OPUS100", "pooling": pooling, "language_pair": f"{lang.upper()}-EN",
                    "n_examples": len(delta), "n_relations": len(unique_rel),
                    "l9_global_mutual_10nn_pct": round(float(score_l9.mean()), 5),
                    "l12_global_mutual_10nn_pct": round(float(score_l12.mean()), 5),
                    "delta_l12_minus_l9_pp": round(observed, 5),
                    "readme_delta_pp": pub, "matches_readme_within_0_15pp": matches,
                })
                print(f"{pooling:4s} {lang.upper()}-EN: global shift {observed:+.2f} pp; README {pub:+.2f} pp; matches={matches}")

                for rid, relation in enumerate(unique_rel):
                    ids = rel_indices == rid
                    diff = delta[ids]
                    detail_rows.append({
                        "model": "SIGReg 5-language OPUS100",
                        "pooling": pooling, "language_pair": f"{lang.upper()}-EN",
                        "relation": relation, "n_examples": int(ids.sum()),
                        "l9_global_mutual_10nn_pct": round(float(score_l9[ids].mean()), 5),
                        "l12_global_mutual_10nn_pct": round(float(score_l12[ids].mean()), 5),
                        "delta_l12_minus_l9_pp": round(float(diff.mean()), 5),
                        "pct_examples_negative_delta": round(float(np.mean(diff < 0) * 100), 3),
                    })

    if not detail_rows:
        raise ValueError("No usable language/pooling keys were found in the NPZ cache.")
    detail_path = args.out_dir / "jepa_sigreg_relation_shifts.csv"
    overall_path = args.out_dir / "jepa_sigreg_overall_checks.csv"
    write_csv(detail_path, list(detail_rows[0]), detail_rows)
    write_csv(overall_path, list(overall_rows[0]), overall_rows)
    print(f"\nSaved: {detail_path.resolve()}")
    print(f"Saved: {overall_path.resolve()}")
    print("Input requirement: benchmark and cache rows must correspond in order.")
    print("Relation means are computed from the full-pool 10-NN scores.")


if __name__ == "__main__":
    main()
