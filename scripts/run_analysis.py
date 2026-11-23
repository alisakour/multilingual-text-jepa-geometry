#!/usr/bin/env python3
"""CLI runner to compute Tables 1-5 (BH-FDR) and 300-DPI Figures 1-2 from cached layers.

Runs entirely on CPU once scripts/run_extraction.py has populated data/cache/.

Examples:
    python scripts/run_analysis.py --out_dir results
    python scripts/run_analysis.py --pooling_override mean --out_dir results_meanpool
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from src.analysis.plots import generate_all_figures
from src.analysis.stats import run_full_statistical_suite
from src.extraction.extract import load_models_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute geometric tables and publication figures.")
    parser.add_argument("--config", type=str, default="configs/models.yaml")
    parser.add_argument("--benchmark", type=str, default="data/processed/mlama_quad_parallel_benchmark.csv")
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--k", type=int, default=10, help="k for Mutual k-NN overlap.")
    parser.add_argument("--n_folds", type=int, default=5, help="Number of CV folds for Procrustes.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pooling_override",
        type=str,
        default=None,
        choices=["cls", "mean"],
        help="Optional pooling override (e.g., 'mean' to run a MeanPool ablation from cached layers).",
    )
    parser.add_argument("--no_plots", action="store_true", help="Skip generating PNG/PDF figures.")
    return parser.parse_args()


def _print_table(title: str, df: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    print(df.to_string(index=False) if not df.empty else "(Empty table)")


def main() -> None:
    args = parse_args()

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        raise FileNotFoundError(
            f"Benchmark CSV not found: {benchmark_path}. Run scripts/run_extraction.py first."
        )

    df_benchmark = pd.read_csv(benchmark_path)
    registry = load_models_registry(args.config)

    outputs = run_full_statistical_suite(
        models_registry=registry,
        df_benchmark=df_benchmark,
        cache_dir=args.cache_dir,
        out_dir=args.out_dir,
        k=args.k,
        n_folds=args.n_folds,
        seed=args.seed,
        pooling_override=args.pooling_override,
    )

    _print_table("TABLE 1: TERMINAL-LAYER PLATONIC BENCHMARK (VS. XLM-R [CLS], BH-FDR)", outputs["table1"])
    _print_table("TABLE 2: PRE-TERMINAL (L9) -> TERMINAL (L12) SHIFT ACROSS JEPA MODELS (BH-FDR)", outputs["table2"])
    _print_table("TABLE 3: 2x2 FACTORIAL — MONOLINGUAL VS. BILINGUAL PACKING (BH-FDR)", outputs["table3"])
    _print_table("TABLE 4: TEMPLATE-DEBIASED CONTROL (TRAIN-ONLY RELATION CENTROIDS, BH-FDR)", outputs["table4"])
    _print_table("TABLE 5: 1:1 RELATION-MATCHED CULTURAL AUDIT (GLOBAL VS. ARAB/MENA, BH-FDR)", outputs["table5"])

    if not args.no_plots:
        print("\nGenerating 300-DPI figures (PNG + PDF)...")
        generate_all_figures(outputs["trajectories"], outputs["table5"], out_dir=args.out_dir)

    print(f"\nAll outputs saved under: {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
