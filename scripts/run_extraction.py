#!/usr/bin/env python3
"""CLI runner to build the quad-parallel benchmark (if missing) and cache model layers.

Examples:
    python scripts/run_extraction.py
    python scripts/run_extraction.py --models M1_XLMR_CLS M4_JEPA_OPUS_SIGREG5
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from src.data.build_benchmark import build_quad_parallel_benchmark
from src.extraction.extract import extract_and_cache_all_models, load_models_registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract and cache multi-layer representations.")
    parser.add_argument("--config", type=str, default="configs/models.yaml")
    parser.add_argument("--benchmark", type=str, default="data/processed/mlama_quad_parallel_benchmark.csv")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--cache_dir", type=str, default="data/cache")
    parser.add_argument("--models", nargs="+", default=None, help="Optional subset of model IDs to extract.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true", help="Re-extract even if cache file exists.")
    parser.add_argument(
        "--only_configured_pooling",
        action="store_true",
        help="Save only the configured pooling instead of both [CLS] and MeanPool.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    benchmark_path = Path(args.benchmark)
    if not benchmark_path.exists():
        benchmark_path = build_quad_parallel_benchmark(
            raw_dir=args.raw_dir,
            out_dir=benchmark_path.parent,
            seed=args.seed,
        )

    df_benchmark = pd.read_csv(benchmark_path)
    registry = load_models_registry(args.config)

    if args.models:
        wanted = set(args.models)
        registry = [cfg for cfg in registry if cfg["id"] in wanted]
        if not registry:
            raise ValueError(f"None of the requested model IDs were found in {args.config}: {args.models}")

    extract_and_cache_all_models(
        models_registry=registry,
        df_benchmark=df_benchmark,
        cache_dir=args.cache_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
        overwrite=args.overwrite,
        save_both_poolings=not args.only_configured_pooling,
    )


if __name__ == "__main__":
    main()
