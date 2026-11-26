#!/usr/bin/env python3
"""Evaluate all 14 checkpoints/models on Amazon MASSIVE v1.1 (test split, N=2,974)
across ar-SA, en-US, zh-CN, and fr-FR, including within-intent vs across-intent decomposition.
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
EXT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_eval import (
    CHECKPOINT_SPECS,
    atomic_csv,
    compute_layer_shifts,
    evaluate_single_checkpoint,
    sha256_file,
)

MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
MASSIVE_TAR_SHA256 = "4cba5faa11c71437928e17cb1b9b3d8b8e727e7ea363a3a9a8045e19c0491577"
LOCALE_MAP = {
    "en": "en-US",
    "ar": "ar-SA",
    "zh": "zh-CN",
    "fr": "fr-FR",
}


def acquire_massive_dataframe(raw_dir: Path, partition: str = "test") -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    tar_path = raw_dir / "amazon-massive-dataset-1.1.tar.gz"
    if not tar_path.exists():
        print(f"Downloading MASSIVE v1.1 from {MASSIVE_URL} ...")
        urllib.request.urlretrieve(MASSIVE_URL, tar_path)

    obs_sha = sha256_file(tar_path)
    if obs_sha != MASSIVE_TAR_SHA256:
        raise ValueError(f"MASSIVE archive SHA-256 mismatch: {obs_sha} != {MASSIVE_TAR_SHA256}")

    by_lang: Dict[str, Dict[str, dict]] = {lang: {} for lang in LOCALE_MAP}
    with tarfile.open(tar_path, "r:gz") as tf:
        members = {m.name: m for m in tf.getmembers() if m.isfile()}
        for lang, locale in LOCALE_MAP.items():
            target_suffix = f"/{locale}.jsonl"
            matches = [name for name in members if name.endswith(target_suffix)]
            if not matches:
                raise FileNotFoundError(f"Could not find {locale}.jsonl inside {tar_path}")
            f_obj = tf.extractfile(members[matches[0]])
            assert f_obj is not None
            for raw_line in f_obj:
                item = json.loads(raw_line.decode("utf-8"))
                if item.get("partition") == partition:
                    by_lang[lang][str(item["id"])] = {
                        "utt": str(item["utt"]).strip(),
                        "intent": str(item["intent"]).strip(),
                        "scenario": str(item["scenario"]).strip(),
                    }

    common_ids = sorted(
        set.intersection(*(set(by_lang[l].keys()) for l in LOCALE_MAP)),
        key=lambda x: int(x) if x.isdigit() else x,
    )
    rows = []
    for uid in common_ids:
        intent = by_lang["en"][uid]["intent"]
        scenario = by_lang["en"][uid]["scenario"]
        if any(by_lang[l][uid]["intent"] != intent for l in LOCALE_MAP):
            raise ValueError(f"Intent mismatch across locales for id={uid}")
        rows.append(
            {
                "id": uid,
                "intent": intent,
                "scenario": scenario,
                "stmt_en": by_lang["en"][uid]["utt"],
                "stmt_ar": by_lang["ar"][uid]["utt"],
                "stmt_zh": by_lang["zh"][uid]["utt"],
                "stmt_fr": by_lang["fr"][uid]["utt"],
            }
        )
    return pd.DataFrame(rows)


def compute_decomp_shifts(decomp_df: pd.DataFrame) -> pd.DataFrame:
    idx = ["checkpoint", "group", "pooling", "language_pair"]
    l9 = decomp_df[decomp_df["layer"] == 9].set_index(idx)
    l12 = decomp_df[decomp_df["layer"] == 12].set_index(idx)
    cols = ["total_10nn_pct", "within_label_pct", "across_label_pct"]
    out = l9[cols].join(l12[cols], lsuffix="_l9", rsuffix="_l12")
    for c in cols:
        out[f"{c}_shift"] = out[f"{c}_l12"] - out[f"{c}_l9"]
    return out.reset_index()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw_dir", type=Path, default=EXT_ROOT / "data_raw")
    p.add_argument("--out_dir", type=Path, default=EXT_ROOT / "results" / "massive")
    p.add_argument("--partition", type=str, default="test", choices=["test", "dev", "train"])
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("Warning: Running on CPU; GPU is recommended.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df_massive = acquire_massive_dataframe(args.raw_dir, partition=args.partition)
    intents_arr = df_massive["intent"].values
    print(
        f"Verified MASSIVE v1.1 ({args.partition}): {len(df_massive)} parallel utterances, "
        f"{df_massive['intent'].nunique()} intents, {df_massive['scenario'].nunique()} scenarios."
    )

    metric_frames = []
    decomp_frames = []
    for spec in CHECKPOINT_SPECS:
        ckpt = spec["checkpoint"]
        m_csv = args.out_dir / f"{ckpt}_metrics.csv"
        d_csv = args.out_dir / f"{ckpt}_decomp.csv"
        if m_csv.exists() and d_csv.exists():
            print(f"[Cached] Skipping {ckpt} -> {m_csv}")
            metric_frames.append(pd.read_csv(m_csv))
            decomp_frames.append(pd.read_csv(d_csv))
        else:
            m_df, d_df = evaluate_single_checkpoint(
                spec=spec,
                df=df_massive,
                out_dir=args.out_dir,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=device,
                group_labels=intents_arr,
            )
            metric_frames.append(m_df)
            if d_df is not None:
                decomp_frames.append(d_df)

    df_all = pd.concat(metric_frames, ignore_index=True)
    df_shifts = compute_layer_shifts(df_all)
    df_decomp = pd.concat(decomp_frames, ignore_index=True)
    df_decomp_shifts = compute_decomp_shifts(df_decomp)

    atomic_csv(df_all, args.out_dir / "massive_all_metrics.csv")
    atomic_csv(df_shifts, args.out_dir / "massive_all_layer_shifts.csv")
    atomic_csv(df_decomp_shifts, args.out_dir / "massive_intent_decomposition.csv")

    print("\n" + "=" * 110)
    print("MASSIVE v1.1 LAYER SHIFTS (L9 -> L12, max_length=128)")
    print("=" * 110)
    cols = ["checkpoint", "pooling", "language_pair", "knn_pct_l9", "knn_pct_l12", "knn_pct_shift", "rank_src_shift"]
    print(df_shifts[cols].round(2).to_string(index=False))
    print(f"\nSaved -> {args.out_dir / 'massive_all_layer_shifts.csv'}")


if __name__ == "__main__":
    main()
