#!/usr/bin/env python3
"""Evaluate remaining checkpoints (SIGReg E10, InfoNCE 0.1 E5, 15-lang, XLM-R, mE5) on NTREX-128
at max_length=256 and merge with the 5 already-completed checkpoints in ntrex_validation/.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path
from typing import Dict

import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
EXT_ROOT = SCRIPT_DIR.parent
REPO_ROOT = EXT_ROOT.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common_eval import (
    CHECKPOINT_SPECS,
    atomic_csv,
    compute_layer_shifts,
    evaluate_single_checkpoint,
)

# Pinned NTREX-128 source revision and Git blob SHA-1 hashes matching ntrex_validation/
NTREX_REV = "468c6b69c7f6a75d31d4743d9daba2af566cc18d"
NTREX_FILES = {
    "en": ("newstest2019-src.eng.txt", "ebc28ab1a5b3561d7e6fee2173441a7d2bfa1edd"),
    "ar": ("newstest2019-ref.arb.txt", "bfe5e9fd9f1f89363ff5320a83a487cddf8412e5"),
    "zh": ("newstest2019-ref.zho-CN.txt", "20b794983bd59c10adb16714f9a7d880e8a8fba1"),
    "fr": ("newstest2019-ref.fra.txt", "74d0fa25e19f64580ef1d75d6188bdcd354c5b37"),
}

TARGET_NEW_CKPTS = {
    "sigreg_01_e10",
    "infonce_01_e5",
    "infonce_1_e1",
    "flores_mono15",
    "flores_biling15",
    "opus_mono15",
    "opus_biling15",
    "xlmr_base",
    "me5_base",
}


def load_ntrex_dataframe() -> pd.DataFrame:
    cols: Dict[str, list[str]] = {}
    for lang, (fname, expected_blob) in NTREX_FILES.items():
        url = f"https://raw.githubusercontent.com/MicrosoftTranslator/NTREX/{NTREX_REV}/NTREX-128/{fname}"
        with urllib.request.urlopen(url, timeout=90) as resp:
            raw = resp.read()
        blob = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
        if blob != expected_blob:
            raise ValueError(f"NTREX blob SHA-1 mismatch for {fname}: {blob} != {expected_blob}")
        lines = raw.decode("utf-8-sig").splitlines()
        if len(lines) != 1997 or any(not x.strip() for x in lines):
            raise ValueError(f"Unexpected NTREX line count for {fname}: {len(lines)}")
        cols[f"stmt_{lang}"] = lines
    return pd.DataFrame(cols)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out_dir",
        type=Path,
        default=EXT_ROOT / "results" / "ntrex_full",
    )
    p.add_argument(
        "--existing_ntrex",
        type=Path,
        default=REPO_ROOT / "ntrex_validation" / "results" / "max_length_256" / "ntrex_metrics.csv",
    )
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("Warning: Running on CPU; GPU is recommended.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df_ntrex = load_ntrex_dataframe()
    print(f"Verified NTREX-128: {len(df_ntrex)} parallel news sentences.")

    spec_by_id = {s["checkpoint"]: s for s in CHECKPOINT_SPECS}
    selected_specs = [s for s in CHECKPOINT_SPECS if s["checkpoint"] in TARGET_NEW_CKPTS]

    new_frames = []
    for spec in selected_specs:
        ckpt = spec["checkpoint"]
        ckpt_csv = args.out_dir / f"{ckpt}_metrics.csv"
        if ckpt_csv.exists():
            print(f"[Cached] Skipping {ckpt} -> {ckpt_csv}")
            new_frames.append(pd.read_csv(ckpt_csv))
        else:
            m_df, _ = evaluate_single_checkpoint(
                spec=spec,
                df=df_ntrex,
                out_dir=args.out_dir,
                max_length=args.max_length,
                batch_size=args.batch_size,
                device=device,
            )
            new_frames.append(m_df)

    all_frames = list(new_frames)
    if args.existing_ntrex.exists():
        df_prev = pd.read_csv(args.existing_ntrex)
        df_prev["group"] = df_prev["checkpoint"].map(
            lambda c: spec_by_id[c]["group"] if c in spec_by_id else "existing"
        )
        # Keep only checkpoints not re-run in this pass
        df_prev = df_prev[~df_prev["checkpoint"].isin(TARGET_NEW_CKPTS)].copy()
        all_frames.insert(0, df_prev)
        print(f"Merged {df_prev['checkpoint'].nunique()} existing NTREX checkpoints from {args.existing_ntrex}")

    df_all = pd.concat(all_frames, ignore_index=True)
    df_shifts = compute_layer_shifts(df_all)

    atomic_csv(df_all, args.out_dir / "ntrex_all_metrics.csv")
    atomic_csv(df_shifts, args.out_dir / "ntrex_all_layer_shifts.csv")

    print("\n" + "=" * 110)
    print("COMPLETE NTREX-128 LAYER SHIFTS (L9 -> L12, max_length=256)")
    print("=" * 110)
    cols = ["checkpoint", "pooling", "language_pair", "knn_pct_l9", "knn_pct_l12", "knn_pct_shift", "rank_src_shift"]
    print(df_shifts[cols].round(2).to_string(index=False))
    print(f"\nSaved -> {args.out_dir / 'ntrex_all_layer_shifts.csv'}")


if __name__ == "__main__":
    main()
