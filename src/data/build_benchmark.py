"""Build the strictly quad-parallel mLAMA benchmark (AR, EN, ZH, FR) with 1:1 cultural split."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.ontology import is_mena_entity

MLAMA_URL = "http://cistern.cis.lmu.de/mlama/mlama1.1.zip"
TARGET_LANGS = ("ar", "en", "zh", "fr")
EXCLUDED_RELATIONS = {"P27"}


def acquire_mlama(raw_dir: Path) -> Dict[str, Path]:
    """Download and extract official mLAMA v1.1 if needed and locate language directories."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "mlama1.1.zip"
    extract_dir = raw_dir / "mlama1.1"

    if not extract_dir.exists():
        if not zip_path.exists():
            print(f"Downloading mLAMA v1.1 from {MLAMA_URL} -> {zip_path} ...")
            urllib.request.urlretrieve(MLAMA_URL, zip_path)
        print(f"Extracting {zip_path} -> {raw_dir} ...")
        with zipfile.ZipFile(zip_path, "r") as ref:
            ref.extractall(raw_dir)

    lang_dirs: Dict[str, Path] = {}
    for lang in TARGET_LANGS:
        matches = sorted(
            Path(p) for p in glob.glob(str(raw_dir / "**" / lang), recursive=True) if os.path.isdir(p)
        )
        if not matches:
            raise FileNotFoundError(f"Could not locate language directory '{lang}' under {raw_dir}.")
        lang_dirs[lang] = matches[0]
    return lang_dirs


def is_valid_tpl(tpl: str) -> bool:
    """Return True if [X] strictly precedes [Y]."""
    return bool("[X]" in tpl and "[Y]" in tpl and tpl.index("[X]") < tpl.index("[Y]"))


def build_quad_parallel_benchmark(
    raw_dir: Path | str = "data/raw",
    out_dir: Path | str = "data/processed",
    seed: int = 42,
    min_rel_count: int = 10,
) -> Path:
    """Curate strictly quad-parallel mLAMA statements and export mlama_quad_parallel_benchmark.csv."""
    raw_path = Path(raw_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    lang_dirs = acquire_mlama(raw_path)

    templates: Dict[str, Dict[str, str]] = defaultdict(dict)
    for lang in TARGET_LANGS:
        t_file = lang_dirs[lang] / "templates.jsonl"
        with open(t_file, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item["relation"] not in EXCLUDED_RELATIONS:
                    templates[lang][item["relation"]] = item["template"]

    valid_quad_relations = sorted(
        r
        for r in templates["ar"]
        if all(r in templates[l] and is_valid_tpl(templates[l][r]) for l in TARGET_LANGS)
    )

    quad_records: List[dict] = []
    for rel_id in valid_quad_relations:
        lang_facts: Dict[str, Dict[tuple, tuple]] = defaultdict(dict)
        for lang in TARGET_LANGS:
            f_path = lang_dirs[lang] / f"{rel_id}.jsonl"
            if f_path.exists():
                with open(f_path, "r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                            pair_key = (item["sub_uri"].strip(), item["obj_uri"].strip())
                            lang_facts[lang][pair_key] = (
                                item["sub_label"].strip(),
                                item["obj_label"].strip(),
                            )
                        except (KeyError, json.JSONDecodeError):
                            continue

        common_keys = sorted(
            set(lang_facts["ar"].keys())
            & set(lang_facts["en"].keys())
            & set(lang_facts["zh"].keys())
            & set(lang_facts["fr"].keys())
        )

        for pair_key in common_keys:
            sub_ar, obj_ar = lang_facts["ar"][pair_key]
            sub_en, obj_en = lang_facts["en"][pair_key]
            sub_zh, obj_zh = lang_facts["zh"][pair_key]
            sub_fr, obj_fr = lang_facts["fr"][pair_key]

            if any("(" in o or not o for o in (obj_ar, obj_en, obj_zh, obj_fr)):
                continue
            if (
                len(obj_ar.split()) > 2
                or len(obj_en.split()) > 2
                or len(obj_fr.split()) > 2
                or len(obj_zh) > 4
            ):
                continue

            stmt_ar = templates["ar"][rel_id].replace("[X]", sub_ar).replace("[Y]", obj_ar).strip()
            stmt_en = templates["en"][rel_id].replace("[X]", sub_en).replace("[Y]", obj_en).strip()
            stmt_zh = templates["zh"][rel_id].replace("[X]", sub_zh).replace("[Y]", obj_zh).strip()
            stmt_fr = templates["fr"][rel_id].replace("[X]", sub_fr).replace("[Y]", obj_fr).strip()

            is_local = is_mena_entity(sub_ar) or is_mena_entity(obj_ar)
            quad_records.append(
                {
                    "sub_qid": pair_key[0],
                    "obj_qid": pair_key[1],
                    "relation": rel_id,
                    "domain_category": "Local (Arab/MENA)" if is_local else "Global",
                    "stmt_ar": stmt_ar,
                    "stmt_en": stmt_en,
                    "stmt_zh": stmt_zh,
                    "stmt_fr": stmt_fr,
                }
            )

    df_full_quad = (
        pd.DataFrame(quad_records)
        .drop_duplicates(subset=["sub_qid", "obj_qid", "relation"])
        .drop_duplicates(subset=["stmt_ar"])
        .drop_duplicates(subset=["stmt_en"])
        .drop_duplicates(subset=["stmt_zh"])
        .drop_duplicates(subset=["stmt_fr"])
        .reset_index(drop=True)
    )

    df_local_pool = df_full_quad[df_full_quad["domain_category"] == "Local (Arab/MENA)"]
    df_global_pool = df_full_quad[df_full_quad["domain_category"] == "Global"]

    sampled_global = []
    for rel, grp in df_local_pool.groupby("relation", sort=True):
        cands = df_global_pool[df_global_pool["relation"] == rel]
        take_n = min(len(cands), max(len(grp) * 2, 15))
        if take_n > 0:
            sampled_global.append(cands.sample(n=take_n, random_state=seed))

    df_benchmark = (
        pd.concat([df_local_pool, *sampled_global], ignore_index=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )

    rel_counts = df_benchmark["relation"].value_counts()
    valid_rels = rel_counts[rel_counts >= min_rel_count].index
    df_benchmark = df_benchmark[df_benchmark["relation"].isin(valid_rels)].reset_index(drop=True)

    # Exact 1:1 relation-matched Global vs. Local cultural audit split
    rng_match = np.random.RandomState(seed)
    matched_global_idx: List[int] = []
    matched_local_idx: List[int] = []
    for rel, grp_loc in df_benchmark[df_benchmark["domain_category"] == "Local (Arab/MENA)"].groupby(
        "relation", sort=True
    ):
        grp_glb = df_benchmark[(df_benchmark["domain_category"] == "Global") & (df_benchmark["relation"] == rel)]
        m_n = min(len(grp_loc), len(grp_glb))
        if m_n > 0:
            matched_local_idx.extend(rng_match.choice(grp_loc.index.values, size=m_n, replace=False))
            matched_global_idx.extend(rng_match.choice(grp_glb.index.values, size=m_n, replace=False))

    df_benchmark["in_size_matched_cultural_eval"] = False
    df_benchmark.loc[matched_local_idx + matched_global_idx, "in_size_matched_cultural_eval"] = True

    out_csv = out_path / "mlama_quad_parallel_benchmark.csv"
    df_benchmark.to_csv(out_csv, index=False)

    n_total = len(df_benchmark)
    n_glb = int((df_benchmark["domain_category"] == "Global").sum())
    n_loc = int((df_benchmark["domain_category"] == "Local (Arab/MENA)").sum())
    print(
        f"Saved quad-parallel benchmark -> {out_csv}\n"
        f"Quad-Parallel N={n_total} (R={df_benchmark['relation'].nunique()}) | "
        f"Global={n_glb}, Local={n_loc} | Matched 1:1 Split={len(matched_local_idx)} per domain"
    )
    return out_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build quad-parallel mLAMA benchmark.")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--out_dir", type=str, default="data/processed")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_quad_parallel_benchmark(raw_dir=args.raw_dir, out_dir=args.out_dir, seed=args.seed)
