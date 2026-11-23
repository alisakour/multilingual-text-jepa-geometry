"""Statistical benchmarking across all layers and tables (Tables 1-5 with BH-FDR)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.analysis.geometry import compute_instance_level_metrics
from src.extraction.extract import load_cached_layers

LANG_PAIRS_CFG: Sequence[Tuple[str, str]] = (
    ("ar", "AR <-> EN (Semitic)"),
    ("zh", "ZH <-> EN (Logographic)"),
    ("fr", "FR <-> EN (Latin)"),
)

BASELINE_MODEL_ID = "M1_XLMR_CLS"
JEPA_TABLE2_IDS = (
    "M2_JEPA_FLORES_MONO15",
    "M2_JEPA_FLORES_BILING15",
    "M3_JEPA_OPUS_MONO15",
    "M3_JEPA_OPUS_BILING15",
    "M4_JEPA_OPUS_INFO5",
    "M4_JEPA_OPUS_SIGREG5",
)
FACTORIAL_PAIRS = (
    ("FLORES (15-Lang)", "M2_JEPA_FLORES_MONO15", "M2_JEPA_FLORES_BILING15"),
    ("OPUS100 (15-Lang)", "M3_JEPA_OPUS_MONO15", "M3_JEPA_OPUS_BILING15"),
)


def _safe_wilcoxon(diff: np.ndarray) -> float:
    nonzero = diff[np.abs(diff) > 1e-12]
    if len(nonzero) < 2:
        return 1.0
    try:
        _, p_val = stats.wilcoxon(nonzero)
        return float(p_val)
    except ValueError:
        return 1.0


def _apply_bh_fdr(raw_pvals: Sequence[float]) -> np.ndarray:
    arr = np.asarray(raw_pvals, dtype=float)
    fdr = np.ones_like(arr, dtype=float)
    valid = np.isfinite(arr)
    if valid.any():
        _, corrected, _, _ = multipletests(arr[valid], alpha=0.05, method="fdr_bh")
        fdr[valid] = corrected
    return fdr


def _fmt_fdr(p: float, is_base: bool = False) -> str:
    if is_base:
        return " (Base)"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return " ns"


def run_full_statistical_suite(
    models_registry: Sequence[Dict[str, Any]],
    df_benchmark: pd.DataFrame,
    cache_dir: Path | str = "data/cache",
    out_dir: Path | str = "results",
    k: int = 10,
    n_folds: int = 5,
    seed: int = 42,
    pooling_override: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """Compute layer trajectories and Tables 1-5 from cached .npz representations."""
    out_path = Path(out_dir)
    tables_dir = out_path / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    id_to_cfg = {cfg["id"]: cfg for cfg in models_registry}
    base_id = BASELINE_MODEL_ID if BASELINE_MODEL_ID in id_to_cfg else models_registry[0]["id"]

    full_traj_records: List[Dict[str, Any]] = []
    instance_store: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    key_layers_cache: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {}
    model_layer_meta: Dict[str, Tuple[int, int]] = {}  # model_id -> (l_pre, l_final)

    print(f"Running layer-wise geometric evaluation across {len(models_registry)} model configurations...")
    for cfg in models_registry:
        m_id = cfg["id"]
        m_name = cfg["name"]
        m_short = cfg["short"]

        cached = load_cached_layers(cfg, cache_dir=cache_dir, pooling_override=pooling_override)
        num_layers = int(cached["en"].shape[0] - 1)
        l_final = num_layers
        l_pre = int(round(num_layers * 0.75))  # Layer 9 for 12-layer backbones
        model_layer_meta[m_id] = (l_pre, l_final)

        # Keep only l_pre and l_final in memory for Tables 4 and 5 (low RAM footprint)
        key_layers_cache[m_id] = {
            lang: {l_pre: cached[lang][l_pre], l_final: cached[lang][l_final]}
            for lang in ("en", "ar", "zh", "fr")
        }

        z_en_layers = cached["en"]
        for lang_code, lang_label in LANG_PAIRS_CFG:
            l_key = lang_code.upper()
            z_src_layers = cached[lang_code]

            for l_idx in range(1, num_layers + 1):
                res = compute_instance_level_metrics(
                    z_src_layers[l_idx], z_en_layers[l_idx], k=k, n_folds=n_folds, seed=seed
                )
                full_traj_records.append(
                    {
                        "Model_ID": m_id,
                        "Model": m_name,
                        "Model_Short": m_short,
                        "Group": cfg["group"],
                        "Lang_Code": l_key,
                        "Language_Pair": lang_label,
                        "Layer": l_idx,
                        "Rel_Depth": round((l_idx / float(num_layers)) * 100.0, 1),
                        "KNN_Mean": float(np.mean(res["knn_i"])),
                        "KNN_SEM": float(stats.sem(res["knn_i"])),
                        "Unrot_Cos_Mean": float(np.mean(res["unrot_cos_i"])),
                        "Unrot_Cos_SEM": float(stats.sem(res["unrot_cos_i"])),
                        "Rot_Cos_Mean": float(np.mean(res["rot_cos_i"])),
                        "Rot_Cos_SEM": float(stats.sem(res["rot_cos_i"])),
                        "Rot_Gain_Mean": float(np.mean(res["rot_gain_i"])),
                        "Unrot_Top1": float(np.mean(res["unrot_top1_i"])),
                        "Rot_Top1": float(np.mean(res["rot_top1_i"])),
                        "Rot_Top5": float(np.mean(res["rot_top5_i"])),
                        "Eff_Rank_Src": float(res["erank_src"]),
                        "Eff_Rank_EN": float(res["erank_tgt"]),
                    }
                )
                if l_idx in (l_pre, l_final):
                    instance_store[(m_id, l_key, l_idx)] = res

        print(f"  [Done] {m_short} (L1..L{num_layers})")

    df_master_traj = pd.DataFrame(full_traj_records)
    df_master_traj.to_csv(tables_dir / "df_jepa_platonic_layer_trajectories.csv", index=False)

    # ------------------------------------------------------------------
    # TABLE 1: Terminal-Layer Platonic Benchmark vs. XLM-R [CLS] Baseline
    # ------------------------------------------------------------------
    t1_raw: List[Dict[str, Any]] = []
    non_base_idx: List[int] = []
    raw_p_knn: List[float] = []
    raw_p_rot: List[float] = []

    _, base_l_final = model_layer_meta[base_id]
    for lang_code, lang_label in LANG_PAIRS_CFG:
        l_key = lang_code.upper()
        base_res = instance_store[(base_id, l_key, base_l_final)]

        for cfg in models_registry:
            m_id = cfg["id"]
            _, m_l_final = model_layer_meta[m_id]
            curr_res = instance_store[(m_id, l_key, m_l_final)]
            is_base = m_id == base_id

            row_idx = len(t1_raw)
            t1_raw.append(
                {
                    "Language Pair": lang_label,
                    "Model": cfg["name"],
                    "knn_mean": float(np.mean(curr_res["knn_i"])),
                    "knn_sem": float(stats.sem(curr_res["knn_i"])),
                    "unrot_mean": float(np.mean(curr_res["unrot_cos_i"])),
                    "unrot_sem": float(stats.sem(curr_res["unrot_cos_i"])),
                    "rot_mean": float(np.mean(curr_res["rot_cos_i"])),
                    "rot_sem": float(stats.sem(curr_res["rot_cos_i"])),
                    "gain_mean": float(np.mean(curr_res["rot_gain_i"])),
                    "top1_rot": float(np.mean(curr_res["rot_top1_i"])),
                    "top5_rot": float(np.mean(curr_res["rot_top5_i"])),
                    "erank": float(curr_res["erank_src"]),
                    "is_base": is_base,
                }
            )
            if not is_base:
                non_base_idx.append(row_idx)
                raw_p_knn.append(_safe_wilcoxon(curr_res["knn_i"] - base_res["knn_i"]))
                raw_p_rot.append(_safe_wilcoxon(curr_res["rot_cos_i"] - base_res["rot_cos_i"]))

    fdr_knn_map = dict(zip(non_base_idx, _apply_bh_fdr(raw_p_knn)))
    fdr_rot_map = dict(zip(non_base_idx, _apply_bh_fdr(raw_p_rot)))

    df_table1 = pd.DataFrame(
        [
            {
                "Language Pair": r["Language Pair"],
                "Model": r["Model"],
                "Mutual 10-NN (%) [FDR]": f"{r['knn_mean']:.2f} ± {r['knn_sem']:.2f}{_fmt_fdr(fdr_knn_map.get(i, 1.0), r['is_base'])}",
                "Unrotated Cos (I)": f"{r['unrot_mean']:.3f} ± {r['unrot_sem']:.3f}",
                "Procrustes Cos (R*) [FDR]": f"{r['rot_mean']:.3f} ± {r['rot_sem']:.3f}{_fmt_fdr(fdr_rot_map.get(i, 1.0), r['is_base'])}",
                "Rotation Gain (ΔCos)": f"{r['gain_mean']:+.3f}",
                "5-Fold Top-1 / Top-5 (R*)": f"{r['top1_rot']:.1f}% / {r['top5_rot']:.1f}%",
                "Eff. Spectral Rank": f"{r['erank']:.1f}",
            }
            for i, r in enumerate(t1_raw)
        ]
    )
    df_table1.to_csv(tables_dir / "table1_jepa_platonic_benchmark.csv", index=False)

    # ------------------------------------------------------------------
    # TABLE 2: Pre-Terminal (L9) -> Terminal (L12) Shift Across JEPA Models
    # ------------------------------------------------------------------
    t2_raw = []
    t2_pk: List[float] = []
    t2_pr: List[float] = []
    for lang_code, _ in LANG_PAIRS_CFG:
        l_key = lang_code.upper()
        for m_id in JEPA_TABLE2_IDS:
            if m_id not in id_to_cfg:
                continue
            l_pre, l_final = model_layer_meta[m_id]
            r_pre = instance_store[(m_id, l_key, l_pre)]
            r_fin = instance_store[(m_id, l_key, l_final)]
            diff_knn = r_fin["knn_i"] - r_pre["knn_i"]
            diff_rot = r_fin["rot_cos_i"] - r_pre["rot_cos_i"]
            t2_pk.append(_safe_wilcoxon(diff_knn))
            t2_pr.append(_safe_wilcoxon(diff_rot))
            t2_raw.append((l_key, id_to_cfg[m_id]["name"], r_pre, r_fin, diff_knn, diff_rot))

    fdr_t2_k = _apply_bh_fdr(t2_pk)
    fdr_t2_r = _apply_bh_fdr(t2_pr)
    df_table2 = pd.DataFrame(
        [
            {
                "Lang": l_key,
                "Model": m_name.split(". ", 1)[-1],
                "L9 Mutual 10-NN": f"{np.mean(r_pre['knn_i']):.2f}%",
                "L12 Mutual 10-NN": f"{np.mean(r_fin['knn_i']):.2f}%",
                "Shift Δ(L12 - L9) [10-NN]": f"{np.mean(diff_knn):+.2f}% ± {stats.sem(diff_knn):.2f}%{_fmt_fdr(fdr_t2_k[i])}",
                "L9 -> L12 RotCos (R*)": f"{np.mean(r_pre['rot_cos_i']):.3f} -> {np.mean(r_fin['rot_cos_i']):.3f}",
                "Shift Δ(L12 - L9) [RotCos]": f"{np.mean(diff_rot):+.3f} ± {stats.sem(diff_rot):.3f}{_fmt_fdr(fdr_t2_r[i])}",
                "Eff. Rank (L9 -> L12)": f"{r_pre['erank_src']:.1f} -> {r_fin['erank_src']:.1f}",
            }
            for i, (l_key, m_name, r_pre, r_fin, diff_knn, diff_rot) in enumerate(t2_raw)
        ]
    )
    df_table2.to_csv(tables_dir / "table2_sigreg_terminal_shattering.csv", index=False)

    # ------------------------------------------------------------------
    # TABLE 3: 2x2 Monolingual vs. Bilingual Factorial (FLORES & OPUS100)
    # ------------------------------------------------------------------
    t3_raw = []
    t3_pk: List[float] = []
    t3_pc: List[float] = []
    for corpus_label, mono_id, biling_id in FACTORIAL_PAIRS:
        if mono_id not in id_to_cfg or biling_id not in id_to_cfg:
            continue
        _, l_fin_m = model_layer_meta[mono_id]
        _, l_fin_b = model_layer_meta[biling_id]
        for lang_code, lang_label in LANG_PAIRS_CFG:
            l_key = lang_code.upper()
            r_mono = instance_store[(mono_id, l_key, l_fin_m)]
            r_bili = instance_store[(biling_id, l_key, l_fin_b)]
            diff_k = r_mono["knn_i"] - r_bili["knn_i"]
            diff_c = r_mono["rot_cos_i"] - r_bili["rot_cos_i"]
            t3_pk.append(_safe_wilcoxon(diff_k))
            t3_pc.append(_safe_wilcoxon(diff_c))
            t3_raw.append((corpus_label, lang_label, r_mono, r_bili, diff_k, diff_c))

    fdr_t3_k = _apply_bh_fdr(t3_pk)
    fdr_t3_c = _apply_bh_fdr(t3_pc)
    df_table3 = pd.DataFrame(
        [
            {
                "Training Corpus": corpus_label,
                "Language Pair": lang_label,
                "Monolingual 10-NN (%)": f"{np.mean(r_mono['knn_i']):.2f}% ± {stats.sem(r_mono['knn_i']):.2f}%",
                "Bilingual 10-NN (%)": f"{np.mean(r_bili['knn_i']):.2f}% ± {stats.sem(r_bili['knn_i']):.2f}%",
                "Δ(Mono - Biling) [10-NN]": f"{np.mean(diff_k):+.2f}%{_fmt_fdr(fdr_t3_k[i])}",
                "Mono RotCos / Biling RotCos": f"{np.mean(r_mono['rot_cos_i']):.3f} / {np.mean(r_bili['rot_cos_i']):.3f} ({np.mean(diff_c):+.3f}{_fmt_fdr(fdr_t3_c[i])})",
                "Mono Erank / Biling Erank": f"{r_mono['erank_src']:.1f} / {r_bili['erank_src']:.1f}",
            }
            for i, (corpus_label, lang_label, r_mono, r_bili, diff_k, diff_c) in enumerate(t3_raw)
        ]
    )
    df_table3.to_csv(tables_dir / "table3_monolingual_vs_bilingual_factorial.csv", index=False)

    # ------------------------------------------------------------------
    # TABLE 4: Relation-Template Debiased Control (Train-Only Centroids)
    # ------------------------------------------------------------------
    relations_arr = df_benchmark["relation"].values
    t4_raw: List[Dict[str, Any]] = []
    t4_non_base_idx: List[int] = []
    t4_pk: List[float] = []
    t4_pr: List[float] = []
    debiased_store_fin: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for lang_code, _ in LANG_PAIRS_CFG:
        l_key = lang_code.upper()
        res_base_deb = compute_instance_level_metrics(
            key_layers_cache[base_id][lang_code][base_l_final],
            key_layers_cache[base_id]["en"][base_l_final],
            k=k,
            n_folds=n_folds,
            seed=seed,
            relations=relations_arr,
        )
        debiased_store_fin[(base_id, l_key)] = res_base_deb

        for cfg in models_registry:
            m_id = cfg["id"]
            l_pre, l_fin = model_layer_meta[m_id]
            is_base = m_id == base_id

            if (m_id, l_key) in debiased_store_fin:
                res_deb_fin = debiased_store_fin[(m_id, l_key)]
            else:
                res_deb_fin = compute_instance_level_metrics(
                    key_layers_cache[m_id][lang_code][l_fin],
                    key_layers_cache[m_id]["en"][l_fin],
                    k=k,
                    n_folds=n_folds,
                    seed=seed,
                    relations=relations_arr,
                )
                debiased_store_fin[(m_id, l_key)] = res_deb_fin

            res_deb_pre = compute_instance_level_metrics(
                key_layers_cache[m_id][lang_code][l_pre],
                key_layers_cache[m_id]["en"][l_pre],
                k=k,
                n_folds=n_folds,
                seed=seed,
                relations=relations_arr,
            )

            raw_top1 = float(np.mean(instance_store[(m_id, l_key, l_fin)]["rot_top1_i"]))
            deb_top1 = float(np.mean(res_deb_fin["rot_top1_i"]))

            row_idx = len(t4_raw)
            t4_raw.append(
                {
                    "Lang Pair": f"{l_key} <-> EN",
                    "Model": cfg["name"],
                    "deb_knn": float(np.mean(res_deb_fin["knn_i"])),
                    "deb_knn_sem": float(stats.sem(res_deb_fin["knn_i"])),
                    "deb_rot_cos": float(np.mean(res_deb_fin["rot_cos_i"])),
                    "deb_rot_sem": float(stats.sem(res_deb_fin["rot_cos_i"])),
                    "raw_top1": raw_top1,
                    "deb_top1": deb_top1,
                    "deb_top5": float(np.mean(res_deb_fin["rot_top5_i"])),
                    "deb_pre_knn": float(np.mean(res_deb_pre["knn_i"])),
                    "is_base": is_base,
                }
            )
            if not is_base:
                t4_non_base_idx.append(row_idx)
                t4_pk.append(_safe_wilcoxon(res_deb_fin["knn_i"] - res_base_deb["knn_i"]))
                t4_pr.append(_safe_wilcoxon(res_deb_fin["rot_cos_i"] - res_base_deb["rot_cos_i"]))

    fdr_t4_k_map = dict(zip(t4_non_base_idx, _apply_bh_fdr(t4_pk)))
    fdr_t4_r_map = dict(zip(t4_non_base_idx, _apply_bh_fdr(t4_pr)))

    df_table4 = pd.DataFrame(
        [
            {
                "Lang Pair": r["Lang Pair"],
                "Model": r["Model"],
                "Debiased 10-NN (L12) [FDR]": f"{r['deb_knn']:.2f}% ± {r['deb_knn_sem']:.2f}%{_fmt_fdr(fdr_t4_k_map.get(i, 1.0), r['is_base'])}",
                "Debiased RotCos (L12) [FDR]": f"{r['deb_rot_cos']:.3f} ± {r['deb_rot_sem']:.3f}{_fmt_fdr(fdr_t4_r_map.get(i, 1.0), r['is_base'])}",
                "Raw -> Debiased Top-1 (L12)": f"{r['raw_top1']:.1f}% -> {r['deb_top1']:.1f}% ({r['deb_top1'] - r['raw_top1']:+.1f}%)",
                "Debiased Top-5 (L12)": f"{r['deb_top5']:.1f}%",
                "Debiased L9 10-NN (Pre-Terminal)": f"{r['deb_pre_knn']:.2f}%",
            }
            for i, r in enumerate(t4_raw)
        ]
    )
    df_table4.to_csv(tables_dir / "table4_template_debiased_control.csv", index=False)

    # ------------------------------------------------------------------
    # TABLE 5: 1:1 Relation-Matched Cultural Domain Audit (Global vs. MENA)
    # ------------------------------------------------------------------
    domains_arr = df_benchmark["domain_category"].values
    matched_eval_mask = df_benchmark["in_size_matched_cultural_eval"].astype(bool).values

    loc_matched_idx = np.where((domains_arr == "Local (Arab/MENA)") & matched_eval_mask)[0]
    glb_matched_idx = np.where((domains_arr == "Global") & matched_eval_mask)[0]
    if len(loc_matched_idx) != len(glb_matched_idx):
        raise ValueError("Matched Global and Local subsets must have identical N.")
    n_matched = len(loc_matched_idx)

    t5_raw = []
    t5_p_dom: List[float] = []
    for cfg in models_registry:
        m_id = cfg["id"]
        _, l_fin = model_layer_meta[m_id]
        res_joint = instance_store[(m_id, "AR", l_fin)]
        z_ar_fin = key_layers_cache[m_id]["ar"][l_fin]
        z_en_fin = key_layers_cache[m_id]["en"][l_fin]

        res_glb_sub = compute_instance_level_metrics(
            z_ar_fin[glb_matched_idx], z_en_fin[glb_matched_idx], k=k, n_folds=n_folds, seed=seed
        )
        res_loc_sub = compute_instance_level_metrics(
            z_ar_fin[loc_matched_idx], z_en_fin[loc_matched_idx], k=k, n_folds=n_folds, seed=seed
        )
        _, p_dom = stats.mannwhitneyu(res_joint["knn_i"][glb_matched_idx], res_joint["knn_i"][loc_matched_idx])
        t5_p_dom.append(float(p_dom))
        t5_raw.append((cfg["short"], res_joint, res_glb_sub, res_loc_sub))

    fdr_t5_dom = _apply_bh_fdr(t5_p_dom)
    t5_rows: List[Dict[str, Any]] = []
    for idx, (m_short, res_joint, res_glb_sub, res_loc_sub) in enumerate(t5_raw):
        sig_dom = _fmt_fdr(fdr_t5_dom[idx])
        for dom_label, f_idx, r_sub in (
            ("Global (1:1 Relation-Matched)", glb_matched_idx, res_glb_sub),
            ("Local (Arab/MENA Ontology)", loc_matched_idx, res_loc_sub),
        ):
            t5_rows.append(
                {
                    "Model": m_short,
                    "Cultural Domain": dom_label,
                    "Matched N": n_matched,
                    f"Joint-Pool 10-NN (N={len(df_benchmark)})": f"{np.mean(res_joint['knn_i'][f_idx]):.2f}% ± {stats.sem(res_joint['knn_i'][f_idx]):.2f}%",
                    "Joint-Pool RotCos (R*)": f"{np.mean(res_joint['rot_cos_i'][f_idx]):.3f} ± {stats.sem(res_joint['rot_cos_i'][f_idx]):.3f}",
                    f"Matched-Pool 10-NN (N={n_matched})": f"{np.mean(r_sub['knn_i']):.2f}% ± {stats.sem(r_sub['knn_i']):.2f}%",
                    "Matched-Pool Top-1 / Top-5": f"{np.mean(r_sub['rot_top1_i']):.1f}% / {np.mean(r_sub['rot_top5_i']):.1f}%",
                    "Matched Eff. Rank (AR)": f"{r_sub['erank_src']:.1f}",
                    "Domain Gap Sig [FDR]": sig_dom if "Local" in dom_label else "—",
                }
            )

    df_table5 = pd.DataFrame(t5_rows)
    df_table5.to_csv(tables_dir / "table5_cultural_domain_breakdown.csv", index=False)

    return {
        "trajectories": df_master_traj,
        "table1": df_table1,
        "table2": df_table2,
        "table3": df_table3,
        "table4": df_table4,
        "table5": df_table5,
    }
