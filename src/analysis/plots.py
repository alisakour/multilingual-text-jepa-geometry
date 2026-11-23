"""Publication-grade (300 DPI PNG + PDF) figures for cross-script Text-JEPA geometry."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib

matplotlib.use("Agg")  # Safe headless rendering on Linux GPU clusters

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE_TRAJ: Dict[str, str] = {
    "XLM-R [CLS]": "#6a51a3",
    "XLM-R [MeanPool]": "#8c6d31",
    "JEPA Mono-15L (FLORES)": "#008837",
    "JEPA Mono-15L (OPUS)": "#66bd63",
    "JEPA 5L-InfoNCE (OPUS)": "#1f78b4",
    "JEPA 5L-SIGReg (OPUS)": "#e6550d",
}

MARKERS_TRAJ: Dict[str, str] = {
    "XLM-R [CLS]": "s",
    "XLM-R [MeanPool]": "D",
    "JEPA Mono-15L (FLORES)": "o",
    "JEPA Mono-15L (OPUS)": "^",
    "JEPA 5L-InfoNCE (OPUS)": "P",
    "JEPA 5L-SIGReg (OPUS)": "X",
}

WHITE_HALO = [pe.withStroke(linewidth=3.0, foreground="white")]


def _configure_style() -> None:
    sns.set_theme(style="ticks", font="sans-serif")
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.labelsize": 11.5,
            "axes.titlesize": 12.2,
            "xtick.labelsize": 10.0,
            "ytick.labelsize": 10.0,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 1.0,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_spaced_grouped_bars(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,
    x_order: Sequence[str],
    hue_order: Sequence[str],
    palette_dict: Dict[str, str],
    fmt: str = "{:.1f}",
    font_sz: float = 8.6,
) -> None:
    """Draw cleanly spaced grouped bar charts with white-halo value labels."""
    n_groups = len(x_order)
    n_hues = len(hue_order)
    total_group_width = 0.82
    slot_width = total_group_width / n_hues
    bar_width = slot_width * 0.84

    x_base = np.arange(n_groups)
    for h_idx, h_val in enumerate(hue_order):
        offsets = x_base - (total_group_width / 2.0) + (h_idx + 0.5) * slot_width
        vals = []
        for x_val in x_order:
            sub = df[(df[x_col] == x_val) & (df[hue_col] == h_val)]
            vals.append(float(sub[y_col].iloc[0]) if len(sub) > 0 else 0.0)

        bars = ax.bar(
            offsets,
            vals,
            width=bar_width,
            label=h_val,
            color=palette_dict[h_val],
            edgecolor="#222222",
            linewidth=0.9,
            zorder=3,
        )
        for rect, v in zip(bars, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height() + 0.65,
                fmt.format(v),
                ha="center",
                va="bottom",
                fontsize=font_sz,
                fontweight="bold",
                color="#111111",
                path_effects=WHITE_HALO,
                zorder=4,
            )

    ax.set_xticks(x_base)
    ax.set_xticklabels(x_order, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.45, color="#cccccc", zorder=0)
    sns.despine(ax=ax, top=True, right=True)


def plot_figure1_trajectories(df_master_traj: pd.DataFrame, figs_dir: Path) -> Path:
    """Figure 1 (1x3): Mutual 10-NN, Effective Spectral Rank, and Platonic Procrustes Alignment."""
    _configure_style()
    fig1, axes1 = plt.subplots(1, 3, figsize=(21, 5.6))

    df_ar_traj = df_master_traj[
        (df_master_traj["Lang_Code"] == "AR")
        & (df_master_traj["Model_Short"].isin(PALETTE_TRAJ.keys()))
    ].copy()
    max_layer = int(df_ar_traj["Layer"].max()) if not df_ar_traj.empty else 12
    pre_layer = int(round(max_layer * 0.75))

    # Panel A: Topological Alignment (Mutual 10-NN)
    ax = axes1[0]
    ax.axvspan(pre_layer - 1.0, max_layer - 0.8, color="#fff7bc", alpha=0.35, zorder=0)
    ax.axvline(pre_layer, color="#e6550d", linestyle="--", lw=1.5, alpha=0.85, zorder=1)
    for m_short, color in PALETTE_TRAJ.items():
        grp = df_ar_traj[df_ar_traj["Model_Short"] == m_short]
        if grp.empty:
            continue
        ax.plot(
            grp["Layer"],
            grp["KNN_Mean"],
            marker=MARKERS_TRAJ[m_short],
            lw=2.4,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=m_short,
            color=color,
            zorder=3,
        )
        ax.fill_between(
            grp["Layer"],
            grp["KNN_Mean"] - 1.96 * grp["KNN_SEM"],
            grp["KNN_Mean"] + 1.96 * grp["KNN_SEM"],
            color=color,
            alpha=0.10,
            zorder=2,
        )
    ax.text(
        pre_layer + 0.15,
        38.2,
        f"Pre-Terminal\nZone (L{pre_layer - 1}–L{max_layer - 1})",
        color="#b34000",
        fontsize=8.8,
        fontweight="bold",
        va="top",
        path_effects=WHITE_HALO,
    )
    ax.set_title("(A) Topological Alignment (Mutual 10-NN)", fontweight="bold", pad=12)
    ax.set_xlabel(f"Transformer Layer Depth (ℓ = 1 to {max_layer})")
    ax.set_ylabel("Mutual 10-NN Overlap (%) [AR ↔ EN]")
    ax.set_xticks(range(1, max_layer + 1))
    ax.set_ylim(7.5, 40.5)
    ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    sns.despine(ax=ax, top=True, right=True)

    # Panel B: Spectral Funneling vs. MLM Re-Expansion
    ax = axes1[1]
    ax.axvspan(pre_layer - 1.0, max_layer - 0.8, color="#fff7bc", alpha=0.35, zorder=0)
    ax.axvline(pre_layer, color="#e6550d", linestyle="--", lw=1.5, alpha=0.85, zorder=1)
    for m_short, color in PALETTE_TRAJ.items():
        grp = df_ar_traj[df_ar_traj["Model_Short"] == m_short]
        if grp.empty:
            continue
        ax.plot(
            grp["Layer"],
            grp["Eff_Rank_Src"],
            marker=MARKERS_TRAJ[m_short],
            lw=2.4,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=m_short,
            color=color,
            zorder=3,
        )

    xlmr_sub = df_ar_traj[(df_ar_traj["Model_Short"] == "XLM-R [CLS]") & (df_ar_traj["Layer"] == max_layer)]
    jepa_sub = df_ar_traj[(df_ar_traj["Model_Short"] == "JEPA Mono-15L (FLORES)") & (df_ar_traj["Layer"] == max_layer)]

    if not xlmr_sub.empty:
        xlmr_l12_rank = float(xlmr_sub["Eff_Rank_Src"].iloc[0])
        ax.annotate(
            f"MLM Lexical\nRe-Expansion ({xlmr_l12_rank:.0f}d)",
            xy=(max_layer - 0.1, xlmr_l12_rank),
            xytext=(max_layer - 3.8, 330),
            arrowprops=dict(arrowstyle="->", color="#6a51a3", lw=1.8, connectionstyle="arc3,rad=-0.12"),
            fontweight="bold",
            color="#6a51a3",
            fontsize=9.2,
            path_effects=WHITE_HALO,
        )
    if not jepa_sub.empty:
        jepa_l12_rank = float(jepa_sub["Eff_Rank_Src"].iloc[0])
        ax.annotate(
            f"15L-JEPA Spectral\nFunnel (~{jepa_l12_rank:.0f}d)",
            xy=(max_layer - 0.2, jepa_l12_rank),
            xytext=(4.3, 55),
            arrowprops=dict(arrowstyle="->", color="#008837", lw=1.8, connectionstyle="arc3,rad=0.12"),
            fontweight="bold",
            color="#008837",
            fontsize=9.2,
            path_effects=WHITE_HALO,
        )

    ax.set_title("(B) Spectral Funneling vs. MLM Re-Expansion", fontweight="bold", pad=12)
    ax.set_xlabel(f"Transformer Layer Depth (ℓ = 1 to {max_layer})")
    ax.set_ylabel(r"Effective Spectral Rank $\mathrm{Erank}(Z_\ell)$ [AR]")
    ax.set_xticks(range(1, max_layer + 1))
    ax.set_ylim(25, 460)
    ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    sns.despine(ax=ax, top=True, right=True)

    # Panel C: Platonic Alignment (R* vs. Identity I)
    ax = axes1[2]
    ax.axvspan(pre_layer - 1.0, max_layer - 0.8, color="#fff7bc", alpha=0.35, zorder=0)
    ax.axvline(pre_layer, color="#e6550d", linestyle="--", lw=1.5, alpha=0.85, zorder=1)
    for m_short in ("XLM-R [CLS]", "JEPA Mono-15L (FLORES)", "JEPA 5L-InfoNCE (OPUS)", "JEPA 5L-SIGReg (OPUS)"):
        color = PALETTE_TRAJ[m_short]
        grp = df_ar_traj[df_ar_traj["Model_Short"] == m_short]
        if grp.empty:
            continue
        ax.plot(
            grp["Layer"],
            grp["Rot_Cos_Mean"],
            marker=MARKERS_TRAJ[m_short],
            lw=2.5,
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.7,
            color=color,
            zorder=3,
        )
        ax.plot(
            grp["Layer"],
            grp["Unrot_Cos_Mean"],
            linestyle=":",
            lw=1.8,
            color=color,
            alpha=0.75,
            zorder=2,
        )
    ax.plot([], [], color="#333333", lw=2.3, linestyle="-", label=r"Rotated ($R^* \in O(d)$)")
    ax.plot([], [], color="#333333", lw=1.8, linestyle=":", label=r"Unrotated ($I$)")
    ax.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#cccccc", fontsize=9.2)

    ax.set_title(r"(C) Platonic Alignment ($R^* \in O(d)$ vs. Identity $I$)", fontweight="bold", pad=12)
    ax.set_xlabel(f"Transformer Layer Depth (ℓ = 1 to {max_layer})")
    ax.set_ylabel("Out-of-Sample Cosine Alignment [AR ↔ EN]")
    ax.set_xticks(range(1, max_layer + 1))
    ax.set_ylim(-0.08, 0.98)
    ax.grid(True, linestyle="--", alpha=0.4, color="#cccccc")
    sns.despine(ax=ax, top=True, right=True)

    handles, labels = axes1[0].get_legend_handles_labels()
    fig1.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.11),
        ncol=6,
        frameon=True,
        facecolor="#f8f9fa",
        edgecolor="#cccccc",
        fontsize=10.0,
        title="Model Architecture & Self-Supervised Objective (Shared 12-Layer XLM-R Backbone)",
        title_fontsize=10.5,
    )

    plt.subplots_adjust(wspace=0.24, top=0.84)
    png_path = figs_dir / "figure1_jepa_platonic_trajectories_1x3.png"
    pdf_path = figs_dir / "figure1_jepa_platonic_trajectories_1x3.pdf"
    fig1.savefig(png_path, dpi=300, bbox_inches="tight")
    fig1.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved Figure 1 -> {png_path} & {pdf_path}")
    return png_path


def plot_figure2_factorial_and_cultural(
    df_master_traj: pd.DataFrame,
    df_table5_domain: pd.DataFrame,
    figs_dir: Path,
) -> Path:
    """Figure 2 (1x3): 5-Lang Twins (L9/L12), 2x2 Mono vs. Bilingual Factorial, and 1:1 Cultural Audit."""
    _configure_style()
    fig2, axes2 = plt.subplots(1, 3, figsize=(21.5, 5.8))

    # Panel A: 5-Lang Twins InfoNCE vs. SIGReg (L9 & L12)
    ax = axes2[0]
    df_twins = df_master_traj[
        (df_master_traj["Model_Short"].isin(["JEPA 5L-InfoNCE (OPUS)", "JEPA 5L-SIGReg (OPUS)"]))
        & (df_master_traj["Layer"].isin([9, 12]))
    ].copy()
    df_twins["Twin_Stage"] = (
        df_twins["Model_Short"].str.replace(" (OPUS)", "", regex=False)
        + " (L"
        + df_twins["Layer"].astype(str)
        + ")"
    )
    twin_order = [
        "JEPA 5L-InfoNCE (L9)",
        "JEPA 5L-InfoNCE (L12)",
        "JEPA 5L-SIGReg (L9)",
        "JEPA 5L-SIGReg (L12)",
    ]
    twin_palette = {
        "JEPA 5L-InfoNCE (L9)": "#9ecae1",
        "JEPA 5L-InfoNCE (L12)": "#08519c",
        "JEPA 5L-SIGReg (L9)": "#fdb863",
        "JEPA 5L-SIGReg (L12)": "#d95f02",
    }
    plot_spaced_grouped_bars(
        ax,
        df_twins,
        "Lang_Code",
        "KNN_Mean",
        "Twin_Stage",
        x_order=["AR", "ZH", "FR"],
        hue_order=twin_order,
        palette_dict=twin_palette,
        fmt="{:.1f}",
        font_sz=8.6,
    )
    ax.set_title("(A) 5-Lang Twins: InfoNCE vs. SIGReg (L9 & L12)", fontweight="bold", pad=12)
    ax.set_xlabel("Source Language Paired with English")
    ax.set_ylabel("Mutual 10-NN Alignment (%)")
    ax.set_ylim(0, 53)
    ax.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        facecolor="#f8f9fa",
        edgecolor="#cccccc",
        fontsize=8.4,
        title="5-Lang OPUS100 Twin & Layer",
        title_fontsize=8.8,
    )

    # Panel B: 2x2 Factorial Mono vs. Bilingual (L12)
    ax = axes2[1]
    fact_order = [
        "JEPA Mono-15L (FLORES)",
        "JEPA Biling-15L (FLORES)",
        "JEPA Mono-15L (OPUS)",
        "JEPA Biling-15L (OPUS)",
    ]
    df_fact = df_master_traj[
        (df_master_traj["Model_Short"].isin(fact_order)) & (df_master_traj["Layer"] == 12)
    ].copy()
    fact_palette = {
        "JEPA Mono-15L (FLORES)": "#006d2c",
        "JEPA Biling-15L (FLORES)": "#74c476",
        "JEPA Mono-15L (OPUS)": "#08519c",
        "JEPA Biling-15L (OPUS)": "#6baed6",
    }
    plot_spaced_grouped_bars(
        ax,
        df_fact,
        "Lang_Code",
        "KNN_Mean",
        "Model_Short",
        x_order=["AR", "ZH", "FR"],
        hue_order=fact_order,
        palette_dict=fact_palette,
        fmt="{:.1f}",
        font_sz=7.5,
    )
    ax.set_title("(B) 2×2 Factorial: Mono vs. Bilingual (L12)", fontweight="bold", pad=12)
    ax.set_xlabel("Source Language Paired with English")
    ax.set_ylabel("Mutual 10-NN Alignment (%)")
    ax.set_ylim(0, 50)
    ax.legend(
        loc="upper left",
        ncol=2,
        frameon=True,
        facecolor="#f8f9fa",
        edgecolor="#cccccc",
        fontsize=8.1,
        title="15-Lang JEPA Packing & Corpus",
        title_fontsize=8.6,
    )

    # Panel C: 1:1 Cultural Audit Global vs. MENA
    ax = axes2[2]
    cult_models = [
        "XLM-R [CLS]",
        "XLM-R [MeanPool]",
        "JEPA Mono-15L (FLORES)",
        "JEPA Mono-15L (OPUS)",
        "JEPA 5L-InfoNCE (OPUS)",
    ]
    df_cult_plot = df_table5_domain[df_table5_domain["Model"].isin(cult_models)].copy()
    matched_col = [c for c in df_cult_plot.columns if c.startswith("Matched-Pool 10-NN")][0]
    n_matched_val = int(df_cult_plot["Matched N"].iloc[0])
    df_cult_plot["Matched_KNN_Val"] = df_cult_plot[matched_col].apply(lambda s: float(str(s).split("%")[0]))

    label_map_2line = {
        "XLM-R [CLS]": "XLM-R\n[CLS]",
        "XLM-R [MeanPool]": "XLM-R\n[MeanPool]",
        "JEPA Mono-15L (FLORES)": "Mono-15L\n(FLORES)",
        "JEPA Mono-15L (OPUS)": "Mono-15L\n(OPUS)",
        "JEPA 5L-InfoNCE (OPUS)": "5L-InfoNCE\n(OPUS)",
    }
    df_cult_plot["Model_2Line"] = df_cult_plot["Model"].map(label_map_2line)
    x_2line_order = [label_map_2line[m] for m in cult_models]

    dom_order = ["Global (1:1 Relation-Matched)", "Local (Arab/MENA Ontology)"]
    dom_palette = {
        "Global (1:1 Relation-Matched)": "#756bb1",
        "Local (Arab/MENA Ontology)": "#1b9e77",
    }
    plot_spaced_grouped_bars(
        ax,
        df_cult_plot,
        "Model_2Line",
        "Matched_KNN_Val",
        "Cultural Domain",
        x_order=x_2line_order,
        hue_order=dom_order,
        palette_dict=dom_palette,
        fmt="{:.1f}",
        font_sz=8.8,
    )
    ax.set_title(f"(C) 1:1 Cultural Audit: Global vs. MENA (N={n_matched_val})", fontweight="bold", pad=12)
    ax.set_xlabel("Model Architecture / Objective")
    ax.set_ylabel("Matched-Pool Mutual 10-NN (%) [AR ↔ EN]")
    ax.set_ylim(0, 56)
    ax.legend(
        loc="upper left",
        ncol=1,
        frameon=True,
        facecolor="#f8f9fa",
        edgecolor="#cccccc",
        fontsize=8.8,
        title=f"Entity Domain (N={n_matched_val} Each)",
        title_fontsize=9.0,
    )

    plt.subplots_adjust(wspace=0.26)
    png_path = figs_dir / "figure2_jepa_factorial_and_shattering_1x3.png"
    pdf_path = figs_dir / "figure2_jepa_factorial_and_shattering_1x3.pdf"
    fig2.savefig(png_path, dpi=300, bbox_inches="tight")
    fig2.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved Figure 2 -> {png_path} & {pdf_path}")
    return png_path


def generate_all_figures(
    df_master_traj: pd.DataFrame,
    df_table5_domain: pd.DataFrame,
    out_dir: Path | str = "results",
) -> None:
    """Generate both publication figures (PNG + PDF) inside out_dir/figures."""
    figs_dir = Path(out_dir) / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    plot_figure1_trajectories(df_master_traj, figs_dir)
    plot_figure2_factorial_and_cultural(df_master_traj, df_table5_domain, figs_dir)
