"""Plot observed training trajectories from the archived layer-shift table."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = pd.read_csv(ROOT / 'results/trajectory_summary.csv')
RUNS = [('infonce_01', 'InfoNCE, λ = 0.1', '#0072B2', 'o'),
        ('infonce_1', 'InfoNCE, λ = 1.0', '#009E73', 's'),
        ('sigreg_01', 'SIGReg, λ = 0.1', '#D55E00', '^')]


def draw(columns, labels, name):
    fig, axes = plt.subplots(2, len(columns), figsize=(3.35 * len(columns), 5.3), sharex=True, squeeze=False)
    for row, pool in enumerate(('cls', 'mean')):
        for col, (metric, ylabel) in enumerate(zip(columns, labels)):
            ax = axes[row, col]
            for run, label, color, marker in RUNS:
                d = DATA[(DATA.run == run) & (DATA.pooling == pool)].sort_values('epoch')
                # Reindexing leaves gaps where a checkpoint is not available.
                d = d.set_index('epoch').reindex(range(1, int(d.epoch.max()) + 1))
                ax.plot(d.index, d[metric], color=color, marker=marker, markersize=4, lw=1.5, label=label)
            if metric.endswith('_shift'):
                ax.axhline(0, color='0.45', lw=.8, ls=':')
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_title(f'{"CLS" if pool == "cls" else "Mean pooling"}', fontsize=10, loc='left')
            ax.set_xticks(range(1, 11))
            ax.grid(axis='y', alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
            ax.tick_params(labelsize=8)
            if row == 1:
                ax.set_xlabel('Recorded epoch', fontsize=9)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .92))
    for ext in ('png', 'svg'):
        fig.savefig(ROOT / f'figures/{name}.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    draw(['knn_pct_shift', 'rank_src_shift', 'rank_en_shift'],
         ['10-NN overlap change (pp)', 'Source effective rank change', 'English effective rank change'], 'layer_shifts')
    draw(['knn_pct_l9', 'knn_pct_l12', 'knn_pct_shift'],
         ['L9 overlap (%)', 'L12 overlap (%)', 'L12 minus L9 (pp)'], 'absolute_alignment')
