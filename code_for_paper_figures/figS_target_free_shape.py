#%%
"""
Shape of each cell type's embedding without specifying a target shape, in the
whole-hemibrain embedding, compared with block-shuffled embeddings.

Two target-free measures per cell type (types with at least 15 neurons), both
computed by score_type_geometry.py:
  - intrinsic dimension, Gride estimator (Denti et al. 2022) at neighbour scale
    n1 = 4 (ratio of the distances to the 8th and 4th nearest neighbours), on all
    neurons of the type (dimension.csv);
  - persistent homology: lifetime of the longest-lived H1 feature (loop), after
    scaling the type to unit RMS radius (homology.csv). A ring gives one
    long-lived loop.

Null: the same measures on embeddings fitted to block-shuffled connectomes (weights
shuffled within each pair of cell types, then re-embedded; embed_hemibrain.py,
CONFIG = 'whole_hemibrain_null', one embedding per seed). The shuffle keeps the
mean connectivity between types and removes the structure within types.

How well H1 separates central complex types from the others (ROC, AUC) is in
ring_detection_auc.py.

Needs the homology.csv and dimension.csv files made by score_type_geometry.py.
Run from code_for_paper_figures/.
"""
import glob
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from connectome_embedding import plots

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
EMBEDDING_DIR = 'embeddings/'
REAL = 'whole_hemibrain_embedding'
NULLS = 'whole_hemibrain_embedding_block_shuffled_seed*'
RESULTS_DIR = 'results/type_geometry/'
SCORE_FILES = ['homology', 'dimension']   # CSVs per embedding (score_type_geometry.py)

GROUP_COLORS = {'CX columnar': 'goldenrod', 'CX other': 'tab:orange',
                'ring / TuBu': 'tab:blue', 'other': 'gray'}


def type_table(name):
    """Longest H1 lifetime and intrinsic dimension per type for one embedding."""
    tables = [pd.read_csv(os.path.join(RESULTS_DIR, name, f'{score}.csv')) for score in SCORE_FILES]
    return tables[0][['type', 'n_neurons', 'group', 'h1_1st']].merge(
        tables[1][['type', 'gride_dim']], on='type')


#%% Real and block-shuffled embeddings
plots.set_rc_params()
D = pickle.load(open(os.path.join(EMBEDDING_DIR, REAL + '.pkl'), 'rb'))['hyperparams']['embedding_dim']

real = type_table(REAL)
# block-shuffled embeddings with every score computed by score_type_geometry.py
null_names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(RESULTS_DIR, NULLS))
                    if all(os.path.isfile(os.path.join(p, f'{s}.csv')) for s in SCORE_FILES))
null = pd.concat([type_table(n).assign(embedding=n) for n in null_names], ignore_index=True)
print(f'{len(real)} cell types; {len(null_names)} block-shuffled embeddings')

#%% Summary
real['h1_above_null'] = [h > null.loc[null['type'] == t, 'h1_1st'].max()
                         for t, h in zip(real['type'], real['h1_1st'])]

print(f"Gride estimate > D = {D} for {100 * (real['gride_dim'] > D).mean():.0f}% of types "
      f"(block-shuffled: {100 * (null['gride_dim'] > D).mean():.0f}%)")
summary = real.groupby('group').agg(
    n_types=('type', 'size'),
    gride_median=('gride_dim', 'median'),
    h1_median=('h1_1st', 'median'),
    frac_h1_above_all_nulls=('h1_above_null', 'mean'))
summary['null_gride_median'] = null.groupby('group')['gride_dim'].median()
summary['null_h1_median'] = null.groupby('group')['h1_1st'].median()
print(summary.round(2).to_string())

#%% Plot: each measure per cell type, real vs block-shuffled; each panel sorted by its own measure
panels = [('h1_1st', 'longest H1 lifetime\n(units of RMS radius)', False),
          ('gride_dim', 'intrinsic dimension\n(Gride, n1 = 4)', True)]
fig, axes = plt.subplots(len(panels), 1, figsize=(7, 1.6 * len(panels)), dpi=300)
for ax, (column, ylabel, mark_D) in zip(axes, panels):
    plot_data = real.sort_values(column, ascending=False).set_index('type')
    order = plot_data.index.tolist()
    x = np.arange(len(order))
    for name in null_names:
        values = null[null['embedding'] == name].set_index('type').reindex(order)[column]
        ax.scatter(x, values, s=1, color='k', marker='_', lw=0.4, zorder=2)
    ax.scatter(x, plot_data[column], s=3, color=plot_data['group'].map(GROUP_COLORS), zorder=3)
    if mark_D:
        ax.axhline(D, color='k', lw=0.5, ls='--')
    ax.set_ylim(bottom=0)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=90, fontsize=3)
    ax.set_xlim(-1, len(order))
    ax.set_ylabel(ylabel, fontsize=5)
axes[0].set_title(f'{len(real)} cell types with >= 15 neurons, each panel sorted by its own measure; '
                  f'dots: embedding; black ticks: {len(null_names)} block-shuffled embeddings; '
                  f'dashed: D = {D}', fontsize=5)
axes[0].legend(handles=[plt.Line2D([], [], color=c, marker='o', ls='', ms=2, label=g)
                        for g, c in GROUP_COLORS.items()], fontsize=4, frameon=False)
plt.tight_layout()

#%% Plot: intrinsic dimension vs H1 lifetime
fig, axes = plt.subplots(1, 2, figsize=(4, 2), dpi=300, sharex=True, sharey=True)
for ax, table, title in [(axes[0], real, 'embedding'), (axes[1], null, 'block-shuffled')]:
    ax.scatter(table['gride_dim'], table['h1_1st'], c=table['group'].map(GROUP_COLORS),
               s=3, alpha=1 if table is real else 0.5)
    ax.axvline(D, color='k', lw=0.5, ls='--')
    ax.set_xlabel('intrinsic dimension (Gride, n1 = 4)')
    ax.set_title(title, fontsize=6)
axes[0].set_ylabel('longest H1 lifetime\n(units of RMS radius)')
plt.tight_layout()
