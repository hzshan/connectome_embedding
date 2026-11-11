#%%
"""
Which score best tells central complex (CX) cell types apart from the others:
H1 from persistent homology, the intrinsic dimension (Gride), or the circularity score?

Reads the per-type tables made by score_type_geometry.py and
  1. computes the AUC of each score for CX vs non-CX types and for columnar CX vs
     non-CX types (ring / TuBu types count as non-CX), with 95% bootstrap intervals,
     and the paired difference between the AUCs of each pair of scores;
  2. plots ROC curves, H1 against circularity, and each score by group;
  3. lists the exceptions: non-CX types above the null, columnar CX types below it;
  4. compares each score on the real whole-hemibrain embedding with the same score on
     embeddings fitted to block-shuffled weights (embed_hemibrain.py null config
     whole_hemibrain_null): median per group, AUC on the shuffled embeddings, ROC
     curves of the real and shuffled embeddings together, and each type's score.

Scores compared:
  h1_1st          longest H1 lifetime (units of the type's RMS radius)
  gride_dim       intrinsic dimension, Gride estimator (score_type_geometry.py, GRIDE_N1);
                  a ring is low-dimensional, so a lower value counts as more CX-like
                  (SCORE_SIGNS): the AUC and ROC use -gride_dim
  circularity     circularity score of the best 2D projection
Also reported: h1_relative, the H1 lifetime relative to matched Gaussian clouds (if
score_type_geometry.py computed it).
Run from code_for_paper_figures/.
"""
import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from connectome_embedding import plots, data_utils, anatomy

plots.set_rc_params()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
RESULTS_DIR = 'results/type_geometry/'
REAL = 'whole_hemibrain_embedding'
NULLS = 'whole_hemibrain_embedding_block_shuffled_seed*'

SCORES = {'h1_1st': 'longest H1 lifetime', 'gride_dim': 'intrinsic dimension (Gride)',
          'circularity': 'circularity score'}
SCORE_SIGNS = {'gride_dim': -1}   # scores where a lower value is more CX-like
SCORE_COLORS = {'h1_1st': 'k', 'gride_dim': 'tab:blue', 'circularity': 'r'}
EXTRA_SCORES = {'h1_relative': 'H1 lifetime / Gaussian null 95th pct.'}
SCORE_FILES = ['circularity', 'homology', 'dimension']   # CSVs per embedding (score_type_geometry.py)
N_BOOTSTRAP = 1000
SEED = 0
GROUP_COLORS = {'CX columnar': 'r', 'CX other': 'goldenrod', 'ring / TuBu': 'C0', 'other': 'gray'}


def load(embedding):
    """All scores of one embedding, one row per type."""
    table = None
    for score_file in SCORE_FILES:
        path = os.path.join(RESULTS_DIR, embedding, score_file + '.csv')
        if not os.path.isfile(path):
            print(f'missing: {path}')
            continue
        t = pd.read_csv(path)
        table = t if table is None else table.merge(t, on=['embedding', 'type', 'n_neurons', 'group'])
    return table


def load_nulls(pattern):
    """Scores of all null embeddings matching the pattern, stacked, with a 'null_seed' column."""
    folders = sorted(glob.glob(os.path.join(RESULTS_DIR, pattern)))
    if not folders:
        print(f'no null embeddings scored for {pattern}')
        return None
    tables = [load(os.path.basename(f)).assign(null_seed=int(f.split('seed')[-1])) for f in folders]
    print(f'{len(folders)} null embeddings for {pattern}')
    return pd.concat(tables, ignore_index=True)


def oriented(table, score):
    """Score values with the sign that makes higher = more CX-like (SCORE_SIGNS)."""
    return SCORE_SIGNS.get(score, 1) * table[score]


def comparisons(table):
    """Positives and negatives (boolean masks) of each comparison."""
    is_cx = table['group'].isin(['CX columnar', 'CX other'])
    return {'all CX vs non-CX': (is_cx, ~is_cx),
            'CX columnar vs non-CX': (table['group'] == 'CX columnar', ~is_cx)}


def auc_table(table, scores, rng):
    rows = []
    for name, (pos, neg) in comparisons(table).items():
        for score in scores:
            values = oriented(table, score)
            value, lo, hi = data_utils.auc_with_ci(values[pos], values[neg], N_BOOTSTRAP, rng)
            rows.append({'comparison': name, 'score': score, 'AUC': value, 'lo': lo, 'hi': hi,
                         'n_positive': pos.sum(), 'n_negative': neg.sum()})
    return pd.DataFrame(rows)


#%% Load the real whole-hemibrain table
rng = np.random.default_rng(SEED)
real = load(REAL)
print(f'{REAL}: {len(real)} types')
print(real['group'].value_counts().to_string())

#%% AUC of each score, and the paired difference between each pair of scores
extra = [s for s in EXTRA_SCORES if s in real.columns]
aucs = auc_table(real, list(SCORES) + extra, rng)
print(aucs.round(3).to_string(index=False))

score_list = list(SCORES)
for name, (pos, neg) in comparisons(real).items():
    for i, score_a in enumerate(score_list):
        for score_b in score_list[i + 1:]:
            diff, lo, hi = data_utils.auc_difference_with_ci(
                oriented(real, score_a), oriented(real, score_b), pos, neg, N_BOOTSTRAP, rng)
            print(f'{name}: AUC({score_a}) - AUC({score_b}) = {diff:.3f} [{lo:.3f}, {hi:.3f}]')

#%% Plot: ROC curves of all scores
fig, axes = plt.subplots(1, 2, figsize=(3.6, 2), dpi=300)
for ax, (name, (pos, neg)) in zip(axes, comparisons(real).items()):
    for score, score_title in SCORES.items():
        color = SCORE_COLORS[score]
        values = oriented(real, score)
        fpr, tpr = data_utils.roc_curve(values[pos].values, values[neg].values)
        row = aucs[(aucs['comparison'] == name) & (aucs['score'] == score)].iloc[0]
        ax.plot(fpr, tpr, color=color, lw=1,
                label=f'{score_title}\nAUC {row["AUC"]:.2f} [{row["lo"]:.2f}, {row["hi"]:.2f}]')
    ax.plot([0, 1], [0, 1], color='gray', lw=0.5, ls='--')
    ax.set_title(f'{name} (n = {pos.sum()} / {neg.sum()})', fontsize=6)
    ax.set_xlabel('false positive rate', fontsize=6)
    ax.legend(fontsize=4, frameon=False, loc='lower right')
    ax.set_aspect('equal')
axes[0].set_ylabel('true positive rate', fontsize=6)
fig.suptitle(f'{REAL}; brackets: 95% bootstrap CI', fontsize=5)
plt.tight_layout()

#%% Plot: H1 against circularity, colored by group
score_a, score_b = 'h1_1st', 'circularity'
plt.figure(figsize=(2.2, 2), dpi=300)
for g in anatomy.GROUPS:
    m = real['group'] == g
    plt.scatter(real.loc[m, score_b], real.loc[m, score_a], s=3, lw=0,
                color=GROUP_COLORS[g], label=f'{g} (n={m.sum()})')
plt.xlabel(SCORES[score_b])
plt.ylabel(SCORES[score_a])
plt.legend(fontsize=4, frameon=False)
plt.tight_layout()

#%% Plot: each score by group
fig, axes = plt.subplots(1, len(SCORES), figsize=(1.8 * len(SCORES), 1.8), dpi=300)
for ax, (score, score_title) in zip(axes, SCORES.items()):
    for i, g in enumerate(anatomy.GROUPS):
        vals = real.loc[real['group'] == g, score].values
        ax.scatter(i + rng.uniform(-0.2, 0.2, len(vals)), vals, s=3, color='k', lw=0, alpha=0.6)
        if len(vals):
            ax.plot([i - 0.3, i + 0.3], [np.median(vals)] * 2, color='r', lw=1)
    ax.set_xticks(range(len(anatomy.GROUPS)))
    ax.set_xticklabels(anatomy.GROUPS, rotation=45, ha='right', fontsize=5)
    ax.set_ylabel(score_title, fontsize=6)
plt.tight_layout()

#%% Exceptions
pd.set_option('display.max_rows', 300)
# thresholds: the median score of the columnar CX types (oriented: higher = more CX-like)
columns = ['type', 'group', 'n_neurons'] + list(SCORES) + extra
is_cx = real['group'].isin(['CX columnar', 'CX other'])
columnar = real['group'] == 'CX columnar'
above = np.column_stack([oriented(real, s) >= oriented(real, s)[columnar].median() for s in SCORES]).any(1)
below = np.column_stack([oriented(real, s) < oriented(real, s)[columnar].median() for s in SCORES]).all(1)
print('Non-CX types at or above the columnar-CX median on any score:')
print(real[~is_cx & above].sort_values(score_a, ascending=False)[columns].round(3).to_string(index=False))
print('Columnar CX types below the columnar-CX median on every score:')
print(real[columnar & below].sort_values(score_a)[columns].round(3).to_string(index=False))


#%% Real vs block-shuffled embeddings: median per group and AUC on the shuffled embeddings
nulls = load_nulls(NULLS)
null_scores = list(SCORES) + [s for s in EXTRA_SCORES if s in real.columns]

# median of each score per group, real vs shuffled
summary = pd.concat({'real': real.groupby('group')[null_scores].median(),
                     'shuffled': nulls.groupby('group')[null_scores].median()}, axis=1)
print(f'{REAL}: median per group')
print(summary.round(3).to_string())

# AUC on the shuffled embeddings
null_aucs = pd.concat([auc_table(t, list(SCORES), rng).assign(null_seed=s)
                       for s, t in nulls.groupby('null_seed')])
print(f'{NULLS}: AUC on shuffled embeddings (mean and range over seeds)')
print(null_aucs.groupby(['comparison', 'score'])['AUC']
      .agg(['mean', 'min', 'max']).round(3).to_string())

#%% Plot: ROC curves on the real and the block-shuffled embeddings
def plot_roc_real_vs_null(real_table, null_table, title):
    """
    ROC curves of each score (columns) for each comparison (rows): the real embedding
    (colored) and each block-shuffled embedding (gray, one curve per null seed).
    """
    comps = list(comparisons(real_table))
    fig, axes = plt.subplots(len(comps), len(SCORES), figsize=(1.9 * len(SCORES), 3.6), dpi=300,
                             squeeze=False)
    for i, (score, score_title) in enumerate(SCORES.items()):
        color = SCORE_COLORS[score]
        for j, name in enumerate(comps):
            ax = axes[j, i]
            null_aucs = []
            for _, t in null_table.groupby('null_seed'):
                pos, neg = comparisons(t)[name]
                values = oriented(t, score)
                p, n = values[pos].values, values[neg].values
                ax.plot(*data_utils.roc_curve(p, n), color='gray', lw=0.4, alpha=0.8)
                null_aucs.append(data_utils.auc(p, n))
            pos, neg = comparisons(real_table)[name]
            values = oriented(real_table, score)
            p, n = values[pos].values, values[neg].values
            ax.plot(*data_utils.roc_curve(p, n), color=color, lw=1,
                    label=f'real, AUC {data_utils.auc(p, n):.2f}')
            ax.plot([], [], color='gray', lw=0.4,
                    label=f'shuffled ({len(null_aucs)} seeds), AUC {np.mean(null_aucs):.2f} '
                          f'[{np.min(null_aucs):.2f}, {np.max(null_aucs):.2f}]')
            ax.plot([0, 1], [0, 1], color='gray', lw=0.5, ls=':')
            ax.set_aspect('equal')
            ax.legend(fontsize=3.5, frameon=False, loc='lower right')
            if j == 0:
                sign = '\n(lower = CX)' if SCORE_SIGNS.get(score, 1) < 0 else ''
                ax.set_title(f'{score_title}{sign}', fontsize=5)
            if j == len(comps) - 1:
                ax.set_xlabel('false positive rate', fontsize=6)
            if i == 0:
                ax.set_ylabel(f'{name}\n(n = {pos.sum()} / {neg.sum()})\ntrue positive rate', fontsize=6)
    fig.suptitle(f'{title}; shuffled: range over seeds', fontsize=5)
    plt.tight_layout()


plot_roc_real_vs_null(real, nulls, f'{REAL} vs block-shuffled')

#%% Plot: each score per type, real vs block-shuffled: raw values, then z-scored against
# the type's block-shuffled values
def plot_real_vs_null(real_table, null_table, score, title, ylabel=None):
    """
    One column per type, sorted by the real score: real (colored by group, one legend
    entry per group) and each null seed (black).
    """
    order = real_table.sort_values(score, ascending=False)['type'].tolist()
    x = {t: i for i, t in enumerate(order)}
    plt.figure(figsize=(max(3, 0.05 * len(order)), 1.8), dpi=300)
    null_rows = null_table[null_table['type'].isin(x)]
    plt.scatter(null_rows['type'].map(x), null_rows[score], s=2, color='k', lw=0,
                label='block-shuffled', zorder=2)
    for g in anatomy.GROUPS:
        rows = real_table[real_table['group'] == g]
        plt.scatter(rows['type'].map(x), rows[score], s=4, lw=0, zorder=3,
                    color=GROUP_COLORS[g], label=f'real, {g}')
    plt.xticks(range(len(order)), order, rotation=90, fontsize=3)
    plt.ylabel({**SCORES, **EXTRA_SCORES}.get(score, score) if ylabel is None else ylabel, fontsize=6)
    plt.title(title, fontsize=6)
    plt.legend(fontsize=4, frameon=False)
    plt.tight_layout()


def zscore_by_null(real_table, null_table, scores):
    """
    Real and null tables with each score z-scored per type against the null seeds:
    (value - mean over null seeds of the type) / (std over null seeds of the type).
    Types that are not in the null table are dropped.
    """
    stats = null_table.groupby('type')[scores].agg(['mean', 'std'])
    real_z = real_table[real_table['type'].isin(stats.index)].copy()
    null_z = null_table.copy()
    for table in (real_z, null_z):
        for score in scores:
            mean = table['type'].map(stats[(score, 'mean')])
            std = table['type'].map(stats[(score, 'std')])
            table[score] = (table[score] - mean) / std
    return real_z, null_z


for score in SCORES:
    plot_real_vs_null(real, nulls, score, f'{REAL} vs block-shuffled')

real_z, nulls_z = zscore_by_null(real, nulls, list(SCORES))
for score in SCORES:
    plot_real_vs_null(real_z, nulls_z, score, f'{REAL} vs block-shuffled',
                      ylabel=f'{SCORES[score]}\n(z-score vs block-shuffled)')
    plt.axhline(0, color='gray', lw=0.5, ls='--', zorder=1)
