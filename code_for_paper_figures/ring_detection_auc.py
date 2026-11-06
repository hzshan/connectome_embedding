#%%
"""
Which score best tells central complex (CX) cell types apart from the others:
H1 from persistent homology, or the circularity score?

Reads the per-type tables made by score_type_geometry.py and
  1. computes the AUC of each score for CX vs non-CX types and for columnar CX vs
     non-CX types (ring / TuBu types count as non-CX), with 95% bootstrap intervals,
     and the paired difference between the two scores' AUCs;
  2. plots ROC curves, the two scores against each other, and each score by group;
  3. lists the exceptions: non-CX types above the null, columnar CX types below it;
  4. compares each score on the real embedding with the same score on embeddings
     fitted to block-shuffled weights (embed_hemibrain.py null configs), for the
     whole-hemibrain and the CX embeddings, including the AUC on the shuffled ones.

Scores compared:
  h1_1st          longest H1 lifetime (units of the type's RMS radius)
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
REAL = 'whole_hemibrain_embedding'                       # CX vs non-CX comparison
NULLS = 'whole_hemibrain_embedding_block_shuffled_seed*'
CX_REAL = 'CX_embedding'
CX_NULLS = 'CX_embedding_block_shuffled_seed*'

SCORES = {'h1_1st': 'longest H1 lifetime', 'circularity': 'circularity score'}
EXTRA_SCORES = {'h1_relative': 'H1 lifetime / Gaussian null 95th pct.'}
SCORE_FILES = ['circularity', 'homology', 'anatomy']   # CSVs per embedding (score_type_geometry.py)
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


def comparisons(table):
    """Positives and negatives (boolean masks) of each comparison."""
    is_cx = table['group'].isin(['CX columnar', 'CX other'])
    return {'all CX vs non-CX': (is_cx, ~is_cx),
            'CX columnar vs non-CX': (table['group'] == 'CX columnar', ~is_cx)}


def auc_table(table, scores, rng):
    rows = []
    for name, (pos, neg) in comparisons(table).items():
        for score in scores:
            value, lo, hi = data_utils.auc_with_ci(table.loc[pos, score], table.loc[neg, score],
                                                   N_BOOTSTRAP, rng)
            rows.append({'comparison': name, 'score': score, 'AUC': value, 'lo': lo, 'hi': hi,
                         'n_positive': pos.sum(), 'n_negative': neg.sum()})
    return pd.DataFrame(rows)


#%% Load the real whole-hemibrain table
rng = np.random.default_rng(SEED)
real = load(REAL)
print(f'{REAL}: {len(real)} types')
print(real['group'].value_counts().to_string())

#%% AUC of each score, and the paired difference between the two main scores
extra = [s for s in EXTRA_SCORES if s in real.columns]
aucs = auc_table(real, list(SCORES) + extra, rng)
print(aucs.round(3).to_string(index=False))

score_a, score_b = list(SCORES)
for name, (pos, neg) in comparisons(real).items():
    diff, lo, hi = data_utils.auc_difference_with_ci(real[score_a], real[score_b], pos, neg,
                                                     N_BOOTSTRAP, rng)
    print(f'{name}: AUC({score_a}) - AUC({score_b}) = {diff:.3f} [{lo:.3f}, {hi:.3f}]')

#%% Plot: ROC curves of both scores
fig, axes = plt.subplots(1, 2, figsize=(3.6, 2), dpi=300)
for ax, (name, (pos, neg)) in zip(axes, comparisons(real).items()):
    for (score, score_title), color in zip(SCORES.items(), ['k', 'r']):
        fpr, tpr = data_utils.roc_curve(real.loc[pos, score].values, real.loc[neg, score].values)
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

#%% Plot: the two scores against each other, colored by group
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
fig, axes = plt.subplots(1, 2, figsize=(3.6, 1.8), dpi=300)
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
# thresholds: the median score of the columnar CX types
columns = ['type', 'group', 'n_neurons'] + list(SCORES) + extra
is_cx = real['group'].isin(['CX columnar', 'CX other'])
columnar = real['group'] == 'CX columnar'
above = np.column_stack([real[s] >= real.loc[columnar, s].median() for s in SCORES]).any(1)
below = np.column_stack([real[s] < real.loc[columnar, s].median() for s in SCORES]).all(1)
print('Non-CX types at or above the columnar-CX median on either score:')
print(real[~is_cx & above].sort_values(score_a, ascending=False)[columns].round(3).to_string(index=False))
print('Columnar CX types below the columnar-CX median on both scores:')
print(real[columnar & below].sort_values(score_a)[columns].round(3).to_string(index=False))


#%% Real vs block-shuffled embeddings
def plot_real_vs_null(real_table, null_table, score, title):
    """One column per type, sorted by the real score: real (colored by group) and each null seed (black)."""
    order = real_table.sort_values(score, ascending=False)['type'].tolist()
    x = {t: i for i, t in enumerate(order)}
    plt.figure(figsize=(max(3, 0.05 * len(order)), 1.8), dpi=300)
    null_rows = null_table[null_table['type'].isin(x)]
    plt.scatter(null_rows['type'].map(x), null_rows[score], s=2, color='k', lw=0,
                label='block-shuffled', zorder=2)
    plt.scatter(real_table['type'].map(x), real_table[score], s=4, lw=0, zorder=3,
                color=real_table['group'].map(GROUP_COLORS), label='real (color: group)')
    plt.xticks(range(len(order)), order, rotation=90, fontsize=3)
    plt.ylabel({**SCORES, **EXTRA_SCORES}.get(score, score), fontsize=6)
    plt.title(title, fontsize=6)
    plt.legend(fontsize=4, frameon=False)
    plt.tight_layout()


for real_name, null_pattern in [(REAL, NULLS), (CX_REAL, CX_NULLS)]:
    nulls = load_nulls(null_pattern)
    if nulls is None:
        continue
    real_table = load(real_name)
    scores = list(SCORES) + [s for s in EXTRA_SCORES if s in real_table.columns] + ['anatomy_tau']

    # median of each score per group, real vs shuffled
    summary = pd.concat({'real': real_table.groupby('group')[scores].median(),
                         'shuffled': nulls.groupby('group')[scores].median()}, axis=1)
    print(f'{real_name}: median per group')
    print(summary.round(3).to_string())

    # AUC on the shuffled embeddings (only meaningful with non-CX types)
    if (real_table['group'] == 'other').any():
        null_aucs = pd.concat([auc_table(t, list(SCORES), rng).assign(null_seed=s)
                               for s, t in nulls.groupby('null_seed')])
        print(f'{null_pattern}: AUC on shuffled embeddings (mean and range over seeds)')
        print(null_aucs.groupby(['comparison', 'score'])['AUC']
              .agg(['mean', 'min', 'max']).round(3).to_string())

    for score in scores:
        plot_real_vs_null(real_table, nulls, score, f'{real_name} vs block-shuffled')
