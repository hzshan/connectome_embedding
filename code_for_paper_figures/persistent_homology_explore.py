#%%
"""
Exploratory: shape of each cell type's embedding without specifying a target shape.

For each cell type, the learned embeddings are a point cloud in D dimensions. We
compute
  - persistent homology (Vietoris-Rips, with ripser): lifetimes of H1 (loops) and
    H2 (voids) features. A ring gives one long-lived H1 feature; a torus two H1
    and one H2; a blob or sheet none.
  - intrinsic dimension: PCA participation ratio, and the TwoNN estimator
    (Facco et al. 2017).
Each type is centered and scaled to unit root-mean-square radius first, so
lifetimes are comparable across types.

Null: Gaussian point clouds with the same number of points and the same PCA
spectrum as the type (same size and anisotropy, no loops or voids). Compared with
the null: the longest H1 lifetime (ring), and the second-longest H1 and longest H2
lifetimes (torus: two loops and a void).

Needs ripser (pip install ripser). Run from code_for_paper_figures/.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from connectome_embedding import plots, topology, data_utils

plots.set_rc_params()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
EMBEDDING_NAME = 'embeddings/embedding_Allhemibrain_D5_09262025.pkl'

MIN_NEURONS = 15     # skip types with fewer neurons (too few for H1/H2)
MAX_POINTS = 300     # subsample larger types to keep ripser fast
MAX_DIM = 2          # compute H0, H1, H2
N_NULL = 100         # Gaussian null samples per type for H1 (fast)
N_NULL_H2 = 20       # Gaussian null samples per type for H2 (slower)
SEED = 0

EXAMPLE_TYPES = ['EPG', 'PEN_b(PEN2)', 'PEG', 'Delta7', 'KCab', 'FB4Z']

# Cell-type groups for asking how well H1 picks out central complex types
# (same matching rules as types_wanted in embed_hemibrain.py: '=' exact, '*' prefix,
# otherwise substring)
CX_TYPES = ["EPG=", 'Delta7', 'PEG', 'EPGt', 'PEN', 'FC', 'PFL', 'PFN',
            'FB', 'hDelta', 'vDelta', 'FR*', 'EL', 'FS', 'PFG', 'PFR', 'IbSpsP']
# CX types that tile the PB or FB (columnar), where a ring is expected
CX_COLUMNAR_TYPES = ["EPG=", 'Delta7', 'PEG', 'EPGt', 'PEN', 'FC', 'PFL', 'PFN',
                     'hDelta', 'vDelta', 'FR*', 'FS', 'PFG', 'PFR', 'IbSpsP']
# ring neurons (EB tangential neurons that synapse onto EPG) and the TuBu neurons
# upstream of them; not columnar, reported separately
RING_AND_TUBU_TYPES = ['ER*', 'TuBu*']
N_BOOTSTRAP = 1000

#%% Load the embedding
model_and_data = pickle.load(open(EMBEDDING_NAME, 'rb'))
model = model_and_data['model']
data = model_and_data['data']
embeddings = model.embeddings.detach().numpy()
print(f'{data.N} neurons, {data.Ntype} types, D = {embeddings.shape[1]}')


# Topology and dimension of every cell type
rng = np.random.default_rng(SEED)
rows = []
all_diagrams = {}
for cell_type in data.types:
    inds = data.neuron_hash[cell_type]
    if len(inds) < MIN_NEURONS:
        continue
    x = topology.normalize(topology.subsample(embeddings[inds], MAX_POINTS, rng))

    life, diagrams = topology.lifetimes(x, maxdim=MAX_DIM)
    all_diagrams[cell_type] = diagrams

    # null: longest H1 lifetime in matched Gaussian clouds
    null_h1 = topology.null_lifetimes(x, N_NULL, rng, features=[(1, 0)], maxdim=1)[:, 0]
    # null for the torus signature: second-longest H1 and longest H2
    null_torus = topology.null_lifetimes(x, N_NULL_H2, rng, features=[(1, 1), (2, 0)], maxdim=2)

    rows.append({
        'type': cell_type,
        'n_neurons': len(inds),
        'participation_ratio': topology.participation_ratio(x),
        'two_nn_dim': topology.two_nn_dimension(x),
        'h1_1st': topology.kth_longest(life, 1, 0),
        'h1_2nd': topology.kth_longest(life, 1, 1),
        'h2_1st': topology.kth_longest(life, 2, 0),
        'h1_null_median': np.median(null_h1),
        'h1_null_95': np.percentile(null_h1, 95),
        'h1_2nd_null_95': np.percentile(null_torus[:, 0], 95),
        'h2_null_95': np.percentile(null_torus[:, 1], 95),
    })
    print(cell_type, len(inds))

results = pd.DataFrame(rows)
results['h1_above_null'] = results['h1_1st'] > results['h1_null_95']
# torus signature: a second long-lived loop and a long-lived void, both above the null
results['torus_like'] = (results['h1_2nd'] > results['h1_2nd_null_95']) & \
                        (results['h2_1st'] > results['h2_null_95'])
pd.set_option('display.max_rows', 300)
print(results.sort_values('h1_1st', ascending=False).round(3).to_string(index=False))

#%% Torus check: longest H2 lifetime per type
print(results.sort_values('h2_1st', ascending=False)
      [['type', 'n_neurons', 'h1_1st', 'h1_2nd', 'h1_2nd_null_95', 'h2_1st', 'h2_null_95', 'torus_like']]
      .round(3).to_string(index=False))
print(f"{results['torus_like'].sum()} of {len(results)} types have the torus signature")

plot_data = results.sort_values('h2_1st', ascending=False)
x = np.arange(len(plot_data))
plt.figure(figsize=(max(3, 0.06 * len(x)), 2), dpi=300)
plt.scatter(x, plot_data['h2_1st'], s=4, color='C0', label='longest H2 lifetime', zorder=3)
plt.scatter(x, plot_data['h2_null_95'], s=4, marker='_', color='k',
            label='matched Gaussian, 95th pct.', zorder=2)
plt.xticks(x, plot_data['type'], rotation=90, fontsize=3)
plt.ylim(bottom=0)
plt.ylabel('H2 lifetime\n(units of RMS radius)')
plt.legend(fontsize=5, frameon=False)
plt.tight_layout()


#%% Plot: longest H1 lifetime per type vs the matched-Gaussian null
plot_data = results.sort_values('h1_1st', ascending=False)
x = np.arange(len(plot_data))
plt.figure(figsize=(max(3, 0.06 * len(x)), 2), dpi=300)
plt.scatter(x, plot_data['h1_1st'], s=4, color='r', label='longest H1 lifetime', zorder=3)
plt.scatter(x, plot_data['h1_null_95'], s=4, marker='_', color='k',
            label='matched Gaussian, 95th pct.', zorder=2)
plt.xticks(x, plot_data['type'], rotation=90, fontsize=3)
plt.ylim(bottom=0)
plt.ylabel('H1 lifetime\n(units of RMS radius)')
plt.legend(fontsize=5, frameon=False)
plt.tight_layout()

#%% Plot: H1 lifetime vs intrinsic dimension
plt.figure(figsize=(2, 2), dpi=300)
plt.scatter(results['two_nn_dim'], results['h1_1st'],
            c=np.where(results['h1_above_null'], 'r', 'gray'), s=4)
for _, row in results[results['type'].isin(EXAMPLE_TYPES)].iterrows():
    plt.annotate(row['type'], (row['two_nn_dim'], row['h1_1st']), fontsize=4)
plt.xlabel('intrinsic dimension (TwoNN)')
plt.ylabel('longest H1 lifetime')
plt.tight_layout()


#%% Plot: persistence diagrams for a few example types
examples = [t for t in EXAMPLE_TYPES if t in all_diagrams]
fig, axes = plt.subplots(1, len(examples), figsize=(1.3 * len(examples), 1.4), dpi=300, squeeze=False)
for ax, t in zip(axes[0], examples):
    for dim, color in zip(range(MAX_DIM + 1), ['gray', 'r', 'C0']):
        d = all_diagrams[t][dim]
        d = d[np.isfinite(d[:, 1])]
        ax.scatter(d[:, 0], d[:, 1], s=2, color=color, label=f'H{dim}')
    lim = max(1e-3, max(np.nanmax(all_diagrams[t][k][np.isfinite(all_diagrams[t][k][:, 1])], initial=0)
                        for k in range(MAX_DIM + 1)))
    ax.plot([0, lim], [0, lim], color='k', lw=0.3)
    ax.set_title(t, fontsize=5)
    ax.set_xlabel('birth', fontsize=5)
axes[0][0].set_ylabel('death', fontsize=5)
axes[0][-1].legend(fontsize=4, frameon=False)
plt.tight_layout()


#%% How well does H1 pick out central complex types?
cx = data_utils.types_matching(results['type'], CX_TYPES)
cx_columnar = data_utils.types_matching(results['type'], CX_COLUMNAR_TYPES)
ring_and_tubu = data_utils.types_matching(results['type'], RING_AND_TUBU_TYPES) - cx

def group_of(t):
    if t in cx_columnar:
        return 'CX columnar'
    if t in cx:
        return 'CX other'
    if t in ring_and_tubu:
        return 'ring / TuBu'
    return 'other'

results['group'] = results['type'].map(group_of)
# H1 lifetime relative to the 95th percentile of the matched-Gaussian null (> 1: above the null)
results['h1_relative'] = results['h1_1st'] / results['h1_null_95']
print(results['group'].value_counts())

# AUC for two choices of positives, with all non-CX types as negatives
# (ring / TuBu count as negatives here; their scores are listed below)
rng = np.random.default_rng(SEED)
comparisons = {
    'all CX vs non-CX': (results['group'].isin(['CX columnar', 'CX other']),
                         ~results['group'].isin(['CX columnar', 'CX other'])),
    'CX columnar vs non-CX': (results['group'] == 'CX columnar',
                              ~results['group'].isin(['CX columnar', 'CX other'])),
}
auc_rows = []
for name, (pos_mask, neg_mask) in comparisons.items():
    for score in ['h1_1st', 'h1_relative']:
        value, lo, hi = data_utils.auc_with_ci(results.loc[pos_mask, score], results.loc[neg_mask, score],
                                    N_BOOTSTRAP, rng)
        auc_rows.append({'comparison': name, 'score': score, 'AUC': value, 'lo': lo, 'hi': hi,
                         'n_positive': pos_mask.sum(), 'n_negative': neg_mask.sum()})
auc_table = pd.DataFrame(auc_rows)
print(auc_table.round(3).to_string(index=False))

# exceptions: non-CX types above the null, and columnar CX types not above it
print('Non-CX types with H1 above the null:')
print(results[~results['group'].isin(['CX columnar', 'CX other']) & (results['h1_relative'] > 1)]
      .sort_values('h1_relative', ascending=False)
      [['type', 'group', 'n_neurons', 'h1_1st', 'h1_null_95', 'h1_relative']].round(3).to_string(index=False))
print('Columnar CX types with H1 not above the null:')
print(results[(results['group'] == 'CX columnar') & (results['h1_relative'] <= 1)]
      .sort_values('h1_relative')
      [['type', 'n_neurons', 'h1_1st', 'h1_null_95', 'h1_relative']].round(3).to_string(index=False))

#%% Plot: ROC curves (H1 relative to the null)
plt.figure(figsize=(1.8, 1.8), dpi=300)
for (name, (pos_mask, neg_mask)), color in zip(comparisons.items(), ['k', 'r']):
    fpr, tpr = data_utils.roc_curve(results.loc[pos_mask, 'h1_relative'].values,
                         results.loc[neg_mask, 'h1_relative'].values)
    value = auc_table[(auc_table['comparison'] == name) & (auc_table['score'] == 'h1_relative')]['AUC'].item()
    plt.plot(fpr, tpr, color=color, lw=1, label=f'{name} (AUC {value:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=0.5, ls='--')
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(fontsize=4, frameon=False)
plt.gca().set_aspect('equal')
plt.tight_layout()

#%% Plot: H1 relative to the null, by group
groups = ['CX columnar', 'CX other', 'ring / TuBu', 'other']
plt.figure(figsize=(2, 1.8), dpi=300)
for i, g in enumerate(groups):
    vals = results.loc[results['group'] == g, 'h1_relative'].values
    plt.scatter(i + rng.uniform(-0.2, 0.2, len(vals)), vals, s=3, color='k', lw=0, alpha=0.6)
    if len(vals):
        plt.plot([i - 0.3, i + 0.3], [np.median(vals)] * 2, color='r', lw=1)
plt.axhline(1, color='gray', lw=0.5, ls='--')
plt.xticks(range(len(groups)), [f'{g}\n(n={np.sum(results["group"] == g)})' for g in groups],
           rotation=45, ha='right', fontsize=5)
plt.ylim(bottom=0)
plt.ylabel('longest H1 lifetime /\nnull 95th percentile')
plt.tight_layout()

