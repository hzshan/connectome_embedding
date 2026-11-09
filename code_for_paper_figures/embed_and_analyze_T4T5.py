#%%
"""
Exploratory: does the embedding find the direction-selective subtypes of T4 and T5
when it is not told about them?

T4 and T5 each come in four subtypes (a-d), one per lobula plate layer and preferred
direction of motion, and each subtype tiles the visual field. So T4 neurons carry
two variables: a discrete direction (4 values) and a 2D retinotopic position. Here
the subtypes are merged into one type, T4 (and T5), before fitting, and we ask
whether the embeddings of T4 and T5 neurons
  - encode retinotopic position (cross-validated prediction of the column
    coordinates from the embedding), and
  - separate the four subtypes once the retinotopy is removed: the embeddings are
    regressed on column position, and the top principal components of the residuals
    are clustered with k-means, with the number of clusters chosen by silhouette
    (label-free); agreement with the subtypes is measured by the adjusted Rand index. k-nearest-neighbour
    classification of the held-out subtype is reported as well.
The subtype labels are used only for scoring.

Data: the maleCNS v0.9 release (body annotations and connection weights, feather
files). One optic lobe, restricted to a patch of columns to keep the connectivity
matrix small. Neurons are chosen by connectivity alone:
  - T4 and T5 neurons in the patch;
  - partner types: cell types that provide at least PARTNER_MIN_FRACTION of the
    input synapses of T4 or of T5, or receive at least that fraction of their output
    synapses (over the whole optic lobe, subtypes pooled, so the choice does not
    depend on the subtype labels);
  - partner neurons: neurons of the partner types with at least PARTNER_MIN_SYN
    synapses (both directions) with the T4 and T5 neurons in the patch;
  - types with fewer than MIN_NUM_PER_TYPE neurons in the selection are dropped.
Columns are given by the annotated hex coordinates (assignedOlHex1/2) of the
columnar medulla neurons. T4 and T5 have none; their column is estimated as the
synapse-weighted mean column of their presynaptic partners that have one.

Run from code_for_paper_figures/. The fitted embedding is saved in the format of
embed_hemibrain.py; if the file exists, it is loaded instead of refitted.
"""
import os
import pickle
import datetime
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from connectome_embedding import models, data_utils, plots

plots.set_rc_params()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DATAPATH = os.path.expanduser('~/Dropbox/Data/data_maleCNS/')
ANNOTATION_FILE = 'body-annotations-male-cns-v0.9-minconf-0.5.feather'
WEIGHTS_FILE = 'connectome-weights-male-cns-v0.9-minconf-0.5.feather'
EMBEDDING_DIR = 'embeddings/'   # the fitted embedding is saved here

# Use a patch of columns to keep the connectivity matrix small; the patch is centered
SIDE = 'R'                 # optic lobe (somaSide of the T4 / T5 neurons)
PATCH_RADIUS = 6           # keep T4 / T5 with |hex1 - c1| <= R and |hex2 - c2| <= R,
                           # around the median column (c1, c2) of Mi1

SUBTYPES = {'T4': ['T4a', 'T4b', 'T4c', 'T4d'],
            'T5': ['T5a', 'T5b', 'T5c', 'T5d']}

# select input/output types to include
PARTNER_MIN_FRACTION = 0.01  # partner type: >= this fraction of T4 or T5 input or output synapses
PARTNER_MIN_SYN = 10         # partner neuron: >= this many synapses with the patch's T4 / T5
MIN_NUM_PER_TYPE = 3         # drop types with fewer neurons in the selection

hp = {
    'embedding_dim': 7,
    'wrapper_fn': 'softplus',
    'embedding_regularizer': 0.001,
    'lr': 0.05,
    'steps': 10000,
    'loss_type': 'poisson',
    'seed': 0,
}
hp['save_filename'] = (f"optic_lobe_T4T5_{SIDE}_patch{PATCH_RADIUS}_partners{PARTNER_MIN_FRACTION}"
                       f"_D{hp['embedding_dim']}_seed{hp['seed']}.pkl")

# Scoring
N_FOLDS = 5
N_NEIGHBORS = 10
K_RANGE = range(2, 9)      # numbers of clusters tried for the silhouette score
RETINOTOPY_DEGREE = 2      # polynomial in column position regressed out before clustering
CLUSTER_PCS = 2            # cluster the top CLUSTER_PCS principal components of the
                           # residuals; None clusters all embedding dimensions
SCORE_SEED = 0


#%% Annotations and the patch of columns
ann = pd.read_feather(DATAPATH + ANNOTATION_FILE)
ann = ann[(ann['status'] == 'Traced') & ann['type'].notna()]
ann = ann[~ann['type'].str.contains('_unclear')]

has_hex = ann[(ann['somaSide'] == SIDE) & ann['assignedOlHex1'].notna()]

# Mi1 neurons are used to define the hexagonal grid
center = has_hex[has_hex['type'] == 'Mi1'][['assignedOlHex1', 'assignedOlHex2']].median().round()
print(f'patch center: hex1 = {center.iloc[0]:.0f}, hex2 = {center.iloc[1]:.0f}')

# since the grid is hexagonal, the patch is a rectangle in coordinate space but
#  a parallelogram in physical space
def in_patch(hex1, hex2):
    return ((np.abs(hex1 - center.iloc[0]) <= PATCH_RADIUS)
            & (np.abs(hex2 - center.iloc[1]) <= PATCH_RADIUS))


#%% Partner types: share of T4 and T5 input and output synapses (subtypes pooled)
conns = pd.read_feather(DATAPATH + WEIGHTS_FILE)

subtype_list = SUBTYPES['T4'] + SUBTYPES['T5']
side_t4t5 = ann[(ann['somaSide'] == SIDE) & ann['type'].isin(subtype_list)]
type_of = ann.set_index('bodyId')['type']

shares = []
for family, subs in SUBTYPES.items():
    ids = set(side_t4t5.loc[side_t4t5['type'].isin(subs), 'bodyId'])
    for direction, own, partner in [('input', 'body_post', 'body_pre'),
                                    ('output', 'body_pre', 'body_post')]:
        c = conns[conns[own].isin(ids)]
        # denominator: all synapses, including those with untyped or untraced partners
        per_type = c.groupby(type_of.reindex(c[partner]).values)['weight'].sum()
        shares.append((per_type / c['weight'].sum()).rename(f'{family} {direction}'))
shares = pd.concat(shares, axis=1).fillna(0).drop(index=subtype_list, errors='ignore')
partner_types = shares.index[(shares >= PARTNER_MIN_FRACTION).any(axis=1)].tolist()
print(f'{len(partner_types)} partner types (>= {PARTNER_MIN_FRACTION:.0%} of T4 or T5 '
      f'input or output synapses):')
#%% Estimated Retinotopic Positions of all T4 and T5: weighted mean column of their
# presynaptic partners
inputs_with_hex = conns[conns['body_post'].isin(set(side_t4t5['bodyId']))
                        & conns['body_pre'].isin(set(has_hex['bodyId']))]
inputs_with_hex = inputs_with_hex.merge(
    has_hex[['bodyId', 'assignedOlHex1', 'assignedOlHex2']],
    left_on='body_pre', right_on='bodyId')
estimated = inputs_with_hex.groupby('body_post').apply(
    lambda g: pd.Series({'hex1': np.average(g['assignedOlHex1'], weights=g['weight']),
                         'hex2': np.average(g['assignedOlHex2'], weights=g['weight']),
                         'syn_with_hex': g['weight'].sum()}),
    include_groups=False)
print(f'estimated the column of {len(estimated)} of {len(side_t4t5)} T4 / T5 neurons; median '
      f'{estimated["syn_with_hex"].median():.0f} synapses from inputs with an annotated column')

positions = pd.concat([
    has_hex.set_index('bodyId')[['assignedOlHex1', 'assignedOlHex2']]
           .set_axis(['hex1', 'hex2'], axis=1).assign(position_source='annotated'),
    estimated[['hex1', 'hex2']].assign(position_source='inputs'),
])

#%% Select neurons: T4 / T5 in the patch, plus partner neurons connected to them
t4t5 = side_t4t5.join(positions, on='bodyId', how='inner')
t4t5 = t4t5[in_patch(t4t5['hex1'], t4t5['hex2'])]
t4t5_ids = set(t4t5['bodyId'])

candidates = ann[ann['type'].isin(partner_types)]
candidate_ids = set(candidates['bodyId'])
to_patch = conns[conns['body_pre'].isin(candidate_ids) & conns['body_post'].isin(t4t5_ids)]
from_patch = conns[conns['body_pre'].isin(t4t5_ids) & conns['body_post'].isin(candidate_ids)]
syn_with_patch = pd.concat([to_patch.groupby('body_pre')['weight'].sum(),
                            from_patch.groupby('body_post')['weight'].sum()]).groupby(level=0).sum()
partners = candidates[candidates['bodyId'].isin(
    syn_with_patch[syn_with_patch >= PARTNER_MIN_SYN].index)]
partners = partners.join(positions, on='bodyId', how='left')   # annotated column, if any

neurons = pd.concat([t4t5, partners], ignore_index=True)
neurons = neurons[['bodyId', 'type', 'instance', 'somaSide', 'hex1', 'hex2', 'position_source']]
neurons = neurons.loc[:, ~neurons.columns.duplicated()]

# the model sees T4 and T5 as single types; the subtypes are kept for scoring
merge = {s: t for t, subs in SUBTYPES.items() for s in subs}
neurons['subtype'] = neurons['type']
neurons['model_type'] = neurons['type'].map(lambda t: merge.get(t, t))

type_counts = neurons['model_type'].value_counts()
dropped = type_counts[type_counts < MIN_NUM_PER_TYPE]
print(f'dropping {len(dropped)} types with fewer than {MIN_NUM_PER_TYPE} neurons: {list(dropped.index)}')
neurons = neurons[neurons['model_type'].isin(type_counts[type_counts >= MIN_NUM_PER_TYPE].index)]
neurons = neurons.sort_values(['model_type', 'bodyId']).reset_index(drop=True)
print(neurons['model_type'].value_counts().to_string())
print(neurons.groupby('model_type')['subtype'].nunique().loc[['T4', 'T5']])

#%% Connectivity matrix: J[i, j] = synapses from neuron i to neuron j
index_of = pd.Series(np.arange(len(neurons)), index=neurons['bodyId'])
within = conns[conns['body_pre'].isin(index_of.index) & conns['body_post'].isin(index_of.index)]
J = np.zeros((len(neurons), len(neurons)), dtype=np.float32)
J[index_of[within['body_pre']].values, index_of[within['body_post']].values] = within['weight'].values
del conns

data = data_utils.ConnectivityData(torch.tensor(J), neurons, 'model_type')
print(f'{data.N} neurons, {data.Ntype} types, {int(J.sum())} synapses, '
      f'{(J > 0).mean():.3f} of pairs connected')

#%% Fit the embedding (or load it if it was already fitted)
save_path = os.path.join(EMBEDDING_DIR, hp['save_filename'])
if os.path.isfile(save_path):
    saved = pickle.load(open(save_path, 'rb'))
    model = saved['model']
    assert saved['data'].neurons['bodyId'].equals(data.neurons['bodyId']), \
        'saved embedding was fitted on a different selection of neurons'
    print(f"loaded {save_path}")
else:
    torch.manual_seed(hp['seed'])
    np.random.seed(hp['seed'])
    model = models.InteractionModel(N=data.N, Ntype=data.Ntype, D=hp['embedding_dim'],
                                    onehot_types=data.onehot_types,
                                    wrapper_fn=hp['wrapper_fn'])

    def reg_fn(model: models.InteractionModel):
        return torch.mean(model.embeddings**2) * hp['embedding_regularizer']

    model.use_transforms = False
    losses, _, _ = models.train_model(model, target_mat=data.J, loss_type=hp['loss_type'],
                                      lr=hp['lr'], print_every=500, reg_fn=reg_fn,
                                      steps=hp['steps'])
    model.embeddings = model.embeddings.detach()
    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    pickle.dump({'time': datetime.datetime.now(), 'model': model, 'data': data,
                 'hyperparams': hp, 'losses': losses, 'save_filename': hp['save_filename']},
                open(save_path, 'wb'))
    print(f"saved {save_path}")

embeddings = model.embeddings.detach().numpy()


#%% Score: are the subtypes and the retinotopy in the embedding?
def subtype_scores(x, labels, rng_seed):
    """k-means (k = number of subtypes) agreement and k-NN classification accuracy."""
    n_sub = len(np.unique(labels))
    clusters = KMeans(n_clusters=n_sub, n_init=20, random_state=rng_seed).fit_predict(x)
    folds = StratifiedKFold(N_FOLDS, shuffle=True, random_state=rng_seed)
    predicted = cross_val_predict(KNeighborsClassifier(N_NEIGHBORS), x, labels, cv=folds)
    silhouettes = {k: silhouette_score(x, KMeans(n_clusters=k, n_init=10, random_state=rng_seed)
                                       .fit_predict(x)) for k in K_RANGE}
    return {'kmeans_ARI': adjusted_rand_score(labels, clusters),
            'knn_accuracy': np.mean(predicted == labels),
            'chance_accuracy': pd.Series(labels).value_counts(normalize=True).max(),
            'best_k_silhouette': max(silhouettes, key=silhouettes.get)}


def retinotopy_r2(x, position, rng_seed):
    """Cross-validated R^2 of the column coordinates predicted from the embedding."""
    folds = KFold(N_FOLDS, shuffle=True, random_state=rng_seed)
    out = {}
    for name, reg in [('linear', LinearRegression()), ('knn', KNeighborsRegressor(N_NEIGHBORS))]:
        predicted = cross_val_predict(reg, x, position, cv=folds)
        out[f'retinotopy_R2_{name}'] = 1 - ((predicted - position)**2).sum() / \
            ((position - position.mean(0))**2).sum()
    return out


rows = []
for t in SUBTYPES:
    inds = data.neuron_hash[t]
    sub = data.neurons.iloc[inds]
    row = {'type': t, 'n_neurons': len(inds)}
    row.update(subtype_scores(embeddings[inds], sub['subtype'].values, SCORE_SEED))
    row.update(retinotopy_r2(embeddings[inds], sub[['hex1', 'hex2']].values, SCORE_SEED))
    rows.append(row)
scores = pd.DataFrame(rows)
print(scores.round(3).to_string(index=False))

#%% Plot: T4 and T5 embeddings (top PCs), colored by subtype and by column
for t in SUBTYPES:
    inds = data.neuron_hash[t]
    sub = data.neurons.iloc[inds]
    pcs = PCA(2).fit_transform(embeddings[inds])
    fig, axes = plt.subplots(1, 3, figsize=(4.5, 1.6), dpi=300)
    for s, color in zip(SUBTYPES[t], plt.cm.tab10.colors):
        m = (sub['subtype'] == s).values
        axes[0].scatter(pcs[m, 0], pcs[m, 1], s=1, color=color, lw=0, label=s)
    axes[0].legend(fontsize=4, frameon=False, markerscale=3)
    axes[0].set_xlabel('PC1'); axes[0].set_ylabel('PC2')
    for ax, coord in zip(axes[1:], ['hex1', 'hex2']):
        ax.scatter(pcs[:, 0], pcs[:, 1], s=1, c=sub[coord], cmap='viridis', lw=0)
        ax.set_title(f'color: {coord}', fontsize=5)
        ax.set_xlabel('PC1')
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f'{t} embeddings (subtypes merged for fitting)', fontsize=6)
    plt.tight_layout()

#%% Interactive 3D plot: T4 and T5 embeddings (PCS_3D), colored by subtype

# Interactive 3D plot: which principal components of the T4 / T5 embeddings to show
PCS_3D = (3, 4, 5)

for t in SUBTYPES:
    inds = data.neuron_hash[t]
    sub = data.neurons.iloc[inds]
    pca = PCA().fit(embeddings[inds])
    pcs = pca.transform(embeddings[inds])[:, list(PCS_3D)]
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    for s, color in zip(SUBTYPES[t], plt.cm.tab10.colors):
        m = (sub['subtype'] == s).values
        ax.scatter(pcs[m, 0], pcs[m, 1], pcs[m, 2], s=3, color=color, label=s)
    variance = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC{PCS_3D[0] + 1} ({variance[PCS_3D[0]]:.0%})')
    ax.set_ylabel(f'PC{PCS_3D[1] + 1} ({variance[PCS_3D[1]]:.0%})')
    ax.set_zlabel(f'PC{PCS_3D[2] + 1} ({variance[PCS_3D[2]]:.0%})')
    ax.set_box_aspect(np.ptp(pcs, axis=0))   # equal scale on all axes
    ax.legend()
    ax.set_title(f'{t} embeddings, colored by subtype (subtypes merged for fitting)')
    plt.show()

#%% Label-free subtype search: remove the retinotopy, then cluster
# 1. Retinotopy: regress the T4 (T5) embeddings on a polynomial of degree RETINOTOPY_DEGREE
#    in the estimated column position (hex1, hex2).
# 2. Remove it: keep the residuals, the part of each embedding not explained by position.
#    Every column holds all four subtypes, so the fit averages over them and the subtype
#    differences stay in the residuals.
# 3. Cluster the residuals (their top CLUSTER_PCS principal components) with k-means,
#    with k chosen by silhouette over K_RANGE.
# The subtype labels are not used; the agreement with them (ARI) is only reported.
cluster_rows = []
residuals, cluster_features, chosen_clusters = {}, {}, {}
for t in SUBTYPES:
    inds = data.neuron_hash[t]
    sub = data.neurons.iloc[inds]
    x = embeddings[inds]
    position = sub[['hex1', 'hex2']].values
    retinotopy = make_pipeline(PolynomialFeatures(RETINOTOPY_DEGREE), LinearRegression())
    residuals[t] = x - retinotopy.fit(position, x).predict(position)
    cluster_features[t] = (residuals[t] if CLUSTER_PCS is None
                           else PCA(CLUSTER_PCS).fit_transform(residuals[t]))

    clusterings = {k: KMeans(n_clusters=k, n_init=20, random_state=SCORE_SEED)
                      .fit_predict(cluster_features[t]) for k in K_RANGE}
    silhouettes = {k: silhouette_score(cluster_features[t], c) for k, c in clusterings.items()}
    best_k = max(silhouettes, key=silhouettes.get)
    chosen_clusters[t] = clusterings[best_k]
    print(f'{t} silhouette by k: ' + ', '.join(f'{k}: {v:.3f}' for k, v in silhouettes.items()))
    cluster_rows.append({
        'type': t,
        'variance_explained_by_position': 1 - residuals[t].var(0).sum() / x.var(0).sum(),
        'best_k': best_k,
        'silhouette': silhouettes[best_k],
        'ARI_best_k': adjusted_rand_score(sub['subtype'], clusterings[best_k]),
        'ARI_k4': adjusted_rand_score(sub['subtype'], clusterings[4]),
    })
print(pd.DataFrame(cluster_rows).round(3).to_string(index=False))

#%% Plot: residuals after removing the retinotopy (top 2 PCs), by k-means cluster and by subtype
for t in SUBTYPES:
    sub = data.neurons.iloc[data.neuron_hash[t]]
    pcs = PCA(2).fit_transform(residuals[t])
    fig, axes = plt.subplots(1, 2, figsize=(4, 2), dpi=150)
    for c, color in zip(np.unique(chosen_clusters[t]), plt.cm.Set2.colors):
        m = chosen_clusters[t] == c
        axes[0].scatter(pcs[m, 0], pcs[m, 1], s=2, color=color, lw=0, label=f'cluster {c}')
    for s, color in zip(SUBTYPES[t], plt.cm.tab10.colors):
        m = (sub['subtype'] == s).values
        axes[1].scatter(pcs[m, 0], pcs[m, 1], s=2, color=color, lw=0, label=s)
    row = [r for r in cluster_rows if r['type'] == t][0]
    features = 'all dims' if CLUSTER_PCS is None else f'top {CLUSTER_PCS} PCs'
    axes[0].set_title(f'k-means on {features} (k = {row["best_k"]}, by silhouette)', fontsize=6)
    axes[1].set_title(f'subtypes (ARI {row["ARI_best_k"]:.2f})', fontsize=6)
    for ax in axes:
        ax.set_xlabel('residual PC1', fontsize=6)
        ax.set_ylabel('residual PC2', fontsize=6)
        ax.legend(fontsize=4, frameon=False, markerscale=2)
        ax.set_aspect('equal')
    fig.suptitle(f'{t} embeddings with retinotopy removed '
                 f'(degree-{RETINOTOPY_DEGREE} fit of column position)', fontsize=6)
    plt.tight_layout()
    plt.show()


# %%
