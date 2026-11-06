#%%
"""
How well do simpler methods recover the anatomical order of central complex
neurons, compared with the embedding model?

Every method gives each neuron an angle, which is scored against the neuron's PB
glomerulus or FB column in the same way as in anatomical_order.py
(connectome_embedding.anatomy: Kendall's tau between the embedding and anatomical
order per cell type and PB side, with a shuffled-label null).

Methods
  Within one cell type, on each neuron's connection weights (log(1 + synapses)):
    features 'in'      synapses received from every neuron in the dataset (column of J)
    features 'out'     synapses sent to every neuron in the dataset (row of J)
    features 'in+out'  both, concatenated
    - pca:          top two principal components ('in', 'out', 'in+out')
    - eigenmap:     Laplacian eigenmaps on a k-nearest-neighbour graph of the features
                    ('in', 'out', 'in+out')
    - isomap:       Isomap on the same k-nearest-neighbour graph, on output weights
                    only ('out'), as in Farrell et al. 2021
  On all neurons at once:
    - spectral:            eigenvectors of the normalized graph Laplacian of the
                           symmetrized connectivity (J + J^T) / 2, for the smallest
                           non-zero eigenvalues
    - spectral_block_norm: the same after dividing each block of J between a pair of
                           cell types by its mean, so that differences between types
                           do not dominate
  The embedding model (EMBEDDING_NAME).

Angle read-out: for the 2D within-type methods, the angle of each neuron in that
2D space. For the spectral methods, the angle in the top two principal components
of each cell type's rows of the eigenvectors. For the embedding model, the angle
used for sorting in the figures (all embeddings projected on the top two principal
components of the REFERENCE_TYPE embeddings, as in anatomical_order.py).

Run from code_for_paper_figures/. Plots tau for each method.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse.csgraph import laplacian
from sklearn.manifold import SpectralEmbedding, Isomap
from connectome_embedding import plots, anatomy

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
EMBEDDING_NAME = 'embeddings/whole_hemibrain_embedding.pkl'   # embedding model; its data (J, neurons) is used for all methods

# the sorting angle projects all embeddings on the top two PCs of this type's embeddings
REFERENCE_TYPE = 'EPG'

# Within-type methods and the connection-weight features each one is run on
WITHIN_TYPE_METHODS = {
    'pca': ['in', 'out', 'in+out'],
    'eigenmap': ['in', 'out', 'in+out'],
    'isomap': ['out'],
}

N_NEIGHBORS = 10      # k for the nearest-neighbour graphs (capped at n_neurons - 1)
N_SPECTRAL = 5        # eigenvectors kept for the whole-dataset spectral methods
SEED = 0

# Scoring against anatomy (same settings as anatomical_order.py)
PB_PERIOD = 8
WINDINGS = (1, 2)
MIN_NEURONS = 4
MIN_POSITIONS = 3
N_SHUFFLES = 0   # shuffles for the null on tau; 0 skips the null (set e.g. 1000 to compute it)

# Methods that embed all neurons simultaneously; the rest embed one cell type at a time
ALL_NEURON_METHODS = ['our method', 'spectral', 'spectral_block_norm']
SCOPE_COLORS = {'all neurons at once': 'tab:blue', 'one cell type at a time': 'k'}

# Cell types whose dots are labelled in the dot plot
LABELLED_TYPES = ('PEG',)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def angle_2d(coords):
    """Angle of each row of an n x 2 array around the mean of the rows."""
    centered = coords - coords.mean(0)
    return np.arctan2(centered[:, 1], centered[:, 0])


def angle_top2_pcs(coords):
    """Angle of each row in the top two principal components of an n x d array."""
    centered = coords - coords.mean(0)
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return angle_2d(centered @ Vt[:2].T)


def type_features(J_log, inds, features):
    """Connection weights of the neurons in inds: rows of J are outputs, columns inputs."""
    out_weights = J_log[inds, :]
    in_weights = J_log[:, inds].T
    if features == 'in':
        return in_weights
    if features == 'out':
        return out_weights
    return np.hstack([in_weights, out_weights])


def within_type_coords(X, method, seed):
    """2D coordinates of the rows of X (n neurons x features) for one method."""
    n = X.shape[0]
    k = min(N_NEIGHBORS, n - 1)
    if method == 'pca':
        centered = X - X.mean(0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        return centered @ Vt[:2].T
    if method == 'eigenmap':
        return SpectralEmbedding(n_components=2, affinity='nearest_neighbors',
                                 n_neighbors=k, random_state=seed).fit_transform(X)
    if method == 'isomap':
        return Isomap(n_components=2, n_neighbors=k).fit_transform(X)
    raise ValueError(method)


def spectral_coords(J, n_components):
    """
    Laplacian eigenmap of the whole connectome: eigenvectors of the normalized
    Laplacian of (J + J^T) / 2 for the smallest non-zero eigenvalues.
    """
    W = (J + J.T) / 2
    L = laplacian(W, normed=True)
    eigvals, eigvecs = np.linalg.eigh(L)   # eigenvalues in increasing order
    # skip the eigenvalue(s) at 0 (one per connected component)
    first = np.searchsorted(eigvals, 1e-8)
    return eigvecs[:, first:first + n_components]


def block_normalize(J, onehot):
    """Divide each block of J between a pair of cell types by that block's mean."""
    counts = onehot.sum(0)
    block_sums = onehot.T @ J @ onehot                 # Ntype x Ntype total synapses
    block_means = block_sums / np.outer(counts, counts)
    per_pair_mean = onehot @ block_means @ onehot.T   # N x N, the mean of each pair's block
    return np.divide(J, per_pair_mean, out=np.zeros_like(J), where=per_pair_mean > 0)


def per_type_angles(coords, data, types):
    """Angle of each neuron in the top two principal components of its type's coordinates."""
    angles = np.full(data.N, np.nan)
    for t in types:
        inds = data.neuron_hash[t]
        if len(inds) >= 3:
            angles[inds] = angle_top2_pcs(coords[inds])
    return angles


def within_type_angles(J_log, data, types, method, features):
    """Angle of each neuron from running one within-type method on each cell type separately."""
    angles = np.full(data.N, np.nan)
    for t in types:
        inds = data.neuron_hash[t]
        if len(inds) < MIN_NEURONS:
            continue
        X = type_features(J_log, inds, features)
        try:
            angles[inds] = angle_2d(within_type_coords(X, method, SEED))
        except Exception as e:   # e.g. a disconnected neighbour graph
            print(f'{method} ({features}) failed for {t}: {e}')
    return angles


def scope_of(method):
    return 'all neurons at once' if method in ALL_NEURON_METHODS else 'one cell type at a time'


def method_label(method, feature):
    return f'{method}, {feature}' if feature else method


def scope_legend(marker):
    return [plt.Line2D([], [], color=c, marker=marker, ls='', ms=3, label=scope)
            for scope, c in SCOPE_COLORS.items()]


#%% Load the data and the embedding model
plots.set_rc_params()
rng = np.random.default_rng(SEED)

model_and_data = pickle.load(open(EMBEDDING_NAME, 'rb'))
model = model_and_data['model']
data = model_and_data['data']
embeddings = model.embeddings.detach()

J = data.J.numpy().astype(float)   # J[i, j] = synapses from neuron i to neuron j
J_log = np.log1p(J)
onehot = data.onehot_types.numpy()

labels = anatomy.add_fb_angles(anatomy.anatomical_labels(data.neurons, pb_period=PB_PERIOD))

# only cell types with anatomical labels need angles
labelled = labels['pb_glomerulus'].notna() | labels['fb_column'].notna()
types = [t for t in data.types if labelled[data.neuron_hash[t]].any()]
print(f'{len(types)} cell types with anatomical labels')

#%% Angles from every method
angle_columns = {}   # column name in labels -> (method, features)

# embedding model: the sorting angle used in the figures
labels['angle_model'] = anatomy.pca_angles(embeddings, data, REFERENCE_TYPE)
angle_columns['angle_model'] = ('our method', '')

# within-type methods on connection weights
for method, feature_sets in WITHIN_TYPE_METHODS.items():
    for features in feature_sets:
        column = f'angle_{method}_{features}'
        labels[column] = within_type_angles(J_log, data, types, method, features)
        angle_columns[column] = (method, features)

# whole-dataset spectral methods
labels['angle_spectral'] = per_type_angles(spectral_coords(J, N_SPECTRAL), data, types)
angle_columns['angle_spectral'] = ('spectral', 'all neurons')
labels['angle_spectral_block_norm'] = per_type_angles(
    spectral_coords(block_normalize(J, onehot), N_SPECTRAL), data, types)
angle_columns['angle_spectral_block_norm'] = ('spectral_block_norm', 'all neurons')

#%% Score every method against anatomy
scores = pd.concat(
    [anatomy.score_groups(labels, column, rng, windings=WINDINGS,
                          min_neurons=MIN_NEURONS, min_positions=MIN_POSITIONS,
                          n_shuffles=N_SHUFFLES).assign(method=method, features=features)
     for column, (method, features) in angle_columns.items()],
    ignore_index=True)
scores['above_null'] = scores['tau'] > scores['tau_null_hi'] if N_SHUFFLES > 0 else np.nan
scores['scope'] = scores['method'].map(scope_of)

summary = scores.groupby(['method', 'features']).agg(
    median_tau=('tau', 'median'),
    frac_above_null=('above_null', lambda s: s.astype(float).mean()),
    n_groups=('tau', 'size'),
).sort_values('median_tau', ascending=False)
print(summary.to_string())

# PEG has no structured connections within its own type
print(scores[scores['type'].str.startswith('PEG')]
      [['method', 'features', 'type', 'pb_side', 'tau', 'tau_null_hi']].to_string(index=False))

#%% Plot: tau across groups for each method (box plot)
order = summary.index.tolist()   # (method, features), best median tau first

plt.figure(figsize=(0.25 * len(order) + 1, 2), dpi=300)
taus = [scores[(scores['method'] == m) & (scores['features'] == f)]['tau'].values
        for m, f in order]
boxes = plt.boxplot(taus, positions=range(len(order)), widths=0.6, showfliers=False,
                    whis=(5, 95), patch_artist=True, medianprops=dict(color='k', lw=1),
                    whiskerprops=dict(lw=0.5), capprops=dict(lw=0.5), boxprops=dict(lw=0.5))
for box, (method, _) in zip(boxes['boxes'], order):
    box.set_facecolor(SCOPE_COLORS[scope_of(method)])
    box.set_alpha(0.4)
plt.xticks(range(len(order)), [method_label(m, f) for m, f in order], rotation=90, fontsize=5)
for tick, (method, _) in zip(plt.gca().get_xticklabels(), order):
    tick.set_color(SCOPE_COLORS[scope_of(method)])
plt.ylim(0, 1.05)
plt.ylabel("Kendall's tau with\nanatomical order", fontsize=5)
plt.title('box: 25th-75th percentile across groups; line: median; whiskers: 5th-95th percentile',
          fontsize=5)
plt.legend(handles=scope_legend('s'), fontsize=5, frameon=False, loc='lower right')
plt.tight_layout()
