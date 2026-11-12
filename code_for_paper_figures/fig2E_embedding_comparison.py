#%%
"""
Fig. 2E: EPG, Delta7 and PEG neurons in three embeddings of the central complex.

All three embeddings are computed on all central complex neurons; only EPG, Delta7
and PEG neurons are plotted.
  First:  spectral embedding (utils.spectral_coords): eigenvectors of the normalized
          graph Laplacian of the symmetrized connectivity (J + J^T) / 2 for the two
          smallest non-zero eigenvalues.
  Second: the same after dividing each block of J between a pair of cell types by
          its mean (utils.block_normalize), so that differences between types do not
          dominate.
  Third:  distance-based embedding without cell-type-specific parameters (all neurons
          one type), top two principal components.
  Fourth: distance-based embedding with cell-type-specific parameters, top two
          principal components.
The two model embeddings are the D = 5 fits of fig2_model_hyperparam_comparison.py
(same data, settings, seed and 80% training pairs; they differ only in the cell-type
parameters), so the panel matches the models of Fig. 2B-C.
Principal components are computed over all neurons of the embedding.
Run from code_for_paper_figures/.
"""
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
from connectome_embedding import plots, utils

plots.set_rc_params()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
EMBEDDING_DIR = 'embeddings/fig2_model_hyperparam_comparison/'
WITH_TYPES = 'CX_D5_seed0_with_types.pkl'
WITHOUT_TYPES = 'CX_D5_seed0_without_types.pkl'
TYPES_SHOWN = {'EPG': 'C0', 'Delta7': 'C1', 'PEG': 'C2'}


def load(filename):
    loaded = pickle.load(open(os.path.join(EMBEDDING_DIR, filename), 'rb'))
    return loaded['model'], loaded['data']


def top_pcs(embeddings, n=2):
    """Coordinates on the top n principal components, computed over all neurons."""
    centered = embeddings - embeddings.mean(0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return centered @ vh[:n].T


#%% Embeddings
model_with, data = load(WITH_TYPES)
model_without, data_without = load(WITHOUT_TYPES)
assert data_without.N == data.N and all(
    np.array_equal(data.neuron_hash[t], data_without.neuron_hash[t]) for t in TYPES_SHOWN)

J = data.J.numpy().astype(np.float64)
coords = {
    'spectral embedding': utils.spectral_coords(J, 2),
    'spectral, block-normalized': utils.spectral_coords(
        utils.block_normalize(J, data.onehot_types.numpy().astype(np.float64)), 2),
    'without cell types': top_pcs(model_without.embeddings.detach().numpy()),
    'with cell types': top_pcs(model_with.embeddings.detach().numpy()),
}
axis_labels = {
    'spectral embedding': ('Laplacian eigenvector 1', 'Laplacian eigenvector 2'),
    'spectral, block-normalized': ('Laplacian eigenvector 1', 'Laplacian eigenvector 2'),
    'without cell types': ('embedding PC 1', 'embedding PC 2'),
    'with cell types': ('embedding PC 1', 'embedding PC 2'),
}
print(f'{data.N} neurons, {data.Ntype} types; plotted: '
      + ', '.join(f'{t} ({len(data.neuron_hash[t])})' for t in TYPES_SHOWN))


#%% Plot
fig, axes = plt.subplots(1, len(coords), figsize=(2 * len(coords), 2), dpi=300)
for ax, (title, xy) in zip(axes, coords.items()):
    for cell_type, color in TYPES_SHOWN.items():
        inds = data.neuron_hash[cell_type]
        ax.scatter(xy[inds, 0], xy[inds, 1], c=color, s=4, lw=0, label=cell_type)
    ax.set_title(title, fontsize=6)
    ax.set_xlabel(axis_labels[title][0], fontsize=6)
    ax.set_ylabel(axis_labels[title][1], fontsize=6)
    ax.set_aspect('equal', adjustable='datalim')
axes[0].legend(fontsize=5, frameon=False)
plt.tight_layout()
plt.show()
