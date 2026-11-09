#%%
"""
Persistent homology of the embeddings learned from the synthetic grid-cell
connectome (excitatory and inhibitory neurons on a twisted sheet, as in
fig7_idealized_grid_cell.py), without specifying a target shape.

A torus has two long-lived H1 features (loops) and one long-lived H2 feature (a
void). Each population's embeddings are compared with Gaussian point clouds that
have the same number of points and the same PCA spectrum (no loops or voids).
As a reference, the same is computed for points sampled exactly on a torus
(grid_cell.make_clifford_torus).

Needs ripser (pip install ripser). Run from code_for_paper_figures/.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from connectome_embedding import models, plots, grid_cell, topology

plots.set_rc_params()

# ---------------------------------------------------------------------------
# Settings (synthetic connectome and training as in fig7_idealized_grid_cell.py)
# ---------------------------------------------------------------------------
n_e = 30   # n_e^2 excitatory neurons
n_i = 20   # n_i^2 inhibitory neurons
N_e, N_i = n_e**2, n_i**2
lambda_w = n_e / 128 * 13
beta = 3.

D = 5
TRAINING = {'lr': 0.05, 'steps': 4000, 'embedding_regularizer': 0.1, 'loss_type': 'poisson'}

MAX_POINTS = 300   # subsample each population to keep ripser fast
N_NULL = 20        # matched Gaussian clouds per population
SEED = 0

# features compared with the null: longest H1, second-longest H1, longest H2
FEATURES = [(1, 0), (1, 1), (2, 0)]
FEATURE_NAMES = ['H1, longest', 'H1, 2nd', 'H2, longest']

#%% Synthetic grid-cell connectome and its embedding
torch.manual_seed(SEED)
np.random.seed(SEED)

W = grid_cell.make_synthetic_grid_cell_weights(
    n_excitatory=n_e, n_inhibitory=n_i, lambda_w=lambda_w, beta=beta)

type_onehot = torch.zeros((N_e + N_i, 2))
type_onehot[:N_e, 0] = 1
type_onehot[N_e:, 1] = 1

model = models.InteractionModel(N=N_e + N_i, Ntype=2, D=D, onehot_types=type_onehot)
model.use_transforms = False

def reg_fn(model):
    return torch.mean(model.embeddings**2) * TRAINING['embedding_regularizer']

models.train_model(model, W, lr=TRAINING['lr'], print_every=500, reg_fn=reg_fn,
                   steps=TRAINING['steps'], loss_type=TRAINING['loss_type'])
embeddings = model.embeddings.detach().numpy()

#%% Persistent homology of each population, and of an exact torus for reference
rng = np.random.default_rng(SEED)
clouds = {
    'excitatory': embeddings[:N_e],
    'inhibitory': embeddings[N_e:],
    'exact torus': grid_cell.make_clifford_torus(n_e).numpy(),
}

rows = []
all_diagrams = {}
for name, points in clouds.items():
    x = topology.normalize(topology.subsample(points, MAX_POINTS, rng))
    life, diagrams = topology.lifetimes(x, maxdim=2)
    all_diagrams[name] = diagrams
    null = topology.null_lifetimes(x, N_NULL, rng, features=FEATURES, maxdim=2)
    for f, (dim, k) in enumerate(FEATURES):
        rows.append({
            'population': name,
            'feature': FEATURE_NAMES[f],
            'lifetime': topology.kth_longest(life, dim, k),
            'null_median': np.median(null[:, f]),
            'null_95': np.percentile(null[:, f], 95),
        })
    print(name, 'done')

results = pd.DataFrame(rows)
results['above_null'] = results['lifetime'] > results['null_95']

#%% Plot: lifetimes of the torus features vs the matched-Gaussian null
fig, axes = plt.subplots(1, len(clouds), figsize=(1.4 * len(clouds), 1.5), dpi=300,
                         sharey=True, squeeze=False)
for ax, name in zip(axes[0], clouds):
    sub = results[results['population'] == name]
    x = np.arange(len(sub))
    ax.scatter(x, sub['lifetime'], s=6, color='r', zorder=3, label='embedding')
    ax.scatter(x, sub['null_95'], s=12, marker='_', color='k', zorder=2,
               label='matched Gaussian,\n95th pct.')
    ax.set_xticks(x)
    ax.set_xticklabels(sub['feature'], rotation=90, fontsize=5)
    ax.set_xlim(-0.5, len(sub) - 0.5)
    ax.set_title(name, fontsize=5)
# y-axis is shared: set its range once, after all populations are plotted
y_top = results[['lifetime', 'null_95']].max().max()
axes[0][0].set_ylim(0, 1.05 * y_top)
axes[0][0].set_ylabel('lifetime\n(units of RMS radius)')
axes[0][-1].legend(fontsize=4, frameon=False, loc='center left', bbox_to_anchor=(1.02, 0.5))
plt.tight_layout()
