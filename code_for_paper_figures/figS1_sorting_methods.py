#%%
"""
Sorting the EPG connectivity matrix with existing matrix-sorting methods (Fig. S1).

The methods are given only the EPG neurons, and the binarized connectivity
J >= threshold (synapse counts), for several thresholds:
  - seung_2009:        orders neurons by the potential t that best explains net
                       flow, L t = in_degree - out_degree, with L the Laplacian of
                       the symmetrized binary graph (solved with the pseudoinverse,
                       since L is singular)
  - borst_recurrency:  Borst (2024) smooth-index sorting, minimizing the length of
                       recurrent (feedback) connections
  - borst_bandwidth:   Borst (2024) smooth-index sorting, minimizing the length of
                       all connections

Each ordering is plotted as a sorted matrix and scored against the EPG neurons' PB
glomeruli: neuron k of n in the ordering is placed at angle 2 pi k / n, and the
angles are compared with anatomy as in anatomical_order.py (Kendall's tau per PB
side, with a shuffled-label null).

Run from code_for_paper_figures/.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from code_for_paper_figures import borst_2024
from connectome_embedding import plots, data_utils, anatomy

plots.set_rc_params()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DATAPATH = 'hemibrain_data/'
TYPES_WANTED = ['EPG=']
MIN_NUM_PER_TYPE = 10

THRESHOLDS = [1, 3, 5]   # a pair counts as connected if it has >= this many synapses
METHODS = ['seung_2009', 'borst_recurrency', 'borst_bandwidth']
SEED = 1                 # Borst sorting starts from a random initialization

# Scoring against anatomy (same settings as anatomical_order.py)
PB_PERIOD = 8
WINDINGS = (1, 0.5, 2)
MIN_NEURONS = 4
MIN_POSITIONS = 3
N_SHUFFLES = 1000

#%% Load the EPG connectivity
W_all, neuronsall = data_utils.get_W_all_neuronall(
    DATAPATH, min_num_per_type=MIN_NUM_PER_TYPE)
data = data_utils.prep_connectivity_data(W_all, neuronsall, TYPES_WANTED, split_LR=[])
J = data.J.numpy()           # J[i, j] = synapses from neuron i to neuron j
np.fill_diagonal(J, 0)       # no self-connections
print(f'{data.N} neurons, {data.Ntype} types')

labels = anatomy.anatomical_labels(data.neurons, pb_period=PB_PERIOD)


#%% Sorting methods
def seung_2009_order(M):
    """Order by the potential t solving L t = in_degree - out_degree (pseudoinverse)."""
    symmetric = M + M.T
    L = np.diag(symmetric.sum(1)) - symmetric
    net_flow = M.sum(0) - M.sum(1)      # in-degree minus out-degree
    t = np.linalg.pinv(L) @ net_flow
    return np.argsort(t)


def borst_order(M, mode, seed):
    """Order from Borst (2024) smooth-index sorting."""
    np.random.seed(seed)
    _, order = borst_2024.SI_sort(M, mode=mode, return_object='both')
    return order


def sort_order(M, method, seed):
    if method == 'seung_2009':
        return seung_2009_order(M)
    if method == 'borst_recurrency':
        return borst_order(M, 'recurrency', seed)
    if method == 'borst_bandwidth':
        return borst_order(M, 'bandwidth', seed)
    raise ValueError(method)


#%% Sort, and score each ordering against the PB glomeruli
rng = np.random.default_rng(SEED)
orders = {}
score_rows = []
for threshold in THRESHOLDS:
    M = (J >= threshold).astype(float)
    for method in METHODS:
        order = sort_order(M, method, SEED)
        orders[(method, threshold)] = order

        # neuron at position k of the ordering gets angle 2 pi k / n
        angles = np.zeros(data.N)
        angles[order] = 2 * np.pi * np.arange(data.N) / data.N
        labels['angle'] = angles

        scores = anatomy.score_groups(labels, 'angle', rng, windings=WINDINGS,
                                      min_neurons=MIN_NEURONS, min_positions=MIN_POSITIONS,
                                      n_shuffles=N_SHUFFLES)
        score_rows.append(scores.assign(method=method, threshold=threshold))

scores = pd.concat(score_rows, ignore_index=True)
print(scores[['method', 'threshold', 'type', 'pb_side', 'tau', 'tau_null_median', 'tau_null_hi']]
      .to_string(index=False))

#%% Plot: sorted binarized matrices, one row per method, one column per threshold
fig, axes = plt.subplots(len(METHODS), len(THRESHOLDS),
                         figsize=(1.3 * len(THRESHOLDS), 1.3 * len(METHODS)), dpi=300,
                         squeeze=False)
for i, method in enumerate(METHODS):
    for j, threshold in enumerate(THRESHOLDS):
        order = orders[(method, threshold)]
        M = (J >= threshold).astype(float)
        ax = axes[i][j]
        ax.imshow(M[order][:, order], cmap='gray_r', vmin=0, vmax=1, interpolation='none')
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_title(f'>= {threshold} synapses', fontsize=5)
        if j == 0:
            ax.set_ylabel(method, fontsize=5)
plt.tight_layout()

#%% Plot: Kendall's tau with PB order for each method and threshold
plt.figure(figsize=(2.2, 1.5), dpi=300)
for k, method in enumerate(METHODS):
    sub = scores[scores['method'] == method]
    x = np.array([THRESHOLDS.index(t) for t in sub['threshold']]) + (k - 1) * 0.2
    plt.scatter(x, sub['tau'], s=4, label=method, zorder=3)
null = scores.groupby('threshold')['tau_null_hi'].max()
plt.plot(range(len(THRESHOLDS)), null.values, color='gray', lw=0.5,
         label='shuffled, 97.5th pct.')
plt.xticks(range(len(THRESHOLDS)), [f'>= {t}' for t in THRESHOLDS])
plt.xlabel('synapses per connection')
plt.ylabel("Kendall's tau with\nPB order")
plt.ylim(0, 1.05)
plt.legend(fontsize=4, frameon=False)
plt.tight_layout()
