#%%
"""
Held-out log-likelihood of the embedding model vs number of parameters, with and
without cell-type information (Fig. 2), on the central complex connectome.

For each random seed, a random TRAIN_FRACTION of all neuron pairs (excluding
self-connections) is used to fit the model and the remaining pairs are held out.
The seed sets both the split and the initialization. Both models are fit on the
same split:
  - with cell types:    kernel parameters (A, B, C) for every pair of cell types
  - without cell types: all neurons treated as a single type (one A, B, C)
Rotations are not fit here.

Reported per run: Poisson log-likelihood per neuron pair on the training and
held-out pairs, after the last training step. The constant log(y!) term is
omitted, so values are comparable across models but are not absolute
log-likelihoods.

Run from code_for_paper_figures/. Saves each fitted embedding to SAVE_DIR (same
format as the pickles from embed_hemibrain.py, plus the train/test split and
log-likelihoods). Runs that already have a pickle are skipped. The plots are built
from those pickles.
"""
import os, pickle, datetime
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from connectome_embedding import models, plots, data_utils

plots.set_rc_params()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DATAPATH = 'hemibrain_data/'

# Central complex cell types (same selection as the 'central_complex' config in
# embed_hemibrain.py); '=' means exact match, '*' means wildcard match
TYPES_WANTED = ["EPG=", 'Delta7', 'PEG', 'EPGt', 'PEN', 'FC', 'PFL', 'PFN',
                'FB', 'hDelta', 'vDelta', 'FR*', 'EL', 'FS', 'PFG',
                'PFR', 'IbSpsP']
MIN_NUM_PER_TYPE = 10
SPLIT_LR = []  # no L/R split

DS = [3, 5, 7, 9]          # embedding dimensions to compare
DS_WITHOUT_TYPES = [3, 5, 7, 9, 11]  # without cell types: skip D = 2, add D = 11
SEEDS = [0, 1, 2, 3, 4, 5]       # one train/test split and initialization per seed
TRAIN_FRACTION = 0.8          # fraction of neuron pairs used for training

TRAINING = {
    'wrapper_fn': 'softplus',
    'embedding_regularizer': 0.01,
    'lr': 0.05,
    'steps': 10000,
    'loss_type': 'poisson',
}

SAVE_DIR = 'fig2_embeddings'                 # one pickle per fitted embedding


def embedding_filename(D, seed, use_cell_types):
    types_tag = 'with_types' if use_cell_types else 'without_types'
    return os.path.join(SAVE_DIR, f'CX_D{D}_seed{seed}_{types_tag}.pkl')


def dims_for(use_cell_types):
    return DS if use_cell_types else DS_WITHOUT_TYPES


os.makedirs(SAVE_DIR, exist_ok=True)

#%% Load the central complex connectome
W_all, neuronsall = data_utils.get_W_all_neuronall(
    DATAPATH, min_num_per_type=MIN_NUM_PER_TYPE)
data = data_utils.prep_connectivity_data(
    W_all, neuronsall, TYPES_WANTED, split_LR=SPLIT_LR)
print(f'{data.N} neurons, {data.Ntype} types')


#%% Helpers
def make_split(N, train_fraction, seed):
    """
    Random train/test split of all neuron pairs (i, j) with i != j.
    Returns two N x N boolean masks that together cover every off-diagonal pair.
    """
    rng = np.random.default_rng(seed)
    off_diagonal = ~np.eye(N, dtype=bool)
    # each off-diagonal pair goes to training with probability train_fraction
    in_train = rng.random((N, N)) < train_fraction
    train_mask = off_diagonal & in_train
    test_mask = off_diagonal & ~in_train
    return train_mask, test_mask


def poisson_log_likelihood(model, target_mat, mask):
    """
    Mean Poisson log-likelihood per neuron pair over the pairs in mask,
    without the constant log(y!) term: mean of y * log(mu) - mu.
    """
    poisson_nll = torch.nn.PoissonNLLLoss(log_input=False)
    with torch.no_grad():
        preds = model.pred_synapses()
        mask = torch.as_tensor(mask)
        return -poisson_nll(preds[mask], target_mat[mask]).item()


def fit_and_evaluate(D, seed, use_cell_types, train_mask, test_mask):
    """
    Fit one model on the training pairs, score it on both sets of pairs, and save it
    (with its scores) to embedding_filename(D, seed, use_cell_types).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    if use_cell_types:
        Ntype, onehot_types = data.Ntype, data.onehot_types
    else:
        # every neuron in a single type, so one set of kernel parameters for all pairs
        Ntype, onehot_types = 1, torch.ones((data.N, 1))

    # all settings of this run, in the same format as embed_hemibrain.py
    hp = {
        'types_wanted': TYPES_WANTED,
        'min_num_per_type': MIN_NUM_PER_TYPE,
        'embedding_dim': D,
        'wrapper_fn': TRAINING['wrapper_fn'],
        'embedding_regularizer': TRAINING['embedding_regularizer'],
        'lr': TRAINING['lr'],
        'steps': TRAINING['steps'],
        'loss_type': TRAINING['loss_type'],
        'split_LR': SPLIT_LR,
        'fit_rotations': False,
        'seed': seed,
        'use_cell_types': use_cell_types,
        'train_fraction': TRAIN_FRACTION,
        'save_filename': embedding_filename(D, seed, use_cell_types),
    }

    model = models.InteractionModel(N=data.N, Ntype=Ntype, D=D,
                                    onehot_types=onehot_types,
                                    wrapper_fn=hp['wrapper_fn'])
    model.use_transforms = False  # no rotations

    def reg_fn(model):
        return torch.mean(model.embeddings**2) * hp['embedding_regularizer']

    losses, _, _ = models.train_model(
        model, target_mat=data.J,
        loss_type=hp['loss_type'], lr=hp['lr'],
        steps=hp['steps'], print_every=hp['steps'] // 10,
        reg_fn=reg_fn, train_mask=train_mask, test_mask=test_mask)

    model.embeddings = model.embeddings.detach()

    train_loglik = poisson_log_likelihood(model, data.J, train_mask)
    test_loglik = poisson_log_likelihood(model, data.J, test_mask)

    # save the fitted embedding (data holds the real cell types even when the model
    # was fit without them; hp['use_cell_types'] says which)
    results_to_save = {
        'time': datetime.datetime.now(),
        'model': model,
        'data': data,
        'hyperparams': hp,
        'losses': losses,
        'save_filename': hp['save_filename'],
        'train_mask': train_mask,
        'test_mask': test_mask,
        'train_loglik': train_loglik,
        'test_loglik': test_loglik,
    }
    pickle.dump(results_to_save, open(hp['save_filename'], 'wb'))


def load_results(seeds=SEEDS):
    """
    Rebuild the summary table (one row per run) from the
    embedding pickles in SAVE_DIR. Runs whose pickle is missing are skipped.
    """
    rows = []
    for seed in seeds:
        for use_cell_types in [True, False]:
            for D in dims_for(use_cell_types):
                filename = embedding_filename(D, seed, use_cell_types)
                if not os.path.isfile(filename):
                    print(f'missing: {filename}')
                    continue
                with open(filename, 'rb') as f:
                    saved = pickle.load(f)
                Ntype = saved['data'].Ntype if saved['hyperparams']['use_cell_types'] else 1
                rows.append({
                    'D': D,
                    'seed': seed,
                    'cell_types': use_cell_types,
                    'n_params': saved['data'].N * D + 3 * Ntype**2,
                    'train_loglik': saved['train_loglik'],
                    'test_loglik': saved['test_loglik'],
                    'embedding_file': filename,
                })
    return pd.DataFrame(rows)


#%% Fit every combination of seed, D and with/without cell types
# Runs whose pickle already exists are skipped, never overwritten, so only new runs
# (e.g. a newly added D) are fit when the script is run again.
for seed in SEEDS:
    # same split for every D and both models with this seed
    train_mask, test_mask = make_split(data.N, TRAIN_FRACTION, seed)
    for use_cell_types in [True, False]:
        for D in dims_for(use_cell_types):
            if os.path.isfile(embedding_filename(D, seed, use_cell_types)):
                print(f'seed {seed}, D = {D}, cell types: {use_cell_types}: already fit, skipping')
                continue
            print(f'seed {seed}, D = {D}, cell types: {use_cell_types}')
            fit_and_evaluate(D, seed, use_cell_types, train_mask, test_mask)

#%% Load the results from the saved pickles
results = load_results()
print(f'Loaded {len(results)} runs from {SAVE_DIR}')
print(results.groupby(['cell_types', 'D'])[['train_loglik', 'test_loglik']].agg(['mean', 'std']))


#%% Plot: log-likelihood vs number of parameters, mean across seeds with error bars (std across seeds)
def plot_loglik_vs_n_params(results, column, ylabel):
    plt.figure(dpi=300, figsize=(2, 1.5))
    for use_cell_types, color, label in [(True, 'r', 'with cell types'),
                                         (False, 'k', 'without cell types')]:
        subset = results[results['cell_types'] == use_cell_types]
        # n_params is the same for every seed at a given D and model
        stats = subset.groupby('n_params')[column].agg(['mean', 'std'])
        plt.errorbar(stats.index, stats['mean'], yerr=stats['std'], color=color,
                     marker='o', markersize=2, capsize=2, lw=1, elinewidth=0.7, label=label)
    plt.xlabel('number of parameters')
    plt.ylabel(ylabel)
    plt.legend(fontsize=6)
    plt.tight_layout()

plot_loglik_vs_n_params(results, 'test_loglik', 'log-likelihood\nper pair (held out)')
plot_loglik_vs_n_params(results, 'train_loglik', 'log-likelihood\nper pair (training)')

# %%
