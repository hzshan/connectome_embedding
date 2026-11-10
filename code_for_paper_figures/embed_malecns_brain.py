#%%
"""
Embedding of the maleCNS brain (central brain and optic lobes, no VNC): every
traced, typed neuron whose superclass belongs to the brain, in types with at least
MIN_NUM_PER_TYPE neurons. About 116,000 neurons in 846 types (v0.9, minconf 0.5).

At this size the connectivity matrix is kept sparse (a dense float32 matrix would
take about 54 GB), and the model is trained with neuron-pair sampling
(models.train_model with pairs_per_step): each step uses an unbiased estimate of
the full Poisson loss from a random draw of connected and unconnected pairs.

The saved pickle has the format of embed_hemibrain.py (time, model, data,
hyperparams, losses, save_filename), except that two large matrices are not
saved: the connectivity matrix (data.J is None) and the N x Ntype one-hot matrix
of cell types (data.onehot_types and model.onehot_types are None). Rebuild them
with
    data.J = data_utils.sparse_connectivity(data.neurons['bodyId'], conns)
    data.onehot_types = data_utils.ConnectivityData.make_onehot_type_vectors(
        data.neurons, 'type')[0]
    model.onehot_types = data.onehot_types
(this script does so when it loads a saved embedding). Without fit_rotations, the
model's rotation_params and translation_vecs, which are then not fitted, are None.

Connection weights are total synapse counts between two neurons over the whole
CNS, so connections of brain neurons made in the VNC (e.g. between descending
neurons) are included.

Run from code_for_paper_figures/. If the embedding file exists, it is loaded
instead of refitted.
"""
import os
import pickle
import datetime
import numpy as np
import pandas as pd
import torch
from connectome_embedding import models, data_utils

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
DATAPATH = os.path.expanduser('~/Dropbox/Data/data_maleCNS/')
ANNOTATION_FILE = 'body-annotations-male-cns-v0.9-minconf-0.5.feather'
WEIGHTS_FILE = 'connectome-weights-male-cns-v0.9-minconf-0.5.feather'
EMBEDDING_DIR = 'embeddings/'   # the fitted embedding is saved here

# superclasses of neurons with their soma in the brain (central brain and optic
# lobes). Ascending neurons (soma in the VNC) are not included.
BRAIN_SUPERCLASSES = [
    'cb_intrinsic', 'cb_sensory', 'cb_motor', 'cb_efferent', 'endocrine',
    'descending_neuron', 'visual_projection', 'visual_projection_tbc',
    'visual_centrifugal', 'ol_intrinsic', 'ol_sensory',
]
MIN_NUM_PER_TYPE = 10   # drop types with fewer neurons in the selection

hp = {
    'dataset': 'maleCNS v0.9 minconf 0.5',
    'superclasses': BRAIN_SUPERCLASSES,
    'min_num_per_type': MIN_NUM_PER_TYPE,
    'embedding_dim': 7,
    'wrapper_fn': 'softplus',
    'embedding_regularizer': 0.01,
    'lr': 0.05,
    'lr_end': 0.005,        # learning rate decays exponentially from lr to lr_end
    'steps': 20000,
    'loss_type': 'poisson',
    'fit_rotations': False,
    'pairs_per_step': (1_000_000, 1_000_000),   # (connected, unconnected) pairs per step
    'seed': 0,              # initialization and pair sampling
    'connectivity_saved': False,   # data.J is not in the pickle
    'onehot_types_saved': False,   # data.onehot_types, model.onehot_types are not in the pickle
}
hp['save_filename'] = (f"malecns_brain_D{hp['embedding_dim']}"
                       f"_reg{hp['embedding_regularizer']:g}_seed{hp['seed']}.pkl")


#%% Neurons: traced, typed brain neurons in types with >= MIN_NUM_PER_TYPE neurons
ann = pd.read_feather(DATAPATH + ANNOTATION_FILE)
ann = ann[(ann['status'] == 'Traced') & ann['type'].notna()]
ann = ann[~ann['type'].str.contains('_unclear')]
neurons = ann[ann['superclass'].isin(BRAIN_SUPERCLASSES)]

type_counts = neurons['type'].value_counts()
neurons = neurons[neurons['type'].isin(type_counts[type_counts >= MIN_NUM_PER_TYPE].index)]
neurons = neurons.sort_values(['type', 'bodyId']).reset_index(drop=True)
print(f'{len(neurons)} neurons in {neurons["type"].nunique()} types')
print(neurons['superclass'].value_counts().to_string())


#%% Sparse connectivity matrix: J[i, j] = synapses from neuron i to neuron j
conns = pd.read_feather(DATAPATH + WEIGHTS_FILE)
J = data_utils.sparse_connectivity(neurons['bodyId'], conns)
del conns
J.setdiag(0)   # self-connections are not modelled
J.eliminate_zeros()

data = data_utils.ConnectivityData(J, neurons, 'type')
print(f'{data.N} neurons, {data.Ntype} types, {J.nnz} connected pairs, '
      f'{int(J.sum())} synapses, density {J.nnz / (data.N * (data.N - 1)):.2e}')


#%% Fit the embedding (or load it if it was already fitted)
save_path = os.path.join(EMBEDDING_DIR, hp['save_filename'])
if os.path.isfile(save_path):
    saved = pickle.load(open(save_path, 'rb'))
    model, losses = saved['model'], saved['losses']
    assert saved['data'].neurons['bodyId'].equals(data.neurons['bodyId']), \
        'saved embedding was fitted on a different selection of neurons'
    model.onehot_types = data.onehot_types   # not saved; rebuilt above
    print(f'loaded {save_path}')
else:
    torch.manual_seed(hp['seed'])
    np.random.seed(hp['seed'])
    model = models.InteractionModel(N=data.N, Ntype=data.Ntype, D=hp['embedding_dim'],
                                    onehot_types=data.onehot_types,
                                    wrapper_fn=hp['wrapper_fn'])
    model.use_transforms = False

    def reg_fn(model: models.InteractionModel):
        return torch.mean(model.embeddings**2) * hp['embedding_regularizer']

    losses, _, _ = models.train_model(
        model, target_mat=data.J, loss_type=hp['loss_type'], lr=hp['lr'],
        lr_end=hp['lr_end'], steps=hp['steps'], print_every=hp['steps'] // 20,
        reg_fn=reg_fn, pairs_per_step=hp['pairs_per_step'], pair_seed=hp['seed'])
    model.embeddings = model.embeddings.detach()

    if not hp['fit_rotations']:
        # the rotations and translations (Ntype x Ntype x D x D and Ntype x Ntype x D)
        # stay at their initial values and are not used with use_transforms = False
        model.params = [p for p in model.params
                        if p is not model.rotation_params and p is not model.translation_vecs]
        model.rotation_params, model.translation_vecs = None, None


#%% Save, without the connectivity matrix and the one-hot matrix of cell types
if not os.path.isfile(save_path):
    onehot_types = data.onehot_types
    data.J, data.onehot_types, model.onehot_types = None, None, None
    os.makedirs(EMBEDDING_DIR, exist_ok=True)
    pickle.dump({'time': datetime.datetime.now(), 'model': model, 'data': data,
                 'hyperparams': hp, 'losses': losses, 'save_filename': hp['save_filename']},
                open(save_path, 'wb'))
    data.J, data.onehot_types, model.onehot_types = J, onehot_types, onehot_types
    print(f'saved {save_path} ({os.path.getsize(save_path) / 1e9:.2f} GB)')

embeddings = model.embeddings.numpy()
