#%%
import torch, pickle, os, datetime, copy
import numpy as np
from connectome_embedding import utils, models, plots, data_utils

CX_LRSPLIT = True  # True: split all CX types into L/R except Delta7; False: no L/R split

DATAPATH = 'hemibrain_data/'

# Reproduces Learned_embeddings/embedding_AllCX_D5_09232025.pkl
central_complex_hp = {
    'types_wanted': ["EPG=", 'Delta7', 'PEG', 'EPGt', 'PEN', 'FC', 'PFL', 'PFN',\
            'FB', 'hDelta', 'vDelta', 'FR*', 'EL', 'FS', 'PFG',
            'PFR', 'IbSpsP'],  # '=' means exact match, '*' means wildcard match
    'min_num_per_type': 10,
    'embedding_dim': 5,
    'wrapper_fn': 'softplus',
    'embedding_regularizer': 0.01,
    'lr': 0.05,
    'steps': 20000,
    'loss_type': 'poisson',
    'split_LR': [],  # do not split any cell type into L/R
    'fit_rotations': True,
    'rotation_steps': 500,
    'rotation_regularizer': 1000.,
    'translation_regularizer': 1.,
    'seed': 0,
    'save_filename': 'CX_embedding.pkl'
}

# Reproduces Learned_embeddings/embedding_AllCX_D5_LRsplit_woDelta7_031225.pkl
# Split all cell types into L/R except Delta7 (types without an L/R instance
# name, such as hDelta/vDelta, stay unsplit).
central_complex_split_hp = copy.deepcopy(central_complex_hp)
central_complex_split_hp['split_LR'] = [
    t.strip('=*') for t in central_complex_hp['types_wanted'] if t != 'Delta7']
central_complex_split_hp['embedding_regularizer'] = 1.0
central_complex_split_hp['save_filename'] = 'CX_LRsplit_embedding.pkl'

hp = central_complex_split_hp if CX_LRSPLIT else central_complex_hp
print(hp)

assert 'traced-neurons.csv' in os.listdir(DATAPATH), "Please download the hemibrain data from https://neuprint.janelia.org/?dataset=hemibrain:v1.2.1 and place it in the DATAPATH folder."
assert 'traced-total-connections.csv' in os.listdir(DATAPATH), "Please download the hemibrain data from https://neuprint.janelia.org/?dataset=hemibrain:v1.2.1 and place it in the DATAPATH folder."

W_all, neuronsall = data_utils.get_W_all_neuronall(
    DATAPATH, min_num_per_type=hp['min_num_per_type'])


data = data_utils.prep_connectivity_data(
    W_all, neuronsall, hp['types_wanted'], split_LR=hp['split_LR'])
print(f'{data.N} neurons, {data.Ntype} types')

#%%
torch.manual_seed(hp['seed'])
np.random.seed(hp['seed'])

model = models.InteractionModel(N=data.N,
                                Ntype=data.Ntype,
                                D=hp['embedding_dim'],
                                onehot_types=data.onehot_types,
                                wrapper_fn=hp['wrapper_fn'])


#%%
# two-stage training. train embeddings only first, then with transforms.
def reg_fn(model: models.InteractionModel):
    return torch.mean(model.embeddings**2) * hp['embedding_regularizer']

model.use_transforms = False
losses, embedding_norms, _ = models.train_model(model, target_mat=data.J,
                    loss_type=hp['loss_type'], lr=hp['lr'], print_every=200,
                    reg_fn=reg_fn, steps=hp['steps'])

if hp['fit_rotations']:
    def reg_fn_rotations(model: models.InteractionModel):
        return torch.mean(model.rotation_params**2) * hp['rotation_regularizer'] +\
                torch.mean(model.translation_vecs**2) * hp['translation_regularizer'] +\
                torch.mean(model.embeddings**2) * hp['embedding_regularizer']

    model.embeddings.requires_grad = False
    model.use_transforms = True
    losses2, embedding_norms2, rotation_norms2 = models.train_model(
        model, target_mat=data.J,
        loss_type=hp['loss_type'], lr=hp['lr'], print_every=100,
        reg_fn=reg_fn_rotations, steps=hp['rotation_steps'])


#%%
model.embeddings = model.embeddings.detach()

results_to_save = {}
results_to_save['time'] = datetime.datetime.now()
results_to_save['model'] = model
results_to_save['data'] = data
results_to_save['hyperparams'] = hp
results_to_save['losses'] = losses
if hp['fit_rotations']:
    results_to_save['losses_rotations'] = losses2
results_to_save['save_filename'] = hp['save_filename']
if os.path.isfile(hp['save_filename']):
    raise ValueError(f"{hp['save_filename']} already exists. Please delete it first.")
print(hp['save_filename'])
pickle.dump(results_to_save, open(hp['save_filename'], 'wb'))
