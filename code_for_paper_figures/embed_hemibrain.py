#%%
import torch, pickle, os, datetime, copy
import numpy as np
from connectome_embedding import utils, models, plots, data_utils

# Which embedding to learn; one of the keys of CONFIGS below:
#   'central_complex'          all CX types, no L/R split
#   'central_complex_LR_split' all CX types split into L/R, except Delta7
#   'whole_hemibrain'          every typed neuron in types with >= 10 neurons
#   'visual_projection'        LC, LPLC, PVLP, AOTU and DN types (types of any size)
CONFIG = 'whole_hemibrain'

# If True, the connection weights are shuffled within each block of the
# connectivity matrix between a pair of cell types before fitting
# (data_utils.shuffle_within_blocks), as a null: type-to-type structure is kept,
# structure among the neurons of a type is destroyed. SHUFFLE_SEED sets the shuffle.
SHUFFLE_WITHIN_BLOCKS = False
SHUFFLE_SEED = 0

DATAPATH = 'hemibrain_data/'
EMBEDDING_DIR = 'embeddings/'   # the fitted embedding is saved here

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
    'steps': 10000,
    'loss_type': 'poisson',
    'split_LR': [],  # do not split any cell type into L/R
    'fit_rotations': False,
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
central_complex_split_hp['fit_rotations'] = True
central_complex_split_hp['embedding_regularizer'] = 1.0
central_complex_split_hp['save_filename'] = 'CX_LRsplit_embedding.pkl'

# Reproduces Learned_embeddings/embedding_Allhemibrain_D5_09262025.pkl
# types_wanted=None keeps every neuron with a type name (untyped neurons are
# dropped) in types with at least min_num_per_type neurons: 7898 neurons, 168 types.
whole_hemibrain_hp = copy.deepcopy(central_complex_hp)
whole_hemibrain_hp['types_wanted'] = None
whole_hemibrain_hp['save_filename'] = 'whole_hemibrain_embedding.pkl'

# Reproduces Learned_embeddings/embedding_LC_PVLP_DN_D5_11142025.pkl
# Visual projection neurons and their partners; types of every size are kept
# (min_num_per_type=1): 2894 neurons, 321 types.
visual_projection_hp = copy.deepcopy(central_complex_hp)
visual_projection_hp['types_wanted'] = ['LC*', 'PVLP*', 'DN*', 'AOTU*', 'LPLC*']
visual_projection_hp['min_num_per_type'] = 1
visual_projection_hp['embedding_regularizer'] = 0.001
visual_projection_hp['lr'] = 0.02
visual_projection_hp['save_filename'] = 'visual_projection_embedding.pkl'

CONFIGS = {
    'central_complex': central_complex_hp,
    'central_complex_LR_split': central_complex_split_hp,
    'whole_hemibrain': whole_hemibrain_hp,
    'visual_projection': visual_projection_hp,
}
hp = copy.deepcopy(CONFIGS[CONFIG])
hp['shuffle_within_blocks'] = SHUFFLE_WITHIN_BLOCKS
if SHUFFLE_WITHIN_BLOCKS:
    hp['shuffle_seed'] = SHUFFLE_SEED
    hp['save_filename'] = hp['save_filename'].replace(
        '.pkl', f'_block_shuffled_seed{SHUFFLE_SEED}.pkl')
print(CONFIG, hp)

assert 'traced-neurons.csv' in os.listdir(DATAPATH), "Please download the hemibrain data from https://neuprint.janelia.org/?dataset=hemibrain:v1.2.1 and place it in the DATAPATH folder."
assert 'traced-total-connections.csv' in os.listdir(DATAPATH), "Please download the hemibrain data from https://neuprint.janelia.org/?dataset=hemibrain:v1.2.1 and place it in the DATAPATH folder."

W_all, neuronsall = data_utils.get_W_all_neuronall(
    DATAPATH, min_num_per_type=hp['min_num_per_type'])


data = data_utils.prep_connectivity_data(
    W_all, neuronsall, hp['types_wanted'], split_LR=hp['split_LR'])
print(f'{data.N} neurons, {data.Ntype} types')

if hp['shuffle_within_blocks']:
    shuffled_J = data_utils.shuffle_within_blocks(
        data.J.numpy(), data.onehot_types.numpy(), np.random.default_rng(hp['shuffle_seed']))
    data.J = torch.tensor(shuffled_J, dtype=data.J.dtype)
    print(f"Shuffled weights within blocks (seed {hp['shuffle_seed']})")

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
# results_to_save['losses'] = losses
if hp['fit_rotations']:
    results_to_save['losses_rotations'] = losses2
results_to_save['save_filename'] = hp['save_filename']
save_path = os.path.join(EMBEDDING_DIR, hp['save_filename'])
if os.path.isfile(save_path):
    raise ValueError(f"{save_path} already exists. Please delete it first.")
os.makedirs(EMBEDDING_DIR, exist_ok=True)
print(save_path)
pickle.dump(results_to_save, open(save_path, 'wb'))
