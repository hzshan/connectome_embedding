#%%
import torch, pickle, os, datetime
import numpy as np
from connectome_embedding import models, data_utils, hemibrain_configs

# Which embedding to learn; the configs are defined in
# connectome_embedding/hemibrain_configs.py:
#   'central_complex'          all CX types, no L/R split
#   'central_complex_LR_split' all CX types split into L/R, except Delta7
#   'whole_hemibrain'          every typed neuron in types with >= 10 neurons
#   'visual_projection'        LC, LPLC, PVLP, AOTU and DN types (types of any size)
#   'central_complex_null'     null models of 'central_complex' / 'whole_hemibrain':
#   'whole_hemibrain_null'     weights shuffled within each type-pair block
CONFIG = 'central_complex_null'

# Null models only: seed of the shuffle, also used to initialize the model.
# Run once per seed to get several null embeddings.
SHUFFLE_SEED = 1

DATAPATH = 'hemibrain_data/'
EMBEDDING_DIR = 'embeddings/'   # the fitted embedding is saved here

hp = hemibrain_configs.get_config(CONFIG, shuffle_seed=SHUFFLE_SEED)
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
results_to_save['losses'] = losses
if hp['fit_rotations']:
    results_to_save['losses_rotations'] = losses2
results_to_save['save_filename'] = hp['save_filename']
save_path = os.path.join(EMBEDDING_DIR, hp['save_filename'])
if os.path.isfile(save_path):
    raise ValueError(f"{save_path} already exists. Please delete it first.")
os.makedirs(EMBEDDING_DIR, exist_ok=True)
print(save_path)
pickle.dump(results_to_save, open(save_path, 'wb'))
