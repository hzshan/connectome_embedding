#%%
import torch, pickle, os, datetime
from connectome_embedding import utils, models, plots, data_utils

DATAPATH = 'hemibrain_data/'

central_complex_hp = {
    'types_wanted': ["EPG=", 'Delta7', 'PEG', 'EPGt', 'PEN', 'FC', 'PFL', 'PFN',\
            'FB', 'hDelta', 'vDelta', 'FR*', 'EL', 'FS', 'PFG',
            'PFR', 'IbSpsP'],  # '=' means exact match, '*' means wildcard match
    'embedding_dim': 5, 
    'wrapper_fn': 'softplus',
    'embedding_regularizer': 0.1,
    'lr': 0.02,
    'steps': 10000,
    'loss_type': 'poisson',
    'split_LR': [],  # do not split any cell type into L/R
    'fit_unitary_transforms': False,
    'save_filename': 'CX_embedding.pkl'
}

central_complex_split_hp = central_complex_hp.copy()
central_complex_split_hp['split_LR'] = central_complex_split_hp['types_wanted'].remove('Delta7')  # split all cell types except Delta7 into L/R

hp = central_complex_hp

assert 'traced-neurons.csv' in os.listdir(DATAPATH), "Please download the hemibrain data from https://neuprint.janelia.org/?dataset=hemibrain:v1.2.1 and place it in the DATAPATH folder."
assert 'traced-total-connections.csv' in os.listdir(DATAPATH), "Please download the hemibrain data from https://neuprint.janelia.org/?dataset=hemibrain:v1.2.1 and place it in the DATAPATH folder."

W_all, neuronsall = data_utils.get_W_all_neuronall(
    DATAPATH, min_num_per_type=10)


data = data_utils.prep_connectivity_data(
    W_all, neuronsall, hp['types_wanted'], split_LR=hp['split_LR'])

#%%
model = models.InteractionModel(N=data.N,
                                Ntype=data.Ntype,
                                D=hp['embedding_dim'],
                                onehot_types=data.onehot_types,
                                wrapper_fn=hp['wrapper_fn'])


#%%

if hp['fit_unitary_transforms']:
    def reg_fn(model: models.InteractionModel):
        return torch.mean(model.rotation_params**2) * 1000 +\
                torch.mean(model.translation_vecs**2) +\
                torch.mean(model.embeddings**2) * hp['embedding_regularizer']
    model.use_transforms = False
    losses, embedding_norms, _ = models.train_model(model, target_mat=data.J,
                        loss_type=hp['loss_type'], lr=hp['lr'], print_every=200,
                        reg_fn=reg_fn, steps=hp['steps'])


    model.embeddings.requires_grad = False
    model.use_transforms = True
    losses2, embedding_norms2, rotation_norms2 = models.train_model(
        model, target_mat=data.J,
        loss_type=hp['loss_type'], lr=hp['lr'], print_every=100,
        reg_fn=reg_fn, steps=500)

else:
    def reg_fn(model: models.InteractionModel):
        return torch.mean(model.embeddings**2) * hp['embedding_regularizer']


    # two-stage training. train embeddings only first, then with transforms.
    model.use_transforms = False

    losses, embedding_norms, _ = models.train_model(model, target_mat=data.J,
                        loss_type=hp['loss_type'], lr=hp['lr'], print_every=200,
                        reg_fn=reg_fn, steps=hp['steps'])


#%%
model.embeddings = model.embeddings.detach()

results_to_save = {}
results_to_save['time'] = datetime.datetime.now()
results_to_save['model'] = model
results_to_save['data'] = data
results_to_save['hyperparams'] = hp
results_to_save['save_filename'] = hp['save_filename']
if os.path.isfile(hp['save_filename']):
    raise ValueError(f"{hp['save_filename']} already exists. Please delete it first.")
print(hp['save_filename'])
pickle.dump(results_to_save, open(hp['save_filename'], 'wb'))