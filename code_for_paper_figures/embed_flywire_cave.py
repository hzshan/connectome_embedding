#%%
"""
Learn an embedding of the central complex (CX) in the FlyWire connectome, with the
data pulled directly from FlyWire's CAVE servers using CAVEclient.

Neurons: from FlyWire's public annotation file (flyconnectome/flywire_annotations),
mapped to v783 root ids with CAVE; cell class 'CX' and a hemibrain_type with at least
min_num_per_type neurons, restricted to the same CX types as the 'central_complex'
config of embed_hemibrain.py. Cell types are FlyWire's hemibrain_type.
Connectivity: synapses between those neurons from the filtered synapse view
(valid_synapses_nt_np_v6), counted per (pre, post) neuron pair.

Training and the saved format are the same as embed_hemibrain.py.

Setup (once): pip install caveclient, then get a token with
    from caveclient import CAVEclient
    CAVEclient().auth.get_new_token()
and follow the printed instructions. Run from code_for_paper_figures/.

Queried data are cached in CACHEPATH, so the servers are only hit on the first run.
"""
import torch, pickle, os, datetime, urllib.request
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from caveclient import CAVEclient
from connectome_embedding import models, data_utils
import embedding_configs

# Which embedding to learn; the configs are defined in
# embedding_configs.py:
#   'flywire_central_complex'          CX types, no L/R split (as 'central_complex')
#   'flywire_central_complex_LR_split' CX types split into L/R, except Delta7, hDelta
#                                      and vDelta (as 'central_complex_LR_split')
CONFIG = 'flywire_central_complex'

DATASTACK = 'flywire_fafb_public'
MATERIALIZATION = 783       # same FlyWire snapshot as the Codex files
ANNOTATION_URL = ('https://raw.githubusercontent.com/flyconnectome/flywire_annotations/'
                  'main/supplemental_files/Supplemental_file1_neuron_annotations.tsv')
SYNAPSE_VIEW = 'valid_synapses_nt_np_v6'
CACHEPATH = 'flywire_cave_cache/'
EMBEDDING_DIR = 'embeddings/'   # the fitted embedding is saved here
PRE_CHUNK = 100             # presynaptic neurons per synapse query
MAX_ROWS = 500000           # server-side row limit per query

hp = embedding_configs.get_config(CONFIG)
print(CONFIG, hp)
os.makedirs(CACHEPATH, exist_ok=True)

client = CAVEclient(DATASTACK)
assert MATERIALIZATION in client.materialize.get_versions(), \
    f'Version {MATERIALIZATION} is not available: {client.materialize.get_versions()}'

#%% Neuron annotations
# hemibrain_type is not in CAVE's hierarchical_neuron_annotations, so use FlyWire's
# public annotation file. Its root ids are newer than the v783 snapshot, so the
# neurons are identified by supervoxel and mapped to the v783 root ids with CAVE.
annot_file = CACHEPATH + 'neuron_annotations.tsv'
if not os.path.isfile(annot_file):
    print('downloading', ANNOTATION_URL)
    urllib.request.urlretrieve(ANNOTATION_URL, annot_file)
annot = pd.read_csv(annot_file, sep='\t', usecols=[
    'supervoxel_id', 'cell_class', hp['type_key'], 'side'])

cx_neurons = annot[annot['cell_class'] == hp['neuron_class']]
cx_neurons = cx_neurons.dropna(subset=[hp['type_key']])
type_counts = cx_neurons[hp['type_key']].value_counts()
cx_neurons = cx_neurons[cx_neurons[hp['type_key']].isin(
    type_counts[type_counts >= hp['min_num_per_type']].index)].reset_index(drop=True)

root_file = CACHEPATH + f"roots_{hp['neuron_class']}_{MATERIALIZATION}.npy"
if os.path.isfile(root_file):
    roots = np.load(root_file)
else:
    timestamp = client.materialize.get_timestamp(MATERIALIZATION)
    roots = np.asarray(client.chunkedgraph.get_roots(
        cx_neurons['supervoxel_id'].values, timestamp=timestamp), dtype=np.int64)
    np.save(root_file, roots)
cx_neurons['root_id'] = roots

# neurons that were merged into one root, or have no root at v783, are dropped
cx_neurons = cx_neurons[cx_neurons['root_id'] != 0].drop_duplicates('root_id')
cx_neurons = cx_neurons.reset_index(drop=True)
print(f"{len(cx_neurons)} {hp['neuron_class']} neurons in "
      f"{cx_neurons[hp['type_key']].nunique()} types")

#%% Synapses among the CX neurons, aggregated to neuron pairs
root_ids = cx_neurons['root_id'].values
conn_file = CACHEPATH + f"synapses_{hp['neuron_class']}_{MATERIALIZATION}.pkl"
if os.path.isfile(conn_file):
    conns = pd.read_pickle(conn_file)
else:
    chunks = []
    for start in range(0, len(root_ids), PRE_CHUNK):
        df = client.materialize.query_view(
            SYNAPSE_VIEW,
            filter_in_dict={'pre_pt_root_id': root_ids[start:start + PRE_CHUNK],
                            'post_pt_root_id': root_ids},
            select_columns=['pre_pt_root_id', 'post_pt_root_id'],
            materialization_version=MATERIALIZATION)
        assert len(df) < MAX_ROWS, \
            'Query hit the row limit and was truncated; lower PRE_CHUNK.'
        chunks.append(df.groupby(['pre_pt_root_id', 'post_pt_root_id']).size())
        print(f'queried {min(start + PRE_CHUNK, len(root_ids))}/{len(root_ids)} presynaptic neurons')
    conns = pd.concat(chunks).rename('syn_count').reset_index()
    conns.to_pickle(conn_file)

conns = conns[conns['syn_count'] >= hp['min_syn_per_connection']]
idx = pd.Series(np.arange(len(root_ids)), index=root_ids)
# W[i, j] is the number of synapses from neuron j to neuron i
cx_W = csr_matrix((conns['syn_count'].values,
                   (idx[conns['post_pt_root_id']].values, idx[conns['pre_pt_root_id']].values)),
                  shape=(len(root_ids), len(root_ids))).toarray()
print(f'{len(conns)} connections, {int(cx_W.sum())} synapses')

data = data_utils.prep_connectivity_data_flywire(
    cx_W, cx_neurons, hp['types_wanted'], split_LR=hp['split_LR'], type_key=hp['type_key'])
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
save_path = os.path.join(EMBEDDING_DIR, hp['save_filename'])
if os.path.isfile(save_path):
    raise ValueError(f"{save_path} already exists. Please delete it first.")
os.makedirs(EMBEDDING_DIR, exist_ok=True)
print(save_path)
pickle.dump(results_to_save, open(save_path, 'wb'))
