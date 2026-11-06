"""
Settings of the embeddings learned by code_for_paper_figures/embed_hemibrain.py
(hemibrain) and code_for_paper_figures/embed_flywire_cave.py (FlyWire).

Each config is a dict of hyperparameters; get_config(name) returns a copy of one.
Configs:
    'central_complex'          all CX types, no L/R split
    'central_complex_LR_split' all CX types split into L/R, except Delta7
    'whole_hemibrain'          every typed neuron in types with >= 10 neurons
    'visual_projection'        LC, LPLC, PVLP, AOTU and DN types (types of any size)
    'central_complex_null'     null models: same as 'central_complex' /
    'whole_hemibrain_null'     'whole_hemibrain', fitted to connection weights shuffled
                               within each block of the connectivity matrix between a
                               pair of cell types (data_utils.shuffle_within_blocks):
                               type-to-type structure is kept, structure among the
                               neurons of a type is destroyed.
FlyWire (embed_flywire_cave.py):
    'flywire_central_complex'          same CX types and settings as 'central_complex'
    'flywire_central_complex_LR_split' same split and settings as
                                       'central_complex_LR_split'
"""
import copy

# Reproduces Learned_embeddings/embedding_AllCX_D5_09232025.pkl
central_complex = {
    'types_wanted': ["EPG=", 'Delta7', 'PEG', 'EPGt', 'PEN', 'FC', 'PFL', 'PFN',
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
    'shuffle_within_blocks': False,
    'save_filename': 'CX_embedding.pkl',
}

# Reproduces Learned_embeddings/embedding_AllCX_D5_LRsplit_woDelta7_031225.pkl
# Split all cell types into L/R except Delta7 (types without an L/R instance
# name, such as hDelta/vDelta, stay unsplit).
central_complex_LR_split = copy.deepcopy(central_complex)
central_complex_LR_split['split_LR'] = [
    t.strip('=*') for t in central_complex['types_wanted'] if t != 'Delta7']
central_complex_LR_split['fit_rotations'] = True
central_complex_LR_split['embedding_regularizer'] = 1.0
central_complex_LR_split['save_filename'] = 'CX_LRsplit_embedding.pkl'

# Reproduces Learned_embeddings/embedding_Allhemibrain_D5_09262025.pkl
# types_wanted=None keeps every neuron with a type name (untyped neurons are
# dropped) in types with at least min_num_per_type neurons: 7898 neurons, 168 types.
whole_hemibrain = copy.deepcopy(central_complex)
whole_hemibrain['types_wanted'] = None
whole_hemibrain['save_filename'] = 'whole_hemibrain_embedding.pkl'

# Reproduces Learned_embeddings/embedding_LC_PVLP_DN_D5_11142025.pkl
# Visual projection neurons and their partners; types of every size are kept
# (min_num_per_type=1): 2894 neurons, 321 types.
visual_projection = copy.deepcopy(central_complex)
visual_projection['types_wanted'] = ['LC*', 'PVLP*', 'DN*', 'AOTU*', 'LPLC*']
visual_projection['min_num_per_type'] = 1
visual_projection['embedding_regularizer'] = 0.001
visual_projection['lr'] = 0.02
visual_projection['save_filename'] = 'visual_projection_embedding.pkl'

# FlyWire central complex: same types and settings as 'central_complex', with cell
# types from FlyWire's hemibrain_type annotation.
flywire_central_complex = copy.deepcopy(central_complex)
del flywire_central_complex['shuffle_within_blocks']
flywire_central_complex.update({
    'neuron_class': 'CX',             # FlyWire cell class the neurons are taken from
    'type_key': 'hemibrain_type',     # annotation column used as the cell type
    'min_syn_per_connection': 1,      # drop neuron pairs connected by fewer synapses
    'save_filename': 'flywire_CX_cave_embedding.pkl',
})

# FlyWire central complex split into L/R (FlyWire's 'side' column), matching what
# 'central_complex_LR_split' gives in the hemibrain: every type except Delta7, hDelta
# and vDelta (the hemibrain splits by instance name, and hDelta/vDelta names carry no
# side, so they stay unsplit there).
flywire_central_complex_LR_split = copy.deepcopy(flywire_central_complex)
flywire_central_complex_LR_split['split_LR'] = [
    t.strip('=*') for t in central_complex['types_wanted']
    if t not in ('Delta7', 'hDelta', 'vDelta')]
flywire_central_complex_LR_split['fit_rotations'] = True
flywire_central_complex_LR_split['embedding_regularizer'] = 1.0
flywire_central_complex_LR_split['save_filename'] = 'flywire_CX_LRsplit_embedding.pkl'

CONFIGS = {
    'central_complex': central_complex,
    'central_complex_LR_split': central_complex_LR_split,
    'whole_hemibrain': whole_hemibrain,
    'visual_projection': visual_projection,
    'flywire_central_complex': flywire_central_complex,
    'flywire_central_complex_LR_split': flywire_central_complex_LR_split,
}

# null model name -> the config it shuffles
NULL_CONFIGS = {
    'central_complex_null': 'central_complex',
    'whole_hemibrain_null': 'whole_hemibrain',
}


def null_config(hp, shuffle_seed):
    """The same embedding fitted to weights shuffled within type-pair blocks.
    shuffle_seed sets both the shuffle and the model initialization."""
    null_hp = copy.deepcopy(hp)
    null_hp['shuffle_within_blocks'] = True
    null_hp['shuffle_seed'] = shuffle_seed
    null_hp['seed'] = shuffle_seed
    null_hp['save_filename'] = hp['save_filename'].replace(
        '.pkl', f'_block_shuffled_seed{shuffle_seed}.pkl')
    return null_hp


def get_config(name, shuffle_seed=0):
    """
    Hyperparameters of config `name` (a copy). shuffle_seed is used by the null
    model configs only.
    """
    if name in NULL_CONFIGS:
        return null_config(CONFIGS[NULL_CONFIGS[name]], shuffle_seed)
    if name in CONFIGS:
        return copy.deepcopy(CONFIGS[name])
    raise ValueError(f'unknown config {name!r}; one of {list(CONFIGS) + list(NULL_CONFIGS)}')
