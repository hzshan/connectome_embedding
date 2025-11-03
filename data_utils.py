"""
This file contains utility functions for loading and processing connectome data.
Notations:
N - number of neurons
Ntype - number of neuron types

Preprocessing workflow:
1. `get_Jall_neuronall` and `get_Jall_neuronall_flywire` directly work with data
files downloaded from hemibrain and flywire, respectively. 
    At this stage, there is the option to remove types with fewer than a certain
    number of neurons.

2. `prep_connectivity_data` and `prep_connectivity_data_flywire` are used to
process the data into a ConnectivityData object, which contains the connectivity
matrix J and information about neurons.
"""

import pandas as pd
import numpy as np
import torch
import pandas
from scipy.sparse import csr_matrix


class ConnectivityData:

    @staticmethod
    def make_onehot_type_vectors(neurons:pandas.DataFrame, type_key:str):
        """
        Make one-hot vectors for neuron types. "type" is defined by the "type_key"
        column in the neurons DataFrame.

        Returns:
        types_1hot: NxNtype torch tensor, the one-hot encoding of neuron types
        Ntypes: int, the number of unique types
        typehash: dict, the mapping from type string to type index
        uniqtypes: list of str, the list of unique types
        neu
        """
        uniqtypes = pd.unique(neurons[type_key])
        Ntypes = len(uniqtypes)
        typehash = dict(zip(uniqtypes, np.arange(Ntypes)))
        typeclasses = np.array([typehash[x] for x in neurons[type_key]])

        types_1hot = np.zeros([len(neurons), Ntypes])
        types_1hot[np.arange(len(neurons)), typeclasses] = 1.
        types_1hot = torch.tensor(types_1hot).float()
        neuron_hash = {uniqtypes[i]:np.where(typeclasses == i)[0] for i in range(Ntypes)}
        return types_1hot, Ntypes, typehash, uniqtypes, neuron_hash

    def __init__(self, J, neurons:pandas.DataFrame, type_key:str):
        """
        Data object of a connectivity matrix and information about neurons.

        Args:
            J: NxN torch tensor, the connectivity matrix
            onehot_types: NxNtype torch tensor, the one-hot encoding of neuron types
            type_hash: dict, the mapping from type string to type index
            types: list of str, the list of unique types
            neuron_hash: dict, the mapping from type index to neuron indices
            N_per_type: list of int, the number of neurons of each type
            neurons: pandas DataFrame, the information about each neuron
        """
        if type(J) != torch.Tensor:
            self.J = torch.tensor(J)
        else:
            self.J = J
        self.N = J.shape[0]

        onehot_types, Ntype, typehash, uniqtypes, neuron_hash = \
            self.make_onehot_type_vectors(neurons, type_key)
        
        self.onehot_types = onehot_types
        self.Ntype = Ntype
        self.type_hash = typehash
        self.types = uniqtypes
        self.neuron_hash = neuron_hash
        self.N_per_type = onehot_types.sum(0).numpy().astype(np.int16)
        self.neurons = neurons

        assert self.onehot_types.shape == (self.N, self.Ntype)
        assert len(self.types) == self.Ntype


def get_Jall_neuronall(datapath = "", min_num_per_type=5):
    """
    Load the J matrix and neuron information from the csv files in the given
    directory.
    Only keep neurons of types with at least min_num_per_type neurons.

    Returns:
    Jall: the connectivity matrix (element i,j is the number of synapses from
    neuron j to neuron i)
    neuronsall: a pandas dataframe with information about each neuron
    """
    # load information about traced neurons and traced connections
    neuronsall = pd.read_csv(datapath + "traced-neurons.csv")
    neuronsall.sort_values(by=['instance'], ignore_index=True, inplace=True)
    conns = pd.read_csv(datapath + "traced-total-connections.csv")

    # only keep neurons with at least min_num_per_type neurons of the same type
    types = np.array(neuronsall.type).astype(str)
    unique_types, counts = np.unique(types, return_counts=True)
    inds_to_keep = []
    for _type, _count in zip(unique_types, counts):
        if _count >= min_num_per_type:
            inds_to_keep += list(np.where(np.array(neuronsall.type).astype(str) == _type)[0])

    neuronsall = neuronsall.take(inds_to_keep)

    # store a large matrix of all connections
    Nall = len(neuronsall)
    Jall = np.zeros([Nall,Nall], dtype=np.uint)

    idhash = dict(zip(neuronsall.bodyId,np.arange(Nall)))
    preinds = [idhash[x] for x in conns.bodyId_pre]
    postinds = [idhash[x] for x in conns.bodyId_post]

    Jall[postinds, preinds] = conns.weight


    return Jall, neuronsall


def get_J_neurons_maleCNS(datapath = "", min_num_per_type=5, types=[]):
    """
    Load the J matrix and neuron information from the feather files in the given
    directory.
    Only keep neurons of types with at least min_num_per_type neurons.
    Due to the large size of the full connectome, only neurons of types
    specified in `types` are kept.

    Returns:
    Jall: the connectivity matrix (element i,j is the number of synapses from
    neuron j to neuron i)
    neuronsall: a pandas dataframe with information about each neuron
    """
    # load information about traced neurons and traced connections
    neuronsall = pd.read_feather(
        datapath + "body-annotations-male-cns-v0.9-minconf-0.5.feather")
    
    # only keep traced neurons
    neuronsall = neuronsall[neuronsall.status == "Traced"]

    inds = get_inds_by_type(neuronsall, type_key='type', types_wanted=types)
    selected_neurons = neuronsall.iloc[inds].copy()
    # selected_neurons = neuronsall[neuronsall.type.isin(types)]


    # only keep neurons with at least min_num_per_type neurons of the same type
    types = np.array(selected_neurons.type).astype(str)
    unique_types, counts = np.unique(types, return_counts=True)
    inds_to_keep = []
    for _type, _count in zip(unique_types, counts):
        if _count >= min_num_per_type:
            inds_to_keep += list(np.where(
                np.array(selected_neurons.type).astype(str) == _type)[0])

    selected_neurons = selected_neurons.take(inds_to_keep)
    selected_neurons.sort_values(by=['instance'], ignore_index=True, inplace=True)

    conns = pd.read_feather(datapath + ""
    "connectome-weights-male-cns-v0.9-minconf-0.5.feather")

    # for the connections, remove ones with pre or post neurons not in neuronsall
    valid_bodyIds = set(selected_neurons.bodyId)
    conns = conns[conns.body_pre.isin(valid_bodyIds) &
                  conns.body_post.isin(valid_bodyIds)]
    
    print(f'Processed {len(selected_neurons)} neurons and {len(conns)} synapses.')

    idhash = dict(zip(selected_neurons.bodyId,np.arange(len(selected_neurons))))


    selected_J = np.zeros((len(selected_neurons), len(selected_neurons)), dtype=np.float32)
    selected_J[list(conns.body_post.apply(lambda x: idhash[x])),
         list(conns.body_pre.apply(lambda x: idhash[x]))] = conns.weight

    return selected_J, selected_neurons


def prep_connectivity_data(full_J_mat,
                           all_neurons: pd.DataFrame,
                           types_wanted=[],
                           split_LR=None):
    """
    Prepares the connectivity data for training. Selects only neurons of types
    containing one of the strings in types_wanted. Optionally, sorts neurons
    into left and right based on the instance name (in hemibrain data, some
    neurons have type names such as 'EPG_L' and 'EPG_R').

    Important: the returned J matrix is transposed, so that J[i,j] is the number
    of synapses from neuron i to neuron j.

    Args:
    full_J_mat: NxN numpy array, the full connectivity matrix. Element i,j is
    the number of synapses from neuron j to neuron i.
    all_neurons: pandas DataFrame, the information about each neuron
    types_wanted: list of str, the list of strings to search for in neuron types
    split_LR: if the neuron type contains one of these strings, split neurons
    of this type into two types, one for left and one for right. Setting it to
    None will split all types.

    Returns:
    """
    if types_wanted is not None:
        types_wanted = np.unique(types_wanted)
    allcx = get_inds_by_type(all_neurons, 'type', types_wanted)

    J = full_J_mat[allcx, :][:, allcx].astype(np.float32)
    N = J.shape[0]
    neurons = all_neurons.iloc[allcx, :]
    neurons.reset_index(inplace=True, drop=True)

    if split_LR is None:
        print('Will attempt to split all types into left and right.')

    # sort neurons into left and right based on some hemibrain specific rules
    for i in range(N):
        if split_LR is None or np.sum([t in neurons.type[i] for t in split_LR]) > 0:

            neuron_name = neurons.instance[i]

            LR_suffix = hemibrain_LR(neuron_name)
            neurons.loc[i, 'type'] += LR_suffix

    new_type_sort = np.argsort(neurons.type)
    neurons = neurons.iloc[new_type_sort, :]
    J = J[new_type_sort, :][:, new_type_sort]

    neurons.reset_index(inplace=True)

    J_data = ConnectivityData(torch.tensor(J.T).float(), neurons, 'type')

    return J_data


def get_inds_by_type(neuronall:pd.DataFrame, type_key:str, types_wanted:list):
    """
    Get indices of neurons whose type contains one of the strings in types_wanted.

    the "type" used for each neuron is specified by `type_key`, which should be 
    one of the columns of neuronall.
    """

    if types_wanted is None:
        print('No types specified. Grabbing all types.')
        types_wanted = list(neuronall[type_key].unique())
        types_wanted = [str(t) + '=' for t in types_wanted if type(t) == str]
    
    contains = [t for t in types_wanted if t[-1] not in ['=', '*']]
    exact = [t[:-1] for t in types_wanted if t[-1] == '=']
    startswith = [t[:-1] for t in types_wanted if t[-1] == '*']

    print('Getting indices for types containing:', contains)
    print('Getting indices for types starting with:', startswith)
    print('Getting indices for types exactly matching:', exact)


    def _sortsubtype(t, list_of_types):
        """
        Get indices of neurons whose type contains the string t.
        if the string ends with '=', then only neurons of that exact type are
        included.
        if the string ends with '*", then only neurons of a type starting with that
        string are included.
        """
        
        if t[-1] == '=':
            t = t[:-1]
            inds = np.nonzero([t == x for x in list_of_types])[0]
        elif t[-1] == '*':
            t = t[:-1]
            inds = np.nonzero([x.startswith(t) for x in list_of_types])[0]
        else:
            inds = np.nonzero([t in x for x in list_of_types])[0]
        sortinds = np.argsort(list_of_types[inds])
        inds = inds[sortinds]
        return inds

    list_of_types = np.array(neuronall[type_key]).astype(str)

    output = []
    for t in types_wanted:
        if type(t) == float:
            continue
        output.append(_sortsubtype(t, list_of_types))

    return np.unique(np.concatenate(output))


def get_Jall_neuronall_flywire(datapath = ""):
    neuronsall = pd.read_csv(datapath + "neurons.csv")
    classif = pd.read_csv(datapath + "classification.csv")
    visual_types = pd.read_csv(datapath + "visual_neuron_types.csv")
    neuronsall = neuronsall.merge(classif, on="root_id", how="left")
    neuronsall = neuronsall.merge(visual_types, on="root_id", how="left")
    neuronsall.rename(columns={'type': "visual_type", 'family':'visual_family',
                            'subsystem':'visual_subsystem'}, inplace=True)
    conns = pd.read_csv(datapath + "connections.csv")

    Nall = len(neuronsall)
    idhash = dict(zip(neuronsall.root_id,np.arange(Nall)))
    neuronsall['J_idx'] = neuronsall['J_idx_post'] = neuronsall['J_idx_pre'] = neuronsall.root_id.apply(lambda x: idhash[x])
    preinds = [idhash[x] for x in conns.pre_root_id]
    postinds = [idhash[x] for x in conns.post_root_id]
    # Jall = np.zeros([Nall, Nall], dtype=np.uint)
    # Jall[postinds, preinds] = conns.syn_count
    Jall = csr_matrix((conns.syn_count, (postinds, preinds)), shape=(Nall, Nall), dtype="int16")

    return neuronsall, Jall


def prep_connectivity_data_flywire(
        full_J_mat, all_neurons, types_wanted=[], split_LR=None, type_key='visual_type'):
    """
    Important: the returned J matrix is transposed, so that J[i,j] is the number
    of synapses from neuron i to neuron j.
    """

    allcx = get_inds_by_type(all_neurons, type_key, types_wanted)

    J = full_J_mat[allcx, :][:, allcx].astype(np.float32)
    N = J.shape[0]
    neurons = all_neurons.iloc[allcx, :]
    neurons.reset_index(inplace=True)

    # sort neurons into left and right (this is supplied directly in flywire)
    for i in range(N):
        if split_LR is None or np.sum([t in neurons.type[i] for t in split_LR]) > 0:

            neuron_side = neurons.side[i]

            if neuron_side == 'left':
                neurons.loc[i, type_key] += '_L'
                continue

            if neuron_side == 'right':
                neurons.loc[i, type_key] += '_R'
                continue

    new_type_sort = np.argsort(neurons[type_key])
    neurons = neurons.iloc[new_type_sort, :]
    J = J[new_type_sort, :][:, new_type_sort]

    J_data = ConnectivityData(torch.tensor(J.T).float(), neurons, type_key)


    return J_data


def hemibrain_LR(neuron_name):
    """
    Try to figure out the L/R of each neuron from its neuron_name.
    This is necessary since L/R is not directly labeled in hemibrain.
    Produces a suffix to be attached to the type of the neuron.
    """

    # if the instance name ends with _L/R; this covers most neurons
    if neuron_name[-2:] == '_L':
        return '_L'

    if neuron_name[-2:] == '_R':
        return '_R'

    # this covers a few FR2(FQ4), PFNp_d, SCL-AVLP neurons
    if '_L_' in neuron_name:
        return '_L'

    if '_R_' in neuron_name:
        return '_R'

    # this handles a few PFGs(PB10) neurons
    if 'irreg' in neuron_name:
        return '_irreg'

    # the neurons that are left are usually labeled as 'L4' or 'R5' etc.
    if '_L' in neuron_name[-6:]:
        return '_L'

    if '_R' in neuron_name[-6:]:
        return '_R'
    
    return ''