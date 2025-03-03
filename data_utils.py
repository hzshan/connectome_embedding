import pandas as pd
import numpy as np
import torch
import tqdm, pandas
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import scipy, analysis


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


def get_Jall_neuronall(datapath = "", min_num_per_type=5):
    """
    Load the J matrix and neuron information from the csv files in the given directory.
    Only keep neurons of types with at least min_num_per_type neurons.

    Returns:
    Jall: the connectivity matrix (element i,j is the number of synapses from neuron j to neuron i)
    neuronsall: a pandas dataframe with information about each neuron
    """
    # load information about traced neurons and traced connections
    neuronsall = pd.read_csv(datapath + "traced-neurons.csv")
    neuronsall.sort_values(by=['instance'], ignore_index=True, inplace=True)
    conns = pd.read_csv(datapath + "traced-total-connections.csv")

    # store a large matrix of all connections
    Nall = len(neuronsall)
    Jall = np.zeros([Nall,Nall], dtype=np.uint)

    idhash = dict(zip(neuronsall.bodyId,np.arange(Nall)))
    preinds = [idhash[x] for x in conns.bodyId_pre]
    postinds = [idhash[x] for x in conns.bodyId_post]

    Jall[postinds, preinds] = conns.weight

    # only keep neurons with at least min_num_per_type neurons of the same type
    types = np.array(neuronsall.type).astype(str)
    unique_types, counts = np.unique(types, return_counts=True)
    inds_to_keep = []
    for _type, _count in zip(unique_types, counts):
        if _count > min_num_per_type:
            inds_to_keep += list(np.where(np.array(neuronsall.type).astype(str) == _type)[0])

    return Jall[inds_to_keep][:, inds_to_keep], neuronsall.take(inds_to_keep)


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

def get_inds_by_type(neuronall:pd.DataFrame, type_key:str, types_wanted:list):
    """
    Get indices of neurons whose type contains one of the strings in types_wanted.

    the "type" used for each neuron is specified by `type_key`, which should be 
    one of the columns of neuronall.
    """

    def _sortsubtype(t, list_of_types):
        """
        Get indices of neurons whose type contains the string t.
        if the string ends with '=', then only neurons of that exact type are
        included.
        """

        if t[-1] == '=':
            t = t[:-1]
            inds = np.nonzero([t == x for x in list_of_types])[0]
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

    return np.concatenate(output)


def prep_connectivity_data(full_J_mat, all_neurons, types_wanted=[], split_LR=None):
    """
    Produce J and one-hot type vectors in torch tensor format. Include only
    neurons of types in types_wanted.
    """

    allcx = get_inds_by_type(all_neurons, 'type', types_wanted)

    J = full_J_mat[allcx, :][:, allcx].astype(np.float32)
    N = J.shape[0]
    neurons = all_neurons.iloc[allcx, :]
    neurons.reset_index(inplace=True)

    # sort neurons into left and right based on some hemibrain specific rules
    for i in range(N):
        if split_LR is None or np.sum([t in neurons.type[i] for t in split_LR]) > 0:

            neuron_name = neurons.instance[i]

            # if the instance name ends with _L/R; this covers most neurons
            if neuron_name[-2:] == '_L':
                neurons.loc[i, 'type'] += '_L'
                continue

            if neuron_name[-2:] == '_R':
                neurons.loc[i, 'type'] += '_R'
                continue

            # this covers a few FR2(FQ4), PFNp_d, SCL-AVLP neurons
            if '_L_' in neuron_name:
                neurons.loc[i, 'type'] += '_L'
                continue
            if '_R_' in neuron_name:
                neurons.loc[i, 'type'] += '_R'
                continue

            # this handles a few PFGs(PB10) neurons
            if 'irreg' in neuron_name:
                neurons.loc[i, 'type'] += '_irreg'
                continue

            # the neurons that are left are usually labeled as 'L4' or 'R5' etc.
            if '_L' in neuron_name[-6:]:
                neurons.loc[i, 'type'] += '_L'
                continue

            if '_R' in neuron_name[-6:]:
                neurons.loc[i, 'type'] += '_R'

    new_type_sort = np.argsort(neurons.type)
    neurons = neurons.iloc[new_type_sort, :]
    J = J[new_type_sort, :][:, new_type_sort]

    J_data = ConnectivityData(torch.tensor(J.T).float(), neurons, 'type')


    return J_data


def prep_connectivity_data_flywire(full_J_mat, all_neurons, types_wanted=[], split_LR=None):
    """
    Produce J and one-hot type vectors in torch tensor format. Include only
    neurons of types in types_wanted.
    """

    allcx = get_inds_by_type(all_neurons, 'type', types_wanted)

    J = full_J_mat[allcx, :][:, allcx].astype(np.float32)
    N = J.shape[0]
    neurons = all_neurons.iloc[allcx, :]
    neurons.reset_index(inplace=True)

    # sort neurons into left and right based on some hemibrain specific rules
    for i in range(N):
        if split_LR is None or np.sum([t in neurons.type[i] for t in split_LR]) > 0:

            neuron_side = neurons.side[i]

            # if the instance name ends with _L/R; this covers most neurons
            if neuron_side == 'left':
                neurons.loc[i, 'type'] += '_L'
                continue

            if neuron_side == 'right':
                neurons.loc[i, 'type'] += '_R'
                continue

    new_type_sort = np.argsort(neurons.type)
    neurons = neurons.iloc[new_type_sort, :]
    J = J[new_type_sort, :][:, new_type_sort]

    J_data = ConnectivityData(torch.tensor(J.T).float(), neurons, 'type')


    return J_data