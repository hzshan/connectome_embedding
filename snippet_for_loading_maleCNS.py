
import pandas as pd
import numpy as np

# The files are
# Info about neurons: https://storage.googleapis.com/flyem-male-cns/v0.9/connectome-data/flat-connectome/body-annotations-male-cns-v0.9-minconf-0.5.feather
# Info about connections: https://storage.googleapis.com/flyem-male-cns/v0.9/connectome-data/flat-connectome/connectome-weights-male-cns-v0.9-minconf-0.5.feather
# Produces a N by N matrix, Jall, and a dataframe neuronsall with info about each neuron.


datapath = 'data_maleCNS/'
neuronsall = pd.read_feather(
        datapath + "body-annotations-male-cns-v0.9-minconf-0.5.feather")

neuronsall = neuronsall[neuronsall['type'].notna()]

def get_J_neurons_maleCNS(datapath = "", min_num_per_type=5, types=[]):
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
    neuronsall = pd.read_feather(
        datapath + "body-annotations-male-cns-v0.9-minconf-0.5.feather")
    
    # only keep traced neurons
    neuronsall = neuronsall[neuronsall.status == "Traced"]

    # only keep neurons with desired types
    neuronsall = neuronsall[neuronsall.type.isin(types)]

    neuronsall.sort_values(by=['instance'], ignore_index=True, inplace=True)

    # only keep neurons with at least min_num_per_type neurons of the same type
    types = np.array(neuronsall.type).astype(str)
    unique_types, counts = np.unique(types, return_counts=True)
    inds_to_keep = []
    for _type, _count in zip(unique_types, counts):
        if _count >= min_num_per_type:
            inds_to_keep += list(np.where(
                np.array(neuronsall.type).astype(str) == _type)[0])

    conns = pd.read_feather(datapath + ""
    "connectome-weights-male-cns-v0.9-minconf-0.5.feather")

    # for the connections, remove ones with pre or post neurons not in neuronsall
    valid_bodyIds = set(neuronsall.bodyId)
    conns = conns[conns.body_pre.isin(valid_bodyIds) &
                  conns.body_post.isin(valid_bodyIds)]
    
    print(f'Processed {len(neuronsall)} neurons and {len(conns)} connections.')

    idhash = dict(zip(neuronsall.bodyId,np.arange(len(neuronsall))))


    Jall = np.zeros((len(neuronsall), len(neuronsall)), dtype=np.float32)
    Jall[list(conns.body_post.apply(lambda x: idhash[x])),
         list(conns.body_pre.apply(lambda x: idhash[x]))] = conns.weight

    return Jall[inds_to_keep][:, inds_to_keep], neuronsall.take(inds_to_keep)

# Example usage. Type matching uses the following convention:
# If the string ends with "*", it matches types that start with the string
# If the string ends with "=", it matches types that exactly match the string
# Otherwise, it matches types that contain the string
DRA_pathway_neurons = ['R7d', 'R8d', 'Dm-DRA1', 'Dm-DRA2', 'MeTu2a', 'TuBu01', 'ER4m']
Jall, neuronsall = get_J_neurons_maleCNS(datapath,
                                              min_num_per_type=0,
                                              types=DRA_pathway_neurons)