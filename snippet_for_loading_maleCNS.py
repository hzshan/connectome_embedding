
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
    
    print(f'Processed {len(selected_neurons)} neurons and {len(conns)} connections.')

    idhash = dict(zip(selected_neurons.bodyId,np.arange(len(selected_neurons))))


    selected_J = np.zeros((len(selected_neurons), len(selected_neurons)), dtype=np.float32)
    selected_J[list(conns.body_post.apply(lambda x: idhash[x])),
         list(conns.body_pre.apply(lambda x: idhash[x]))] = conns.weight

    return selected_J, selected_neurons

# Example usage
# convention is as follows:
# if a string ends with '=', then only neurons of that exact type are
# included.
# if a string ends with '*', then only neurons of a type starting with that
# string are included.
# otherwise, all neurons whose type contains that string are included.
DRA_pathway_neurons = ['R7d', 'R8d', 'Dm-DRA1', 'Dm-DRA2', 'MeTu2a', 'TuBu01', 'ER4m']
Jall, neuronsall = get_J_neurons_maleCNS(datapath,
                                              min_num_per_type=0,
                                              types=DRA_pathway_neurons)