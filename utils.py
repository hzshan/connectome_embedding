import pandas as pd
import numpy as np
import torch
import tqdm, pandas
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

def train_model(model, target_mat, types_1hot,
                loss_type='poisson', lr=0.01, steps=40000,
                print_every=2000, reg_fn=None):

    # only compute loss on off-diagonal elements of J
    _N = target_mat.shape[0]
    validinds = np.where((np.ones([_N, _N]) - np.diag(np.ones(_N))).flatten())[0]
    yvalid = target_mat.flatten()[validinds]

    optim = torch.optim.Adam(params=model.params, lr=lr)

    losses = np.zeros(steps)
    embedding_norms = np.zeros(steps)
    rotation_norms = np.zeros(steps)

    if loss_type == 'poisson':
        possion_loss = torch.nn.PoissonNLLLoss(log_input=False)

    for i in tqdm.trange(steps):
        optim.zero_grad()

        preds = model.pred_synapses(types_1hot)

        if loss_type == 'poisson':
            loss = possion_loss(preds.flatten()[validinds], yvalid)
        else:
            assert loss_type == 'mse'
            loss = torch.norm(preds.flatten()[validinds] - yvalid)**2


        if reg_fn is not None:
            tot_loss = loss + reg_fn(model)
        else:
            tot_loss = loss

        tot_loss.backward()

        if torch.isnan(tot_loss):
            raise ValueError('loss is nan')
        optim.step()

        losses[i] = loss.detach().numpy()
        embedding_norms[i] = torch.norm(model.embeddings).detach().numpy()

        if model.rotation_params is not None:
            rotation_norms[i] = torch.norm(model.rotation_params).detach().numpy()

        if loss_type == 'mse':
            losses[i] /= (torch.norm(yvalid)**2).numpy()

        if i % print_every == 0:
            print(i, f'normalized loss: {losses[i]:.4f}',
                f'avg sq embed norm: {torch.norm(model.embeddings).detach().numpy()**2 / _N:.2f}')
    
    return losses, embedding_norms, rotation_norms

def get_Jall_neuronall(datapath = "", min_num_per_type=5):
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


def sortsubtype(t, list_of_types):
    """
    Get indices of neurons whose type contains the string t.
    if the string ends with '=', then only neurons of that exact type are included.
    """
    if t[-1] == '=':
        t = t[:-1]
        inds = np.nonzero([t == x for x in list_of_types])[0]
    else:
        inds = np.nonzero([t in x for x in list_of_types])[0]
    sortinds = np.argsort(list_of_types[inds])
    inds = inds[sortinds]
    return inds


def get_inds_by_type(neuronall:pd.DataFrame, type_key:str, types_wanted:list):
    """
    Get indices of neurons whose type contains one of the strings in types_wanted.

    the "type" used for each neuron is specified by `type_key`, which should be 
    one of the columns of neuronall.
    """
    list_of_types = np.array(neuronall[type_key]).astype(str)
    return np.concatenate([sortsubtype(t, list_of_types) for t in types_wanted])


def prep_J_data(full_J_mat, all_neurons, types_wanted=[], split_LR=[]):
    """
    Produce J and one-hot type vectors in torch tensor format. Include only
    neurons of types in types_wanted.
    """

    allcx = get_inds_by_type(all_neurons, 'type', types_wanted)

    J = full_J_mat[allcx, :][:, allcx].astype(np.float32)
    N = J.shape[0]
    neurons = all_neurons.iloc[allcx, :]
    neurons.reset_index(inplace=True)

    for i in range(N):
        if np.sum([t in neurons.type[i] for t in split_LR]) > 0:
            # only look for L/R information at the end of the type string
            if '_L' in neurons.instance[i][-6:]:
                neurons.loc[i, 'type'] += '_L'
            if '_R' in neurons.instance[i][-6:]:
                neurons.loc[i, 'type'] += '_R'

    new_type_sort = np.argsort(neurons.type)
    neurons = neurons.iloc[new_type_sort, :]
    J = J[new_type_sort, :][:, new_type_sort]

    types_1hot, Ntype, typehash, uniqtypes = make_onehot_embedding(neurons, 'type')


    #adjacency matrix in row presynaptic, column postsynaptic format
    Adj = torch.tensor(J.T).float()

    return Adj, types_1hot, N, Ntype, neurons, uniqtypes, typehash


def make_onehot_embedding(neurons:pandas.DataFrame, type_key:str):
    uniqtypes = pd.unique(neurons[type_key])
    Ntypes = len(uniqtypes)
    typehash = dict(zip(uniqtypes, np.arange(Ntypes)))
    typeclasses = np.array([typehash[x] for x in neurons[type_key]])

    types_1hot = np.zeros([len(neurons), Ntypes])
    types_1hot[np.arange(len(neurons)), typeclasses] = 1.
    types_1hot = torch.tensor(types_1hot).float()
    return types_1hot, Ntypes, typehash, uniqtypes


def init_uniform_param(shape, scale):
    return torch.tensor(
        np.ones(shape) * scale, dtype=torch.float32, requires_grad=True)


def remove_diag(mat):
    return mat - torch.diag(torch.diag(mat))


def tick_maker(unique_types, types_onehot, return_idx=False):
    """
    Make tick labels for plotting. If dont_count is True, return the indices.
    """
    tick_inds = []
    tick_labels = []
    n_per_type = types_onehot.sum(0)
    for i, ut in enumerate(unique_types):
        tick_inds.append(torch.sum(n_per_type[:i]) + n_per_type[i] / 2)
        tick_labels.append(ut + f" ({int(n_per_type[i])})")
    if return_idx:
        return np.arange(len(unique_types)), tick_labels
    else:
        return tick_inds, tick_labels


def data_summary(J, neurons, type_onehot, plotting='syn_per_n', type_key='type'):
    uniqtypes = neurons[type_key].unique()
    
    n_per_type = type_onehot.sum(0)

    # number of synapses between a pair of cell types
    conn_count = type_onehot.T @ J @ type_onehot
    num_synapse_per_neuron = conn_count / n_per_type[None, :]


    # number of connections between a pair of cell types
    connected = type_onehot.T @ (J > 0).float() @ type_onehot
    num_connections_per_neuron = connected / n_per_type[None, :]

    # number of possible connections between a pair of cell types
    # total_synapses = type_onehot.T @ np.ones_like(J) @ type_onehot

    plt.figure()
    # plt.imshow(connected / total_synapses, cmap='gray_r')
    if plotting == 'syn_per_n':
        plt.imshow(num_synapse_per_neuron, cmap='gray_r')
    elif plotting == 'con_per_n':
        plt.imshow(num_connections_per_neuron, cmap='gray_r')
    plt.xticks(*tick_maker(uniqtypes, type_onehot, return_idx=True),
               rotation=90)
    plt.yticks(*tick_maker(uniqtypes, type_onehot, return_idx=True))
    plt.xlabel('post-synaptic')
    plt.ylabel('pre-synaptic')
    plt.colorbar()