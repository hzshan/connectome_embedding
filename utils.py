import pandas as pd
import numpy as np
import torch
import tqdm

def train_model(model, target_mat, types_1hot,
                loss_type='poisson', lr=0.01, steps=40000,
                print_every=2000, reg_fn=None):
    _N = target_mat.shape[0]
    validinds = np.where((np.ones([_N, _N]) - np.diag(np.ones(_N))).flatten())[0]
    yvalid = target_mat.flatten()[validinds]
    optim = torch.optim.Adam(params=model.params, lr=lr)

    losses = np.zeros(steps)

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
        if loss_type == 'mse':
            losses[i] /= (torch.norm(yvalid)**2).numpy()

        if i % print_every == 0:
            print(i, f'normalized loss: {losses[i]:.4f}',
                f'avg sq embed norm: {torch.norm(model.embeddings).detach().numpy()**2 / _N:.2f}')
    
    return losses

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
        if _count > 10:
            inds_to_keep += list(np.where(np.array(neuronsall.type).astype(str) == _type)[0])

    return Jall[inds_to_keep][:, inds_to_keep], neuronsall.take(inds_to_keep)


def sortsubtype(t, types):
    """
    Get indices of neurons whose type contains the string t.
    if the string ends with '=', then only neurons of that exact type are included.
    """
    if t[-1] == '=':
        t = t[:-1]
        inds = np.nonzero([t == x for x in types])[0]
    else:
        inds = np.nonzero([t in x for x in types])[0]
    sortinds = np.argsort(types[inds])
    inds = inds[sortinds]
    return inds


def prep_J_data(full_J_mat, all_neurons, types_wanted=[], separate_LR=False):
    """
    Produce J and one-hot type vectors in torch tensor format. Include only
    neurons of types in types_wanted.
    """
    types = np.array(all_neurons.type).astype(str)


    allcx = np.concatenate([sortsubtype(t, types) for t in types_wanted])

    J = full_J_mat[allcx, :][:, allcx].astype(np.float32)
    N = J.shape[0]
    neurons = all_neurons.iloc[allcx, :]
    neurons.reset_index(inplace=True)

    if separate_LR:
        for i in range(N):

            # if neurons.type[i] == "EPG":
            #     continue

            # only look for L/R information at the end of the type string
            if '_L' in neurons.instance[i][-6:]:
                neurons.loc[i, 'type'] += '_L'
            if '_R' in neurons.instance[i][-6:]:
                neurons.loc[i, 'type'] += '_R'

    new_type_sort = np.argsort(neurons.type)
    neurons = neurons.iloc[new_type_sort, :]
    J = J[new_type_sort, :][:, new_type_sort]

    uniqtypes = pd.unique(neurons.type)
    Ntype = len(uniqtypes)
    typehash = dict(zip(uniqtypes, np.arange(Ntype)))
    typeclasses = np.array([typehash[x] for x in neurons.type])

    types_1hot = np.zeros([N, Ntype])
    types_1hot[np.arange(N), typeclasses] = 1.
    types_1hot = torch.tensor(types_1hot).float()

    #adjacency matrix in row presynaptic, column postsynaptic format
    Adj = torch.tensor(J.T).float()

    return Adj, types_1hot, N, Ntype, neurons, uniqtypes, typehash


def init_uniform_param(shape, scale):
    return torch.tensor(
        np.ones(shape) * scale, dtype=torch.float32, requires_grad=True)


def remove_diag(mat):
    return mat - torch.diag(torch.diag(mat))