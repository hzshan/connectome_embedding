import pandas as pd
import numpy as np
import torch
import tqdm, pandas
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import scipy


class ConnectivityData:
    def __init__(self, J, neurons:pandas.DataFrame, type_key:str):
        self.J = J
        self.N = J.shape[0]

        types_1hot, Ntype, typehash, uniqtypes, neuron_ind_by_type = \
            make_onehot_embedding(neurons, type_key)
        
        self.onehot_types = types_1hot
        self.Ntype = Ntype
        self.type_hash = typehash
        self.types = uniqtypes
        self.neuron_hash = neuron_ind_by_type
        self.neurons_pandas = neurons
        self.N_per_type = types_1hot.sum(0)

        assert self.onehot_types.shape == (self.N, self.Ntype)
        assert len(self.types) == self.Ntype


def train_model(model, target_mat,
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

        preds = model.pred_synapses()

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

        # if model.rotation_params is not None:
        #     rotation_norms[i] = torch.norm(model.rotation_params).detach().numpy()

        if loss_type == 'mse':
            losses[i] /= (torch.norm(yvalid)**2).numpy()

        if (i+1) % print_every == 0:
            print(i, f'normalized loss: {losses[i]:.4f}',
                f'avg sq embed norm: {torch.norm(model.embeddings).detach().numpy()**2 / _N:.2f}')
    
    return losses, embedding_norms, rotation_norms


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


def prep_J_data(full_J_mat, all_neurons, types_wanted=[], split_LR=None):
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


def make_onehot_embedding(neurons:pandas.DataFrame, type_key:str):
    uniqtypes = pd.unique(neurons[type_key])
    Ntypes = len(uniqtypes)
    typehash = dict(zip(uniqtypes, np.arange(Ntypes)))
    typeclasses = np.array([typehash[x] for x in neurons[type_key]])

    types_1hot = np.zeros([len(neurons), Ntypes])
    types_1hot[np.arange(len(neurons)), typeclasses] = 1.
    types_1hot = torch.tensor(types_1hot).float()
    neuron_ind_by_type = {uniqtypes[i]:np.where(typeclasses == i)[0] for i in range(Ntypes)}
    return types_1hot, Ntypes, typehash, uniqtypes, neuron_ind_by_type


def init_uniform_param(shape, scale):
    return torch.tensor(
        np.ones(shape) * scale, dtype=torch.float32, requires_grad=True)


def remove_diag(mat):
    return mat - torch.diag(torch.diag(mat))


def tick_maker(unique_types, onehot_types, return_idx=False):
    """
    Make tick labels for plotting. If dont_count is True, return the indices.
    """
    tick_inds = []
    tick_labels = []
    n_per_type = onehot_types.sum(0)
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

    plt.gca()
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
    


def calculate_circularity(projected_embeddings):
    """
    
    Args:
        projected_embeddings: N x 2 tensor
    """
    n_neurons = projected_embeddings.shape[0]
    angles = torch.atan2(projected_embeddings[:, 0], projected_embeddings[:, 1])
    sorted_angles = torch.sort(angles, dim=0).values
    adjacent_diff = torch.roll(sorted_angles, shifts=-1, dims=0) - sorted_angles
    adjacent_diff[-1] += 2 * np.pi

    assert torch.all(adjacent_diff >= 0)
    ideal_diff = 2 * np.pi / n_neurons
    angular_deviation = torch.mean((adjacent_diff - ideal_diff)**2) / ideal_diff**2
    # angular_deviation = torch.std(adjacent_diff, correction=0) \
    #     / adjacent_diff.mean() / np.sqrt(n_neurons - 1)
    # this is the normalized standard deviation of the adjacent angles.
    # sqrt(N-1) is the max value it could have (the adjacent angles are
    # [0, 0, ..., 2pi])

    dists = torch.sqrt(torch.sum(projected_embeddings**2, 1))
    dists /= dists.mean()
    distance_deviation = torch.mean((dists - 1)**2) / 0.273

    # distance_deviation = torch.std(dists, correction=0) \
    #     / dists.mean() / np.sqrt(n_neurons - 1)
    # this is the normalized standard deviation of the distance from origin.
    # sqrt(N-1) is the max value it could have under a fixed total sq dist = N.
    # (the distances are [0, 0, ..., sqrt(N)])
    return angular_deviation, distance_deviation


def solve_shadowmatic(embeddings, nseeds_to_try, max_steps=500, lr=0.01):
    """
    For a set of points in embeddings, find a 2D projection such that they
    distribute in a circle-like manner.

    Args:
        embeddings: NxD, the set of N points in D dims to be projected
        nseeds_to_try: number of random seeds to try
        max_steps: maximum number of optimization steps
        lr: learning rate

    Returns:
        proj_mat: Dx2, the found best projection basis (may not be orthonormal)
        bias: 2, the bias term (used to center the set of projected points)
        proj: Nx2, the projected points
    """
    n_neurons, n_dims = embeddings.shape
    torch.manual_seed(0)

    # initialize the projection matrices and bias terms
    proj_mat = torch.normal(0, 1 / np.sqrt(n_dims), (n_dims, 2, nseeds_to_try))
    proj_mat.requires_grad = True
    bias = torch.zeros((2, nseeds_to_try), requires_grad=True)

    adam = torch.optim.Adam([proj_mat, bias], lr=lr)

    losses = []

    convergence_counter = 0

    for step_idx in range(max_steps):
        adam.zero_grad()
        proj = torch.einsum('nd,dab->nab',
                            embeddings, proj_mat) - bias
        angles = torch.atan2(proj[:, 0, :], proj[:, 1, :])  # N x B
        angles_diff = angles.view(-1, 1, nseeds_to_try) - \
            angles.view(1, -1, nseeds_to_try)

        angular_spread_loss = torch.exp(torch.cos(angles_diff)).mean(dim=(0, 1))

        dist_sq = torch.sum(proj**2, 1)
        dist_loss = torch.std(dist_sq, dim=0) / torch.mean(dist_sq, dim=0)

        loss = torch.sum(angular_spread_loss + dist_loss)
        loss.backward()

        adam.step()

        # normalize the projection matrix so that the mean distance
        # of the projected points to the origin is 1
        factor = torch.mean(torch.sqrt(torch.sum(proj**2, 1)), dim=0)
        proj_mat.data = proj_mat.data / factor
        bias.data = bias.data / factor


        if step_idx > 0 and abs(losses[-1] - loss.item()) < 1e-6:
            convergence_counter = 1
            break


        losses.append(loss.item())

    if convergence_counter == 0:
        print('some seeds did not converge') 


    # find out which seed worked best
    angle_dev = torch.zeros(nseeds_to_try)
    dist_dev = torch.zeros(nseeds_to_try)

    for i in range(nseeds_to_try):
        _ang_dev, _dist_dev = calculate_circularity(
            proj[:, :, i].detach())
        angle_dev[i] = _ang_dev
        dist_dev[i] = _dist_dev

    criterion = angle_dev + dist_dev
    best_seed = np.argmin(criterion)

    best_proj_mat = proj_mat[:, :, best_seed].detach()
    best_bias = bias[:, best_seed].detach()

    output = embeddings @ best_proj_mat - best_bias

    return (
        best_proj_mat,
        best_bias,
        losses,
        output,
        angle_dev[best_seed],
        dist_dev[best_seed])


def find_perm(target, embeddings):
    """
    Solves the linear assignment problem to find the best row permutation
    of embeddings that minimizes the distance to target.

    Args:
        target: NxN_dim tensor, the target points
        embeddings: NxN_dim tensor, the points to be permuted

    Returns:
        P: the permutation matrix. P @ embeddings is the permuted embeddings
        P @ embeddings: the permuted embeddings
    """
    cost = torch.cdist(target, embeddings)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
    perm_mat = torch.zeros((embeddings.shape[0], embeddings.shape[0]))
    perm_mat[col_ind, row_ind] = 1
    return perm_mat.T, perm_mat.T @ embeddings


def find_best_proj(circle, embeddings):
    """
    Solves the linear least square problem to find the best N_dimx2 projection
    T such that ||C - E @ T|| is minimized, where C is the circle and E is the
    permuted embeddings.

    Args:
        circle: Nx2 tensor, the target circle
        embeddings: NxN_dim tensor, the embeddings embeddings
    
    Returns:
        best_proj: N_dimx2 tensor, the best projection matrix
    """
    n_neurons, _ = circle.shape
    assert embeddings.shape[0] == n_neurons
    regularizer = torch.eye(embeddings.shape[1]) * 1e-6

    return torch.inverse(embeddings.T @ embeddings + regularizer) @ embeddings.T @ circle


def solve_shadowmatic2(embeddings, nseeds_to_try, max_iter=50):
    """
    Finds the best 2D projection of the given set of points that distributes
    in a circle-like manner. Attempts to solve the optimization problem over 
    permutation matrix Pi and linear projection T to minimize
    ||C - Pi @ E @ T||, where C is the circle and E is the embeddings.

    Args:
        embeddings: NxD tensor, the set of N points in D dims to be projected
        nseeds_to_try: number of random seeds to try
        max_iter: maximum number of iterations for the iterative optimization
    
    Returns:
        projected_embeddings: Nx2 tensor, the projected points
        best_proj: Dx2 tensor, the best projection matrix
        best_perm: the permutation matrix used on the embeddings; 
            best_perm.T @ projected_embeddings recovers the original ordering
    """
    projected_embs_across_seeds = []
    scores = []
    projection_mats = []
    total_perms = []

    n_neurons, n_dims = embeddings.shape

    angles = torch.arange(n_neurons) / n_neurons * 2 * np.pi
    circle = torch.stack([torch.cos(angles), torch.sin(angles)], 1)

    for seed in range(nseeds_to_try):

        embeddings0 = embeddings.clone()
        torch.manual_seed(seed)

        best_proj = torch.normal(
            0, 1, (n_dims, 2)) # initialize a random projection

        curr_best_proj = best_proj.clone()

        total_perm = torch.eye(n_neurons)

        for i in range(max_iter):
            projected_embs0 = embeddings0 @ best_proj

            # center and normalize
            projected_embs0 -= projected_embs0.mean()
            projected_embs0 *= torch.norm(circle) / torch.norm(projected_embs0)

            best_perm, _ = find_perm(circle, projected_embs0)
            embeddings0 = best_perm @ embeddings0

            total_perm = best_perm @ total_perm

            best_proj = find_best_proj(circle, embeddings0)

            if torch.allclose(curr_best_proj, best_proj):
                break
                
            curr_best_proj = best_proj.clone()

        projected_embs0 = embeddings0 @ best_proj

        # center and normalize
        projected_embs0 -= projected_embs0.mean(0)
        projected_embs0 *= torch.norm(circle) / torch.norm(projected_embs0)

        projected_embs_across_seeds.append(projected_embs0)
        scores.append(compute_circularity_via_norm(projected_embs0, skip_alignment=True))
        projection_mats.append(curr_best_proj)
        total_perms.append(total_perm)
    
    return projected_embs_across_seeds[np.argmin(scores)], projection_mats[np.argmin(scores)], total_perms[np.argmin(scores)]


def compute_circularity_via_norm(embeddings, skip_alignment=False):
    """
    Compute the circularity of the given set of points by comparing the
    distance to the unit circle.

    Args:
        embeddings: Nx2 tensor, the set of N points to be evaluated
        skip_alignment: whether to skip the alignment step
    
    Returns:
        circularity: the circularity score
    """ 
    if torch.isnan(embeddings).any():
        return torch.nan

    n_neurons, n_dims = embeddings.shape
    assert n_dims == 2
    angles = torch.arange(n_neurons) / n_neurons * 2 * np.pi
    unit_circle = torch.stack([torch.cos(angles), torch.sin(angles)], 1)

    if skip_alignment:
        return torch.norm(unit_circle - embeddings) / torch.norm(unit_circle)

    else:

        embeddings0 = embeddings.clone() - embeddings.mean(0)
        embeddings0 *= torch.norm(unit_circle) / torch.norm(embeddings0)

        best_perm, _ = find_perm(unit_circle, embeddings0)
        embeddings0 = best_perm @ embeddings0

        two_by_two_proj = find_best_proj(unit_circle, embeddings0)

        embeddings0 = embeddings0 @ two_by_two_proj
        embeddings0 = embeddings0 - embeddings0.mean(0)
        embeddings0 *= torch.norm(unit_circle) / torch.norm(embeddings0)
        

        return torch.norm(unit_circle - embeddings0) / torch.norm(unit_circle)


def calculate_fraction_of_projected_var(embeddings, linear_transform):
    _, n_dim = embeddings.shape
    assert linear_transform.shape == (n_dim, 2)
    u, _, _ = torch.svd(linear_transform)
    subspace = u @ u.T
    centered_embeddings = embeddings - embeddings.mean(0)
    return torch.norm(centered_embeddings @ subspace)**2 / torch.norm(centered_embeddings)**2
