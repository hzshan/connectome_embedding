import numpy as np
import torch
import scipy, data_utils, pandas, tqdm, data_utils
from data_utils import ConnectivityData


def make_rotation_mats(rotation_params):
    """
    rotation_params: (Ntype, D, D)

    Recover SO(D) matrices via matrix exponential
    """
    anti_sym = rotation_params - rotation_params.transpose(-1, -2)

    return torch.matrix_exp(anti_sym)


def make_circle(n_points, wrap=False):
    """
    Make a circle in 2D. If wrap is True, add an extra point to close the
    circle.
    """
    if wrap:
        angles = torch.linspace(0, 2 * np.pi, n_points + 1)
    else:
        angles = torch.arange(n_points) * 2 * np.pi / n_points
    unit_circle = torch.stack([torch.cos(angles), torch.sin(angles)], 1)

    return unit_circle


def tick_maker(unique_types, onehot_types, return_idx=False, include_counts=True):
    """
    Make tick labels for plotting. If dont_count is True, return the indices.
    """
    tick_inds = []
    tick_labels = []
    n_per_type = onehot_types.sum(0)
    for i, ut in enumerate(unique_types):
        tick_inds.append(torch.sum(n_per_type[:i]) + n_per_type[i] / 2)
        if include_counts:
            tick_labels.append(ut + f" ({int(n_per_type[i])})")
        else:
            tick_labels.append(ut)
    if return_idx:
        return np.arange(len(unique_types)), tick_labels
    else:
        return tick_inds, tick_labels


def sort_embeddings(embeddings, data: data_utils.ConnectivityData,
                    full_method=False, reference_type=None):
    """
    Sort the embeddings. 
    
    Respects boundaries between types. E.g. if the first 30 neurons are type 1, 
    second 20 neurons are type 2, ..., then after sorting, the first 30 neurons
    are still type 1, second 20 neurons are still type 2, etc.

    Full method for sorting: solve shadowmatic problem for embeddings of each
    cell type, and then order the neurons by their location on the circle. This
    can be slow for large datasets.

    "Cheap method": project all embeddings on the top two PCs. Sort by the
    arctan.

    Args:
    embeddings: N x D torch tensor, the embeddings
    data: ConnectivityData, the data object
    full_method: bool, whether to use the full method for sorting

    Returns:
    sorted_inds: N-dim np.array, the indices of the sorted embeddings
    """
    if reference_type is None:
        _, _, V = torch.svd(embeddings)
        projected_all_embeddings = embeddings @ V
    else:
        ref_embeddings = embeddings[data.neuron_hash[reference_type]]
        ref_embeddings -= ref_embeddings.mean(0)
        _, proj, _  = solve_shadowmatic2(
            ref_embeddings, nseeds_to_try=50, max_iter=10
        )
        projected_all_embeddings = embeddings @ proj

    sorted_inds = []

    for i in tqdm.trange(data.Ntype):
        start_ind = int(data.N_per_type[:i].sum())
        end_ind = int(data.N_per_type[:i+1].sum())

        if full_method:
            # solve shadowmatic problem for embeddings of each cell type
            _, _, best_perm = solve_shadowmatic2(
                embeddings[start_ind: end_ind], 50
            )
            inds = best_perm @ np.arange(len(best_perm))
            sorted_inds += list(inds + len(sorted_inds))
        else:
            angle = np.arctan2(projected_all_embeddings[start_ind:end_ind, 1],
                            projected_all_embeddings[start_ind:end_ind, 0])
            sorted_inds += list(np.argsort(angle) + len(sorted_inds))
    sorted_inds = np.array(sorted_inds)
    return sorted_inds


def get_adjacency_of_type(cell_type: str, data, cutoff=5):
        """
        For the given cell type, find its most connected cell types in the data.

        For the summary of type A, the "synapses per neuron" from A to B is the
        total number of synapses from A cells to B cells, divided by the number
        of A cells. The number of synapses from B to A is defined similarly (
        also divided by the number of A cells).

        Args:
            cell_type: the type of cells to summarize the connectivity for
            data: the ConnectivityData object
            cutoff: only types with more than this number of synapses (either direction) per neuron are shown
        Returns:
            labels: the labels of the connected types
            in_out_counts: N_connected_types x 2. The first column is the number
              of synapses from each connected type to the inquiry type; the 
              second column is the number of synapses from the inquiry type to each connected type.
        """
        assert cell_type in data.types, f"{cell_type} not in data.types"
        ind = data.type_hash[cell_type]

        num_synapses_bt_cell_types = (data.onehot_types.T @ data.J @ data.onehot_types)

        out_synapses = num_synapses_bt_cell_types[ind] / data.N_per_type[ind]
        in_synapses = num_synapses_bt_cell_types[:, ind] / data.N_per_type[ind]

        tot_synapses = in_synapses + out_synapses

        # only keep types that are connected to the inquiry type
        connected_types = data.types[tot_synapses > cutoff]
        in_synapses = in_synapses[tot_synapses > cutoff]
        out_synapses = out_synapses[tot_synapses > cutoff]
        tot_synapses = tot_synapses[tot_synapses > cutoff]

        sort_ind = np.argsort(tot_synapses.numpy())[::-1].copy()

        labels = connected_types[sort_ind]
        in_out_counts = torch.stack([in_synapses[sort_ind], out_synapses[sort_ind]], dim=1).numpy()

        return labels, in_out_counts


def calculate_circularity_via_dev(projected_embeddings):
    """
    Compute a circularity score for the given set of 2D embeddings. 

    Let (r_i,\theta_i) be the polar coordinates of the i-th point. 
    Let i=1,...,N index the points, sorted by $\theta$.

    The "angular deviation" score is given by
    (2pi/N)^{-2} mean({(\theta_{i+1} - \theta_i - 2pi/N)^2})

    where the index is modulo N.
     This score is 0 if the points are evenly
    distributed on the circle, and 1 if they are all at the same location.

    The "distance deviation" score is given by
    N^{-1} \\sum_i (\\hat{r_i} - 1)^2

    where \\hat{r_i} = r_i / mean({r_i}).
    This score is 0 if the points are all at the same distance from the origin,
    and 1 if they are all at the same location.

    Args:
        projected_embeddings: N x 2 tensor

    Returns:
        angular_deviation: the angular deviation score (smaller=more circular)
        distance_deviation: the distance deviation score (smaller=more circular)
    """
    N = projected_embeddings.shape[0]

    # get polar coordinates
    dists = torch.sqrt(torch.sum(projected_embeddings**2, 1))
    angles = torch.atan2(projected_embeddings[:, 0], projected_embeddings[:, 1])

    # normalize the distances to have mean 1; sort the angles
    dists /= dists.mean()
    sorted_angles = torch.sort(angles, dim=0).values

    adjacent_diff = torch.roll(sorted_angles, shifts=-1, dims=0) - sorted_angles
    adjacent_diff[-1] += 2 * np.pi
    assert torch.all(adjacent_diff >= 0)

    ideal_diff = 2 * np.pi / N
    angular_deviation = torch.mean((adjacent_diff - ideal_diff)**2) / ideal_diff**2

    ideal_dist = 1
    distance_deviation = torch.mean((dists - ideal_dist)**2) / 0.273

    return angular_deviation, distance_deviation


def solve_shadowmatic(embeddings, nseeds_to_try, max_steps=500, lr=0.01):
    """
    For a set of D-dim vectors, find an affine transform to 2D such that they
    distribute in a circle-like manner.

    Let P be the Dx2 projection matrix and B be the 2 bias term. Let Z be the
    NxD embeddings. The projected embeddings are given by Z'=ZP - B.

    The optimization runs gradient descent on P, B to minimize
    
    loss = std({c_i^2})/mean({c_i^2}) + N^{-2} \\sum_{i,j} exp(cos(\theta_i - \theta_j))

    where (c_i, \theta_i) are the polar coordinates of the i-th point in Z'.

    Args:
        embeddings: NxD, the set of N points in D dims to be projected
        nseeds_to_try: number of random seeds to try
        max_steps: maximum number of optimization steps
        lr: learning rate

    Returns:
        proj_mat: Dx2, the found best Dx2 projection
        bias: 2, the bias term (used to center the set of projected points)
        projected_embeddings: Nx2, the projected points
        losses: the loss values at each step
    """
    N, D = embeddings.shape
    torch.manual_seed(0)

    # initialize the Dx2 projection matrices and bias terms
    proj_mat = torch.normal(0, 1 / np.sqrt(D), (D, 2, nseeds_to_try))
    proj_mat.requires_grad = True
    bias = torch.zeros((2, nseeds_to_try), requires_grad=True)

    optim = torch.optim.Adam([proj_mat, bias], lr=lr)

    losses = []

    convergence_counter = 0

    for step_idx in range(max_steps):
        optim.zero_grad()
        projected_embeddings = torch.einsum('nd,das->nas',
                            embeddings, proj_mat) - bias
        angles = torch.atan2(projected_embeddings[:, 0, :],
                             projected_embeddings[:, 1, :])  # N x Nseeds
        angles_diff = angles.view(-1, 1, nseeds_to_try) - \
            angles.view(1, -1, nseeds_to_try)   # N x N x Nseeds

        angular_spread_loss = torch.exp(torch.cos(angles_diff)).mean(dim=(0, 1))

        dist_sq = torch.sum(projected_embeddings**2, 1)  # N x Nseeds
        dist_loss = torch.std(dist_sq, dim=0) / torch.mean(dist_sq, dim=0)

        loss = torch.sum(angular_spread_loss + dist_loss)
        loss.backward()

        optim.step()

        # normalize the projection matrix so that the mean distance
        # of the projected points to the origin is 1
        factor = torch.mean(
            torch.sqrt(torch.sum(projected_embeddings**2, 1)), dim=0)
        proj_mat.data = proj_mat.data / factor
        bias.data = bias.data / factor

        if step_idx > 0 and abs(losses[-1] - loss.item()) < 1e-6:
            convergence_counter = 1
            break

        losses.append(loss.item())

    if convergence_counter == 0:
        print('some seeds did not converge') 


    # find out which seed worked best
    criterion = np.zeros(nseeds_to_try)

    for i in range(nseeds_to_try):
        _ang_dev, _dist_dev = calculate_circularity_via_dev(
            projected_embeddings[:, :, i].detach())
        criterion[i] += _ang_dev
        criterion[i] += _dist_dev

    best_seed = np.argmin(criterion)
    best_proj_mat = proj_mat[:, :, best_seed].detach()
    best_bias = bias[:, best_seed].detach()

    best_projected_embeddings = embeddings @ best_proj_mat - best_bias

    return (
        best_proj_mat,
        best_bias,
        best_projected_embeddings,\
        losses)


def find_perm(target, embeddings):
    """
    Solves the linear assignment problem to find the best row permutation
    of embeddings that minimizes the L2 distance to target.

    Args:
        target: NxD, the target points
        embeddings: NxD, the points to be permuted

    Returns:
        P: the permutation matrix. P @ embeddings is the permuted embeddings
        P @ embeddings: the permuted embeddings
    """
    cost = torch.cdist(target, embeddings)
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cost)
    perm_mat = torch.zeros((embeddings.shape[0], embeddings.shape[0]))
    perm_mat[col_ind, row_ind] = 1
    return perm_mat.T, perm_mat.T @ embeddings


def solve_shadowmatic2(embeddings, nseeds_to_try: int, max_iter=50):
    """
    Finds the best Dx2 projection of the given set of points that distributes
    in a circle-like manner. Attempts to solve the optimization problem over 
    permutation matrix Pi and linear projection T to minimize
    ||C - Pi @ E @ T||, where C is the circle and E is the embeddings.

    Args:
        embeddings: NxD tensor, the set of N points in D dims to be projected
        nseeds_to_try: number of random seeds to try
        max_iter: maximum number of iterations for the iterative optimization
    
    Returns:
        projected_embeddings: Nx2 tensor, the projected points, ordered along the circle
        best_proj: Dx2 tensor, the best projection matrix
        best_perm: the permutation matrix used on the embeddings; best_perm.T @ projected_embeddings recovers the original ordering
    """
    projected_embs_across_seeds = []
    scores_across_seeds = []
    projection_mats_across_seeds = []
    total_perms_across_seeds = []

    N, D = embeddings.shape

    # generate the target circle
    circle = make_circle(N)

    for seed in range(nseeds_to_try):

        embeddings0 = embeddings.clone()
        torch.manual_seed(seed)

        best_proj = torch.normal(
            0, 1, (D, 2)) # initialize a random projection

        curr_best_proj = best_proj.clone()

        total_perm = torch.eye(N)

        for i in range(max_iter):
            projected_embs0 = embeddings0 @ best_proj

            # center and normalize
            # projected_embs0 -= projected_embs0.mean()
            # projected_embs0 *= torch.norm(circle) / torch.norm(projected_embs0)

            best_perm, _ = find_perm(circle, projected_embs0)
            embeddings0 = best_perm @ embeddings0

            total_perm = best_perm @ total_perm

            best_proj = torch.linalg.lstsq(embeddings0, circle).solution

            if torch.allclose(curr_best_proj, best_proj):
                break
                
            curr_best_proj = best_proj.clone()

        projected_embs0 = embeddings0 @ best_proj

        # center and normalize
        projected_embs0 -= projected_embs0.mean(0)
        projected_embs0 *= torch.norm(circle) / torch.norm(projected_embs0)

        projected_embs_across_seeds.append(projected_embs0)
        scores_across_seeds.append(circ_score_via_norm(projected_embs0, skip_alignment=True))
        projection_mats_across_seeds.append(curr_best_proj)
        total_perms_across_seeds.append(total_perm)

    return (projected_embs_across_seeds[np.argmax(scores_across_seeds)],
            projection_mats_across_seeds[np.argmax(scores_across_seeds)],
            total_perms_across_seeds[np.argmax(scores_across_seeds)])


def circ_score_via_norm(embeddings_2d, skip_alignment=False):
    """
    Compute the circularity of the given 2D embeddings by comparing the
    distance to the unit circle.

    The circularity score is given by ||C-phi(E)||/||C||, where C is the unit
    circle and phi is the best 2x2 affine transformation that aligns the
    embeddings to the circle. If skip_alignment is True, the alignment step is
    skipped.

    Args:
        embeddings: Nx2 tensor, the set of N points to be evaluated
        skip_alignment: whether to skip the alignment step
    
    Returns:
        circularity: the circularity score
    """ 
    if torch.isnan(embeddings_2d).any():
        return torch.nan

    N, D = embeddings_2d.shape
    assert D == 2

    # generate the target circle
    unit_circle = make_circle(N)

    if skip_alignment:
        centered_embeddings = embeddings_2d - embeddings_2d.mean(0)
        return 1 - torch.norm(unit_circle - embeddings_2d)**2 / torch.norm(centered_embeddings)**2

    else:
        raise NotImplementedError("This part of the code is deprecated.")
        embeddings0, _, _ = solve_shadowmatic2(
            embeddings_2d, nseeds_to_try=10, max_iter=5)

        embeddings0 = embeddings_2d.clone() - embeddings_2d.mean(0)
        embeddings0 *= torch.norm(unit_circle) / torch.norm(embeddings0)

        best_perm, _ = find_perm(unit_circle, embeddings0)
        embeddings0 = best_perm @ embeddings0

        two_by_two_proj = torch.linalg.lstsq(embeddings0, unit_circle).solution

        embeddings0 = embeddings0 @ two_by_two_proj
        embeddings0 = embeddings0 - embeddings0.mean(0)
        embeddings0 *= torch.norm(unit_circle) / torch.norm(embeddings0)
        

        return torch.norm(unit_circle - embeddings0) / torch.norm(unit_circle)


def calculate_fraction_of_projected_var(embeddings, projection_mat):
    """
    Calculate the fraction of variance of the embeddings that is captured by
    the given linear transformation.

    Args:
        embeddings: NxD tensor, the set of N points in D dims
        projection_mat: Dx2 tensor, the projection matrix
    """
    _, D = embeddings.shape
    assert projection_mat.shape == (D, 2)
    u, _, _ = torch.svd(projection_mat)
    subspace = u @ u.T
    centered_embeddings = embeddings - embeddings.mean(0)
    return (torch.norm(centered_embeddings @ subspace)**2 /
            torch.norm(centered_embeddings)**2)