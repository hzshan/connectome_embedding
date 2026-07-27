import numpy as np
import torch
from . import utils

def make_grid_cell_connections(X1, Y1, X2, Y2, sheet_width, inv_width):
    # Create a grid of neurons
    N1 = len(X1)
    N2 = len(X2)

    X_diff = X1.reshape(-1, 1) - X2.reshape(1, -1)
    Y_diff = Y1.reshape(-1, 1) - Y2.reshape(1, -1)

    # Apply periodic boundary conditions by wrapping the differences
    X_diff = np.abs(X_diff)
    Y_diff = np.abs(Y_diff)

    # For periodic boundary conditions, wrap around when the difference exceeds half the box size
    half_box_size = sheet_width / 2.0

    # Adjust distances for periodic boundary conditions
    X_diff = np.where(X_diff > half_box_size, 2 * half_box_size - X_diff, X_diff)
    Y_diff = np.where(Y_diff > half_box_size, 2 * half_box_size - Y_diff, Y_diff)
    sq_dist = X_diff**2 + Y_diff**2

    # W = np.abs(np.exp(-1.05 * beta / lambda_w**2 * sq_dist) -、
    #  np.exp(-beta / lambda_w**2 * sq_dist))
    W = np.exp(-inv_width * sq_dist)
    W *= 100 / W.max()

    W = torch.tensor(W)
    # W = torch.bernoulli(W)
    assert W.shape == (N1, N2)

    return W


def make_synthetic_grid_cell_weights(
        n_excitatory=30, n_inhibitory=20, lambda_w=None, beta=None):
    """
    This function creates a synthetic weight matrix for a grid cell network
        with periodic boundary conditions. For parameters (e.g., lambda, beta),
        see Burak and Fiete 2009.

    Args:
        n_excitatory: number of excitatory neurons along one dimension (total excitatory neurons = n_excitatory^2)
        n_inhibitory: number of inhibitory neurons along one dimension (total inhibitory neurons = n_inhibitory^2)
        lambda_w: the scale of the connectivity pattern. If None, it will be set to n_excitatory / 128 * 13, following Burak and Fiete.
        beta: width parameter. See Burak and Fiete 2009 for details.
    """


    N_e = n_excitatory**2  # size of each module
    N_i = n_inhibitory**2
    full_N = N_e + N_i  # size of the full network

    sheet_width = n_excitatory
    
    if beta is None:
        beta = 3.0
    if lambda_w is None:
        lambda_w = n_excitatory / 128 * 13  # 128 is the n in Burak and Fiete. 

    range_e = np.linspace(-sheet_width / 2, sheet_width / 2, n_excitatory, endpoint=False)
    range_e -= range_e.mean()  # center the grid around (0,0)
    X_e, Y_e = np.meshgrid(range_e, range_e)
    X_e = X_e.flatten()
    Y_e = Y_e.flatten()

    range_i = np.linspace(-sheet_width / 2, sheet_width / 2, n_inhibitory, endpoint=False)
    range_i -= range_i.mean()  # center the grid around (0,0)
    X_i, Y_i = np.meshgrid(range_i, range_i)
    X_i = X_i.flatten()
    Y_i = Y_i.flatten()


    W_EE = make_grid_cell_connections(X_e, Y_e, X_e, Y_e,
                                    sheet_width=sheet_width,
                                    inv_width=1.05 * beta / lambda_w**2)
    W_EI = make_grid_cell_connections(X_e, Y_e, X_i, Y_i,
                                    sheet_width=sheet_width,
                                    inv_width=beta / 2 / lambda_w**2)
    W_IE = W_EI.T


    full_W = torch.zeros((full_N, full_N))
    full_W[:N_e, :N_e] = W_EE
    full_W[:N_e, N_e:] = W_EI
    full_W[N_e:, :N_e] = W_IE


    full_W -= np.diag(np.diag(full_W))  # remove self-connections
    full_W *= 100 / full_W.max()  # normalize weights
    
    return full_W


def make_clifford_torus(torus_n):
    thetas = np.linspace(0, 2 * np.pi, torus_n, endpoint=False)
    phis = np.linspace(0, 2 * np.pi, torus_n, endpoint=False)

    thetas, phis = np.meshgrid(thetas, phis)

    w = np.cos(thetas)
    x = np.sin(thetas)
    y = np.cos(phis)
    z = np.sin(phis)

    flattened = np.zeros((torus_n**2, 4))
    flattened[:, 0] = w.flatten()
    flattened[:, 1] = x.flatten()
    flattened[:, 2] = y.flatten()
    flattened[:, 3] = z.flatten()
    return torch.tensor(flattened)


def solve_shadowmatic_torus(embeddings, target_shape, nseeds_to_try: int, max_iter=50):
    """
    """
    projected_embs_across_seeds = []
    scores = []
    projection_mats = []
    total_perms = []

    N, D = embeddings.shape

    assert target_shape.shape[0] == N
    D_target = target_shape.shape[1]


    for seed in range(nseeds_to_try):

        embeddings0 = embeddings.clone()
        torch.manual_seed(seed)

        best_proj = torch.normal(
            0, 1, (D, D_target)) # initialize a random projection

        curr_best_proj = best_proj.clone()

        total_perm = torch.eye(N)

        for i in range(max_iter):
            projected_embs0 = embeddings0 @ best_proj

            # center and normalize
            # projected_embs0 -= projected_embs0.mean()
            # projected_embs0 *= torch.norm(circle) / torch.norm(projected_embs0)

            best_perm, _ = utils.find_perm(target_shape, projected_embs0)
            embeddings0 = best_perm @ embeddings0

            total_perm = best_perm @ total_perm

            best_proj = torch.linalg.lstsq(embeddings0, target_shape).solution
            # utils.find_best_proj(target_shape, embeddings0)

            if torch.allclose(curr_best_proj, best_proj):
                break
                
            curr_best_proj = best_proj.clone()

        projected_embs0 = embeddings0 @ best_proj

        # center and normalize
        projected_embs0 -= projected_embs0.mean(0)
        projected_embs0 *= torch.norm(target_shape) / torch.norm(projected_embs0)

        projected_embs_across_seeds.append(projected_embs0)
        scores.append(torch.norm(projected_embs0 - target_shape) / torch.norm(target_shape))
        projection_mats.append(curr_best_proj)
        total_perms.append(total_perm)
    
    return (projected_embs_across_seeds[np.argmin(scores)],
            projection_mats[np.argmin(scores)],
            total_perms[np.argmin(scores)],
            np.min(scores))