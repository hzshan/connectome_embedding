import numpy as np
import torch
import analysis, data_utils, pandas, tqdm
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

class ConnectivityData:
    def __init__(self, J, neurons:pandas.DataFrame, type_key:str):
        """
        Data object of a connectome dataset.

        Notations: N is the number of neurons, Ntype is the number of cell types.

        Args:
            J: NxN torch tensor, the connectivity matrix
            onehot_types: NxNtype torch tensor, the one-hot encoding of neuron types
            type_hash: dict, the mapping from type string to type index
            types: list of str, the list of unique types
            neuron_hash: dict, the mapping from type index to neuron indices
            N_per_type: list of int, the number of neurons of each type
            neurons: pandas DataFrame, the information about each neuron
        """
        self.J = J
        self.N = J.shape[0]

        onehot_types, Ntype, typehash, uniqtypes, neuron_hash = \
            data_utils.make_onehot_type_vectors(neurons, type_key)
        
        self.onehot_types = onehot_types
        self.Ntype = Ntype
        self.type_hash = typehash
        self.types = uniqtypes
        self.neuron_hash = neuron_hash
        self.N_per_type = onehot_types.sum(0).numpy().astype(np.int16)
        self.neurons = neurons

        assert self.onehot_types.shape == (self.N, self.Ntype)
        assert len(self.types) == self.Ntype




def get_squared_dist(source_emb, target_emb):
    """
    Assuming a NxD shape, return a NxN matrix of squared distances."""
    source_sum_sq = torch.sum(source_emb * source_emb, 1, keepdim=True)
    target_sum_sq = torch.sum(target_emb * target_emb, 1, keepdim=True)

    return source_sum_sq + target_sum_sq.T - 2 * (source_emb @ target_emb.T)


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
        angles = torch.linspace(0, 2 * np.pi, n_points)
    unit_circle = torch.stack([torch.cos(angles), torch.sin(angles)], 1)

    return unit_circle


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


def sort_embeddings(embeddings, data: ConnectivityData,
                    full_method=False):
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

    _, _, V = torch.svd(embeddings)
    projected_all_embeddings = embeddings @ V

    sorted_inds = []

    for i in tqdm.trange(data.Ntype):
        start_ind = int(data.N_per_type[:i].sum())
        end_ind = int(data.N_per_type[:i+1].sum())

        if full_method:
            # solve shadowmatic problem for embeddings of each cell type
            _, _, best_perm = analysis.solve_shadowmatic2(
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
