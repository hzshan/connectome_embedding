"""
Helpful plots to make.
"""

import numpy as np
import matplotlib.pyplot as plt
import data_utils, torch, utils, models
from matplotlib.colors import LogNorm


def plot_J_some_types(J: torch.Tensor,
                      types: list,
                      data:utils.ConnectivityData,
                      ax=None):

    if ax is None:
        plt.figure()
    else:
        plt.sca(ax)

    all_neuron_inds = np.concatenate([data.neuron_hash[t] for t in types])
    # get a sorting of cells from this type
    all_type_inds = np.array([data.type_hash[t] for t in types])
    all_onehots = data.onehot_types[all_neuron_inds]
    all_onehots = all_onehots[:, all_type_inds]

    J_submatrix = J[all_neuron_inds][:, all_neuron_inds]

    plt.imshow(np.log(1 + J_submatrix), cmap='gray_r', interpolation='none')
    plt.xticks(*utils.tick_maker(types, all_onehots))
    plt.yticks(*utils.tick_maker(types, all_onehots))

    return J_submatrix



def stacked_bar_plot(labels: list,
                     bar_heights,
                     legends=None, ax=None, colors=None):
    """
    Make a stacked bar plot with len(labels) columns. Each column is a stack of
    bar_heights.shape[1] bars. legends are the labels for each bar in every
    column.
    """
    n_columns, n_bars_per_column = bar_heights.shape
    assert len(labels) == n_columns

    if legends is not None:
        assert len(legends) == n_bars_per_column
    
    if ax is None:
        ax = plt.gca()
    if colors is None:
        colors = plt.cm.coolwarm(np.linspace(0, 1, bar_heights.shape[1]))

    left = np.zeros(bar_heights.shape[0])

    for i in range(bar_heights.shape[1]):
        if legends is not None:
            ax.bar(labels, bar_heights[:, i], bottom=left, color=colors[i],
                   label=legends[i])
        else:
            ax.bar(labels, bar_heights[:, i], bottom=left, color=colors[i])
        left += bar_heights[:, i]

    # rotate xlabels
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    plt.legend()
    
    return ax


def summarize_adjacency_for_type(inquiry_type: str,
                                 data:utils.ConnectivityData,
                                 cutoff=0,
                                 ax=None):
    """
    For cells of the given type, plot the number of synapses in and out of each other type.

    The "synapses per neuron" number for type A cells to type B cells is
    defined as the total number of synapses from A cells to B cells, divided by the number of A cells.

    Args:
        inquiry_type: the type of cells to summarize the connectivity for
        data: the ConnectivityData object
        cutoff: only show connections with more than this number of synapses per neuron
        ax: the axis to plot on
    """
    connected_types, in_out_counts = utils.get_adjacency_of_type(inquiry_type, data, cutoff=cutoff)

    if ax is None:
        plt.figure(figsize=(5, 2), dpi=200)
    else:
        plt.sca(ax)
    stacked_bar_plot(connected_types, in_out_counts, colors=None, legends=['in', 'out'])
    plt.ylabel('synapses per neuron')
    plt.title('adjacency summary for ' + inquiry_type)

    return connected_types, in_out_counts


def summarize_embeddings_for_types(list_of_types, model, typehash):
    _, _, V = torch.svd(model.embeddings.detach())

    sq_norm_on_top_PCs = []

    if list_of_types is None:
        list_of_types = model.type_names

    for type in list_of_types:
        type_ind = typehash[type]
        x = model.embeddings.detach()[model.onehot_types[:, type_ind].bool()].detach()
        sq_norm_on_top_PCs.append(torch.norm(x @ V, dim=0)**2 / torch.norm(x)**2)

    sq_norm_on_top_PCs = np.array(sq_norm_on_top_PCs)
    for type in list_of_types:
        ind = typehash[type]
        print(type)
        print(model.embeddings[model.onehot_types[:, ind].bool()].detach().numpy())

    sq_norm_on_top_PCs = np.array(sq_norm_on_top_PCs)

    plt.figure(figsize=(20, 3))
    _ = stacked_bar_plot(sq_norm_on_top_PCs, list_of_types, colors=None)
    _ = plt.xticks(rotation=45)


def plot_J(J, data: utils.ConnectivityData, types: list=None, N_per_type=None, ax=None, type_borders=True):
    if ax is None:
        ax = plt.gca()

    plt.sca(ax)
    plt.imshow(np.log(1 + J), cmap='gray_r', interpolation='none')
    tick_inds = []
    tick_labels = []

    if types is None:
        types = data.types
    if N_per_type is None:
        N_per_type = data.N_per_type
    if len(types) != len(N_per_type):
        raise ValueError('types and N_per_type must have the same length')

    for i, ut in enumerate(types):
        tick_inds.append(np.sum(N_per_type[:i]) + N_per_type[i] / 2)
        tick_labels.append(ut)

    if type_borders:
        sum_till_now = 0
        for i in range(0, len(N_per_type) - 1):
            sum_till_now += N_per_type[i]
            plt.axvline(sum_till_now - 0.5, color='k', lw=0.1, alpha=0.5)
            plt.axhline(sum_till_now - 0.5, color='k', lw=0.1, alpha=0.5)
    plt.xticks(tick_inds, tick_labels, rotation=90, fontsize=7)
    plt.yticks(tick_inds, tick_labels, rotation=0, fontsize=7)
    return tick_inds, tick_labels


def plot_embeddings_in_2d(cell_types, data, model, ax=None):
    if ax is None:
        plt.figure()
    else:
        plt.sca(ax)
    for i, t in enumerate(cell_types):
        embs = model.embeddings[data.neuron_hash[t]]
        embs_2d, _, _ = utils.solve_shadowmatic2(embs, nseeds_to_try=50)
        
        plt.scatter(embs_2d[:, 0], embs_2d[:, 1], label=t)

    unit_circle = utils.make_circle(100, wrap=True)
    plt.plot(unit_circle[:, 0], unit_circle[:, 1], 'k')
    plt.gca().set_aspect('equal', adjustable='box')


def plot_J_between_types(J, data, typeA:str, typeB:str, log_J=False):
    """
    Plot the connectivity matrix between two types of neurons.
    """
    A2B = J[data.neuron_hash[typeA]][:, data.neuron_hash[typeB]]
    B2A = J[data.neuron_hash[typeB]][:, data.neuron_hash[typeA]]

    if log_J:
        A2B = np.log(1 + A2B)
        B2A = np.log(1 + B2A)

    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    plt.sca(axes[0])
    plt.imshow(np.log(1 + A2B), cmap='gray_r')
    plt.title(f'{typeA} to {typeB}')
    plt.xlabel(f'{typeB} neuron idx')
    plt.ylabel(f'{typeA} neuron idx')
    plt.colorbar()
    plt.sca(axes[1])
    plt.imshow(np.log(1 + B2A), cmap='gray_r')
    plt.title(f'{typeB} to {typeA}')
    plt.xlabel(f'{typeA} neuron idx')
    plt.ylabel(f'{typeB} neuron idx')
    plt.colorbar()
    return A2B, B2A