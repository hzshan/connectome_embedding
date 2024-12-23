import numpy as np
import matplotlib.pyplot as plt
import models, torch, utils

def stacked_bar_plot(labels, data, ax=None, colors=None):
    assert len(labels) == data.shape[0]
    
    if ax is None:
        ax = plt.gca()
    if colors is None:
        colors = plt.cm.coolwarm(np.linspace(0, 1, data.shape[1]))

    left = np.zeros(data.shape[0])
    for i in range(data.shape[1]):
        ax.bar(labels, data[:, i], bottom=left, color=colors[i])
        left += data[:, i]

    # rotate xlabels
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    
    return ax


def make_circle_plot(ax, colors=None):
    ax.set_aspect('equal')
    angles = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(angles), np.sin(angles), color='black', linewidth=0.5, alpha=0.5)


def summarize_adjacency_for_type(inquiry_type,
                                 data: utils.ConnectivityData,
                                 cutoff=5):
    assert inquiry_type in data.types
    ind = data.type_hash[inquiry_type]

    num_synapses_bt_cell_types = (
        data.onehot_types.T @ data.J @ data.onehot_types)

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
    data = torch.stack([in_synapses[sort_ind], out_synapses[sort_ind]], dim=1).numpy()

    plt.figure(figsize=(5, 2), dpi=200)
    stacked_bar_plot(labels, data, colors=None)
    plt.ylabel('synapses per neuron')
    plt.title('adjacency summary for ' + inquiry_type)

    return labels