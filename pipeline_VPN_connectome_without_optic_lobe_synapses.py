"""
This file builds the connectivity matrix of VPN neurons (defined as LC*, MC* and
LPLC*) excluding synapses in the optic lobe.

It first generates a list of bodyIDs of VPN neurons from the hemibrain dataset,
then loads pre-saved hashmaps and find synapse_sets that connect two VPN neurons.
Finally, it counts the number of synapses in each synapse_set that are outside the
optic lobe, and builds the connectivity matrix accordingly.
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle, tqdm

from connectome_embedding import data_utils

types_wanted = ['LC*', 'MC*', 'LPLC*']
raw_data_path = 'data_hemibrain/hemibrain_data_min_10_cells_per_type.pkl'
destination_for_full_matrix = 'data_hemibrain/J_visual_neurons_all_synapses.pkl'
destination_for_matrix_without_optic_lobe_synapses = 'data_hemibrain/J_visual_neurons_outside_optic_lobe.pkl'

#########################

neuronsall, W_all = pickle.load(open(raw_data_path, 'rb'))

# Pick out only neurons of certain types for further analysis.
data = data_utils.prep_connectivity_data(
    W_all, neuronsall, types_wanted, split_LR=[])

# path where the hashmaps are stored
PATH = 'data_hemibrain/visual neuron morphology/'

# load hashmap from neuron bodyId to synapse set
neuron_to_synapseset = pickle.load(open(
    PATH + 'neuron_to_synapse_set_visual.pkl', 'rb'))

# load hashmap from synapse set to synapse
synapse_set_to_synapse = pickle.load(open(
    PATH + 'synapse_set_to_synapse_visual.pkl', 'rb'))

# load hashmap from synapse to coordinates
synapse_to_coordinates_full = pickle.load(open(
    PATH + 'synapse_to_coordinates_visual_neurons.pkl', 'rb'))
synapse_to_coordinates = pickle.load(open(
    PATH + 'synapse_to_coordinates_visual_neurons_in_optic_lobe.pkl', 'rb'))

# filter out synapse sets that connect two visual neurons
synapse_sets_between_visual_neurons = {}
for k in synapse_set_to_synapse.keys():
    bodyId1 = int(k.split('_')[0])
    bodyId2 = int(k.split('_')[1])
    if bodyId1 in data.neurons['bodyId'].values and bodyId2 in data.neurons['bodyId'].values:
        synapse_sets_between_visual_neurons[k] = synapse_set_to_synapse[k]

# create lists of synapses in and not in the optic lobe for quick lookup
list_of_all_synapses = list(synapse_to_coordinates_full.keys())
list_of_all_synapses_in_optic_lobe = list(synapse_to_coordinates.keys())
list_of_all_synapses_not_in_optic_lobe = list(
    set(list_of_all_synapses) - set(list_of_all_synapses_in_optic_lobe)
)

J_with_all_synapses = np.zeros((len(data.neurons), len(data.neurons)))
J_without_optic_lobe_synapses = np.zeros((len(data.neurons), len(data.neurons)))

for k in tqdm.tqdm(synapse_sets_between_visual_neurons.keys()):
    bodyId1, bodyId2, affix = k.split('_')
    if affix == 'post':
        continue

    i = data.neurons.index[data.neurons['bodyId'] == int(bodyId1)][0]
    j = data.neurons.index[data.neurons['bodyId'] == int(bodyId2)][0]
    synapse_inds = synapse_sets_between_visual_neurons[k]

    J_with_all_synapses[i, j] = len(synapse_inds)
    
    count = 0
    for syn in synapse_inds:
        if syn in list_of_all_synapses_not_in_optic_lobe:
            count += 1
    J_without_optic_lobe_synapses[i, j] = count

pickle.dump(J_with_all_synapses,
            open(destination_for_full_matrix, 'wb'))

pickle.dump(J_without_optic_lobe_synapses,
            open(destination_for_matrix_without_optic_lobe_synapses, 'wb'))