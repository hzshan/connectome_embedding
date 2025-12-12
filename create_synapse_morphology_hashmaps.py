""" Some info of hemibrain
* num of synapses: 73717518
* num of synapse_set: 88603904
* neuron to synapse_set: 88603904
* synapse_set to synapse: 128441270

Description:
This notebook contains code that converts raw hemibrain data
(https://storage.cloud.google.com/hemibrain-release/neuprint/hemibrain_v1.0.1_neo4j_inputs.zip))
into Python dictionaries that can be used as hashmaps for efficient lookups.

Since I was primarily interested in visual projection neurons,
it also contains code for creating dictionaries of subsets of data.
"""


import pandas as pd
import tqdm
import pickle, os

RAW_DATA_PATH = os.path.expanduser('~/Downloads/hemibrain_v1.0.1_neo4j_inputs')
PROCESSED_DATA_PATH = os.path.expanduser(
    '~/Dropbox/Codes/connectome_embedding/data_hemibrain/visual neuron morphology')

"""# create a dictionary where the key is the synapse index and the value
is a tuple of its x,y,z coordinates"""

OPTIC_LOBE_ONLY = True  # Set to True to filter for optic lobe synapses only

synapse_coordinates = open(os.path.join(RAW_DATA_PATH,
                                        'Neuprint_Synapses_52a133.csv'), 'r')

location_dict = {}

for line in tqdm.tqdm(synapse_coordinates):

    # Split the line by commas
    row = line.strip().split(',')

    if row[0] == '":ID(Syn-ID)"':
        # skip header line
        continue

    if OPTIC_LOBE_ONLY and row[7] == '':
        # skip synapses not in optic lobe.
        # row[7] encodes whether the synapse is in the optic lobe.
        continue

    synapse_ind = row[0]
    synapse_x = int(row[3].split(':')[1])  # x coordinate
    synapse_y = int(row[4][3:])  # y coordinate
    synapse_z = int(row[5][3:].split('}')[0])  # z coordinate
    synapse_loc = (synapse_x, synapse_y, synapse_z)
    
    location_dict[synapse_ind] = synapse_loc


pickle.dump(location_dict, open(os.path.join(
    PROCESSED_DATA_PATH,
    'synapse_to_coordinates_optic_lobe.pkl' if OPTIC_LOBE_ONLY\
          else 'synapse_to_coordinates.pkl'), 'wb'))

################################################################################
"""create a dictionary where the key is the synapseset and the value is
the list of synapse indices"""

synapse_sets = open(os.path.join(
    RAW_DATA_PATH, 'Neuprint_SynapseSet_to_Synapses_52a133.csv'), 'r')

synapse_set_dict = {}

for line in tqdm.tqdm(synapse_sets):

    # Split the line by commas
    row = line.strip().split(',')

    if row[0] == ':START_ID':
        # skip header line
        continue
    synapse_set_name = row[0]
    synapse_ind = row[1]

    if synapse_set_name not in synapse_set_dict:
        synapse_set_dict[synapse_set_name] = []
    synapse_set_dict[synapse_set_name].append(synapse_ind)

# # Save the dictionary to a pickle file
synapse_sets = None  # release memory
pickle.dump(synapse_set_dict,
            open(os.path.join(
                PROCESSED_DATA_PATH, 'synapse_set_to_synapse.pkl'), 'wb'))

################################################################################
"""# create a dictionary where the key is the neuron index and the value
is the list of synapsesets"""

neuron_to_synapseset = open(os.path.join(
    RAW_DATA_PATH, 'Neuprint_Neuron_to_SynapseSet_52a133.csv'), 'r')

neuron_dict = {}

for line in tqdm.tqdm(neuron_to_synapseset):

    # Split the line by commas
    row = line.strip().split(',')

    if row[0] == ':START_ID(Body-ID)':
        # skip header line
        continue
    neuron_ind = row[0]
    synapse_set_name = row[1]

    if neuron_ind not in neuron_dict:
        neuron_dict[neuron_ind] = []
    neuron_dict[neuron_ind].append(synapse_set_name)

# # Save the dictionary to a pickle file
neuron_to_synapseset = None  # release memory
pickle.dump(neuron_dict, open(
    os.path.join(
        PROCESSED_DATA_PATH, 'neuron_to_synapse_set.pkl'), 'wb'))

################################################################################
"""# Generate smaller files with only some types
In this case, only LC, MC LPLC, LPC and LLPC neurons."""

neuronsall = pd.read_csv('data_hemibrain/' + "traced-neurons.csv")
neuronsall = neuronsall.dropna(subset=['type'])

LC_neurons = neuronsall['type'].str.startswith('LC').dropna()
LPLC_neurons = neuronsall['type'].str.startswith('LPLC').dropna()
LPC_neurons = neuronsall['type'].str.startswith('LPC').dropna()
LLPC_neurons = neuronsall['type'].str.startswith('LLPC').dropna()

# also include neurons with type that start with 'MC'
MC_neurons = neuronsall['type'].str.startswith('MC').dropna()

# combine LC and MC neurons
visual_neurons = LC_neurons | MC_neurons
visual_neurons = visual_neurons | LPLC_neurons | LPC_neurons | LLPC_neurons

# get bodyIds of neurons with type starting with LC
neuron_inds = neuronsall[visual_neurons]['bodyId'].astype(str).tolist()

"""
First, generate a neuron_to_synapseset dictionary with only
the neurons of interest.
"""

neuron_to_synapseset = pickle.load(open(os.path.join(
    PROCESSED_DATA_PATH, 'neuron_to_synapse_set.pkl'), 'rb'))

neuron_to_synapseset = {k: neuron_to_synapseset[k] for k in neuron_inds}

pickle.dump(neuron_to_synapseset,
            open(os.path.join(PROCESSED_DATA_PATH,
                              'neuron_to_synapse_set_visual.pkl'), 'wb'))

"""
Next, generate a synapseset_to_synapse dictionary with only the synapse sets
 in neurons of interest.
 """

all_visual_synapse_sets = []
for synset_list in neuron_to_synapseset.values():
    all_visual_synapse_sets.extend(synset_list)
all_visual_synapse_sets = list(set(all_visual_synapse_sets))

synapse_set_to_synapse = pickle.load(open(os.path.join(
    PROCESSED_DATA_PATH, 'synapse_set_to_synapse.pkl'), 'rb'))

all_visual_synapses = {}
for synset in all_visual_synapse_sets:
    all_visual_synapses[synset] = synapse_set_to_synapse[synset]

pickle.dump(all_visual_synapses, open(os.path.join(
    PROCESSED_DATA_PATH,
    'synapse_set_to_synapse_visual.pkl'), 'wb'))

"""
Finally, a synapse_to_synapse coordinate dictionary with only the relevant synapses.
Optionally, only include synapses in the optic lobe.
"""


OPTIC_LOBE_ONLY = True  # Set to True to filter for optic lobe synapses only
synapse_set_to_synapse_visual = pickle.load(
    open(os.path.join(PROCESSED_DATA_PATH,
                      'synapse_set_to_synapse_visual.pkl'), 'rb'))

list_of_synapses = []
for synset_list in synapse_set_to_synapse_visual.values():
    list_of_synapses.extend(synset_list)

if OPTIC_LOBE_ONLY:
    synapse_to_coordinates = pickle.load(open(os.path.join(
    PROCESSED_DATA_PATH,
    'synapse_to_coordinates_optic_lobe.pkl'), 'rb'))
else:
    synapse_to_coordinates = pickle.load(open(os.path.join(
    PROCESSED_DATA_PATH,
    'synapse_to_coordinates.pkl'), 'rb'))

synapse_to_coordinates_visual = {
    k: synapse_to_coordinates[k] for k in list_of_synapses if k in synapse_to_coordinates}

pickle.dump(synapse_to_coordinates_visual, open(os.path.join(
    PROCESSED_DATA_PATH,
    'synapse_to_coordinates_visual_neurons_in_optic_lobe.pkl'), 'wb'))
