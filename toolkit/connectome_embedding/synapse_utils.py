"""
Utility functions for working with hemibrain on the synapse level.
"""
import numpy as np

def get_synapse_coordinates(bodyIds: list, 
                           neuron_to_synapseset: dict,
                           synapse_set_to_synapse: dict,
                           synapse_to_coordinates: dict,
                           separate_pre_post: bool = False):
    """
    Get 3D spatial coordinates for all synapses of each neuron in bodyIDs.
    If separate_pre_post is True, returns a dictionary mapping each bodyId to a
    dictionary with keys 'pre' and 'post', each containing a list of 3D
    coordinates of pre- and post-synaptic sites.
    Else, combines pre- and post-synaptic coordinates into a single list

    Args:
        bodyIds (list): List of neuron body IDs.
        neuron_to_synapseset (dict): Mapping from neuron body ID to synapse set IDs.
        synapse_set_to_synapse (dict): Mapping from synapse set ID to synapse IDs.
        synapse_to_coordinates (dict): Mapping from synapse ID to 3D coordinates.
        separate_pre_post (bool): Whether to keep pre- and post-synaptic coordinates separate.
    """
    target_coordinates = {}
    for bodyId in bodyIds:
        synapsesets = neuron_to_synapseset[str(bodyId)]

        # sort synapsesets into pre and post
        pre_synapse_sets = [s for s in synapsesets if 'pre' in s]
        post_synapse_sets = [s for s in synapsesets if 'post' in s]
        target_coordinates[str(bodyId)] = {'pre':[], 'post':[]}

        for synapseset in pre_synapse_sets:

            for synapse in synapse_set_to_synapse[synapseset]:
                if synapse not in synapse_to_coordinates:
                    continue
                coord = np.array(synapse_to_coordinates[synapse]) * 8 / 1000  # convert to microns
                target_coordinates[str(bodyId)]['pre'].append(coord)

        for synapseset in post_synapse_sets:

            for synapse in synapse_set_to_synapse[synapseset]:
                if synapse not in synapse_to_coordinates:
                    continue
                coord = np.array(synapse_to_coordinates[synapse]) * 8 / 1000  # convert to microns
                target_coordinates[str(bodyId)]['post'].append(coord)

    if separate_pre_post:
        return target_coordinates
    else:
        for bodyId in target_coordinates.keys():
            target_coordinates[bodyId] = combine_pre_and_post(
                np.array(target_coordinates[bodyId]['pre']),
                np.array(target_coordinates[bodyId]['post'])
            )
        return target_coordinates


def combine_pre_and_post(pre_array, post_array):
    if len(pre_array) == 0 and len(post_array) == 0:
        return np.array([])
    elif len(pre_array) == 0:
        return post_array
    elif len(post_array) == 0:
        return pre_array
    else:
        return np.vstack((pre_array, post_array))