"""
Per-cell-type scores of the shape of an embedding, for asking which types form a ring.

  types_to_score      cell types with at least a given number of neurons
  circularity_scores  best 2D projection of a type's embeddings onto a circle
                      (utils.solve_shadowmatic2) and its circularity score
  h1_scores           longest H1 (loop) lifetime from persistent homology, optionally
                      against Gaussian clouds with the same size and PCA spectrum
  dimension_scores    intrinsic dimension (Gride at one neighbour scale)

Each scoring function takes a numpy array of embeddings (N x D), a ConnectivityData
(data.neuron_hash) and the cell types to score; row i of the embeddings is neuron i
of data. Agreement with anatomy is anatomy.mean_tau_per_type, and the cell-type
groups are anatomy.type_groups.
"""
import numpy as np
import pandas as pd
import torch

from . import utils, topology


def types_to_score(data, min_neurons):
    """Cell types of data with at least min_neurons neurons."""
    return [t for t in data.types if len(data.neuron_hash[t]) >= min_neurons]


def circularity_scores(embeddings, data, types, circle_search):
    """
    Circularity score of each cell type: the type's embeddings are projected to 2D
    with the circle search (utils.solve_shadowmatic2, settings circle_search), and the
    projection is compared with a circle (utils.circ_score_via_norm; 1 = a circle).

    Returns:
        DataFrame with columns type, circularity.
    """
    rows = []
    for cell_type in types:
        x = torch.as_tensor(embeddings[data.neuron_hash[cell_type]], dtype=torch.float32)
        found_circle, _, _ = utils.solve_shadowmatic2(x, **circle_search)
        rows.append({'type': cell_type,
                     'circularity': float(utils.circ_score_via_norm(found_circle, skip_alignment=True))})
    return pd.DataFrame(rows)


def h1_scores(embeddings, data, types, rng, max_points=300, n_null=0):
    """
    Longest H1 lifetime of each cell type's embeddings.

    Each type is subsampled to at most max_points neurons and scaled to unit RMS
    radius (topology.normalize), so lifetimes are comparable across types.

    Returns:
        DataFrame with columns type, h1_1st, and, if n_null > 0, the same for n_null
        Gaussian clouds with the type's size and PCA spectrum: h1_null_median,
        h1_null_95 (95th percentile) and h1_relative = h1_1st / h1_null_95
        (> 1: above the null).
    """
    rows = []
    for cell_type in types:
        x = topology.normalize(topology.subsample(embeddings[data.neuron_hash[cell_type]],
                                                  max_points, rng))
        life, _ = topology.lifetimes(x, maxdim=1)
        row = {'type': cell_type, 'h1_1st': topology.kth_longest(life, 1, 0)}
        if n_null > 0:
            null_h1 = topology.null_lifetimes(x, n_null, rng, features=[(1, 0)], maxdim=1)[:, 0]
            row.update({'h1_null_median': np.median(null_h1),
                        'h1_null_95': np.percentile(null_h1, 95),
                        'h1_relative': row['h1_1st'] / np.percentile(null_h1, 95)})
        rows.append(row)
    return pd.DataFrame(rows)


def dimension_scores(embeddings, data, types, gride_n1):
    """
    Intrinsic dimension of each cell type's embeddings (all neurons of the type).

    Returns:
        DataFrame with columns type and gride_dim (topology.gride_dimension at
        neighbour scale gride_n1).
    """
    rows = []
    for cell_type in types:
        x = embeddings[data.neuron_hash[cell_type]]
        rows.append({'type': cell_type,
                     'gride_dim': topology.gride_dimension(x, gride_n1)})
    return pd.DataFrame(rows)
