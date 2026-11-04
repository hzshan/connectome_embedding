"""
Anatomical labels of hemibrain central complex neurons, and comparison of those
labels with angles read off an embedding.

  anatomical_labels   PB glomerulus, FB column, soma side from instance names
  add_fb_angles       FB column -> angle, columns of a type tiling the circle
  pca_angles          angle of the embeddings projected onto a reference type's plane
                      (or the top two PCs of all embeddings)
                      type (or of all embeddings)
  alignment           circular alignment R between two sets of angles, allowing a
                      global rotation, a reflection and a winding factor
  score_groups        R per cell type (and PB side) with a shuffled-label null
"""
import re
import numpy as np
import pandas as pd

from .utils import projection_directions


# A single PB glomerulus: "_L4" or "_R7", followed by "_" or the end of the name.
# Names with several glomeruli ("_L4R5", "_L2/4") deliberately do not match.
PB_GLOMERULUS = re.compile(r'_([LR])(\d)(?=_|$)')

# An FB column: "_C3", optionally with a letter for sub-columns ("_C6a"), followed
# by "_" or the end of the name. The sub-column letter is ignored.
FB_COLUMN = re.compile(r'_C(\d+)[a-z]?(?=_|$)')

# Soma side written as a bare "_L" / "_R" (e.g. FC1A(FQ6)_C3_L)
SOMA_SIDE = re.compile(r'_([LR])(?=_|$)')


def parse_hemibrain_instance(instance):
    """
    Read the anatomical labels encoded in one hemibrain instance name.

    Returns a dict with:
        pb_side:        'L' or 'R', side of the PB glomerulus (None if not given)
        pb_glomerulus:  glomerulus number 1..9 (NaN if not given or ambiguous)
        fb_column:      FB column number (NaN if not given)
        soma_side:      'L' or 'R' when written as a bare suffix (None otherwise)
        irregular:      True if the name is flagged "_irreg"
    """
    instance = str(instance)

    # PB glomerulus; only use it if exactly one single-glomerulus label is present
    pb_matches = PB_GLOMERULUS.findall(instance)
    if len(pb_matches) == 1:
        pb_side, pb_glomerulus = pb_matches[0][0], int(pb_matches[0][1])
    else:
        pb_side, pb_glomerulus = None, np.nan

    # FB column; again only if exactly one is present
    fb_matches = FB_COLUMN.findall(instance)
    fb_column = int(fb_matches[0]) if len(fb_matches) == 1 else np.nan

    # soma side written as a bare suffix
    side_matches = SOMA_SIDE.findall(instance)
    soma_side = side_matches[-1] if len(side_matches) > 0 else None

    return {
        'pb_side': pb_side,
        'pb_glomerulus': pb_glomerulus,
        'fb_column': fb_column,
        'soma_side': soma_side,
        'irregular': '_irreg' in instance,
    }


def anatomical_labels(neurons, pb_period=8):
    """
    Anatomical labels for every neuron in data.neurons, in the same row order
    (so row i matches model.embeddings[i]).

    Adds pb_angle_deg: the angle implied by the PB glomerulus, within its side,
    (glomerulus - 1) mod pb_period * 360 / pb_period.
    """
    parsed = pd.DataFrame([parse_hemibrain_instance(name) for name in neurons['instance']])

    labels = pd.DataFrame({
        'row': np.arange(len(neurons)),            # row in model.embeddings
        'bodyId': neurons['bodyId'].values,
        'type': neurons['type'].values,           # type as used in the embedding (may have _L/_R)
        'instance': neurons['instance'].values,
    })
    labels = pd.concat([labels, parsed], axis=1)

    # angle from the PB glomerulus, separately for each side of the PB
    labels['pb_angle_deg'] = ((labels['pb_glomerulus'] - 1) % pb_period) * 360. / pb_period

    return labels


def summarize_by_type(labels):
    """How many neurons of each type have each kind of label."""
    summary = labels.groupby('type').agg(
        n_neurons=('row', 'size'),
        n_pb_glomerulus=('pb_glomerulus', lambda x: x.notna().sum()),
        n_fb_column=('fb_column', lambda x: x.notna().sum()),
        pb_sides=('pb_side', lambda x: ''.join(sorted(set(x.dropna())))),
        n_irregular=('irregular', 'sum'),
    )
    return summary


def add_fb_angles(labels):
    """FB column -> angle, with the columns of each cell type tiling the circle evenly."""
    n_columns = labels.groupby('type')['fb_column'].transform('max')
    labels['fb_angle_deg'] = 360. * (labels['fb_column'] - 1) / n_columns
    return labels


def pca_angles(embeddings, data, reference_type=None):
    """
    Angle of each neuron: project the embeddings onto a reference-optimized plane.

    The directions come from utils.projection_directions: solve_shadowmatic2 on the
    embeddings of reference_type (e.g. 'EPG'), or the top two PCs of all embeddings
    if reference_type is None. Same arguments as utils.sort_embeddings.
    """
    reference_inds = None if reference_type is None else data.neuron_hash[reference_type]
    projection = projection_directions(embeddings, reference_inds)
    projected = (embeddings @ projection).detach().numpy()
    return np.arctan2(projected[:, 1], projected[:, 0])


def alignment(theta_embedding, theta_anatomy, windings=(1, 0.5, 2)):
    """
    Compare two sets of angles (radians), allowing one global rotation, one
    reflection, and a winding factor k from windings on the anatomical angles.

    Returns:
        R: circular alignment in [0, 1] (1 = same order up to rotation/reflection)
        reflected: True if the best match flips the direction of the embedding angles
        winding: the k of the best match
        rotation: the best global rotation (radians)
        median_error_deg: median absolute angle difference after the best match
    """
    best = None
    for k in windings:
        for sign in [1, -1]:
            # differences between (possibly reflected) embedding angles and scaled
            # anatomy; if the orders agree, these all equal the global rotation
            diff = sign * theta_embedding - k * theta_anatomy
            mean_vector = np.mean(np.exp(1j * diff))
            R = np.abs(mean_vector)
            if best is None or R > best[0]:
                rotation = np.angle(mean_vector)
                # remaining error per neuron, wrapped to (-180, 180]
                error = np.angle(np.exp(1j * (diff - rotation)))
                best = (R, sign == -1, k, rotation, np.degrees(np.median(np.abs(error))))
    return best


def null_alignment(theta_embedding, theta_anatomy, n_shuffles, rng, windings=(1, 0.5, 2)):
    """R for each of n_shuffles shuffles of the anatomical labels across neurons."""
    return np.array([alignment(theta_embedding, rng.permutation(theta_anatomy), windings)[0]
                     for _ in range(n_shuffles)])


def score_groups(labels, angle_column, rng, windings=(1, 0.5, 2),
                 min_neurons=4, min_positions=3, n_shuffles=1000):
    """
    Score every group of neurons with anatomical labels:
      - PB glomerulus: one group per cell type and PB side
      - FB column:     one group per cell type
    Returns one row per group. A label kind is skipped if its anatomy column is
    not in labels (e.g. no fb_angle_deg when add_fb_angles was not called).
    """
    rows = []
    for label_kind, anatomy_column, group_keys in [
            ('PB glomerulus', 'pb_angle_deg', ['type', 'pb_side']),
            ('FB column', 'fb_angle_deg', ['type'])]:
        if anatomy_column not in labels.columns:
            continue

        labelled = labels[labels[anatomy_column].notna() & labels[angle_column].notna()]
        for keys, group in labelled.groupby(group_keys):
            keys = keys if isinstance(keys, tuple) else (keys,)
            n_positions = group[anatomy_column].nunique()
            if len(group) < min_neurons or n_positions < min_positions:
                continue

            theta_embedding = group[angle_column].values
            theta_anatomy = np.radians(group[anatomy_column].values)
            R, reflected, winding, _, median_error = alignment(theta_embedding, theta_anatomy, windings)

            null = null_alignment(theta_embedding, theta_anatomy, n_shuffles, rng, windings)
            rows.append({
                'type': keys[0],
                'pb_side': keys[1] if len(keys) > 1 else None,
                'label': label_kind,
                'n_neurons': len(group),
                'n_positions': n_positions,
                'R': R,
                'R_null_median': np.median(null),
                'R_null_lo': np.percentile(null, 2.5),
                'R_null_hi': np.percentile(null, 97.5),
                'median_error_deg': median_error,
                'reflected': reflected,
                'winding': winding,
            })
    return pd.DataFrame(rows)
