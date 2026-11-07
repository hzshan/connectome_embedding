"""
Anatomical labels of hemibrain central complex neurons, and comparison of those
labels with angles read off an embedding.

  anatomical_labels   PB glomerulus, FB column, soma side from instance names
  add_fb_angles       FB column -> angle, columns of a type tiling the circle
  pca_angles          angle of the embeddings projected onto the top two PCs of a
                      reference type (or of all embeddings)
  order_tau           Kendall's tau between embedding and anatomical order, allowing a
                      cut and direction of the circle and a winding factor
  score_groups        tau per cell type (and PB side) with a shuffled-label null
  mean_tau_per_type   tau per cell type, averaged over its PB sides and FB columns
  type_groups         CX columnar / CX other / ring / TuBu / other, per cell type
"""
import re
import numpy as np
import pandas as pd

from .utils import projection_directions
from .data_utils import types_matching


# Cell-type groups, as patterns with the matching rules of types_wanted in
# embed_hemibrain.py ('=' exact match, '*' prefix, otherwise substring).
# Central complex types (same list as the 'central_complex' embedding config)
CX_TYPES = ["EPG=", 'Delta7', 'PEG', 'EPGt', 'PEN', 'FC', 'PFL', 'PFN',
            'FB', 'hDelta', 'vDelta', 'FR*', 'EL', 'FS', 'PFG', 'PFR', 'IbSpsP']
# CX types that tile the PB or FB (columnar), where a ring is expected
CX_COLUMNAR_TYPES = ["EPG=", 'Delta7', 'PEG', 'EPGt', 'PEN', 'FC', 'PFL', 'PFN',
                     'hDelta', 'vDelta', 'FR*', 'FS', 'PFG', 'PFR', 'IbSpsP']
# ring neurons (EB tangential neurons that synapse onto EPG) and the TuBu neurons
# upstream of them; not columnar
RING_AND_TUBU_TYPES = ['ER*', 'TuBu*']
GROUPS = ['CX columnar', 'CX other', 'ring / TuBu', 'other']


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
    Angle of each neuron: project the embeddings onto two directions and take arctan.

    The directions (utils.projection_directions) are the top two PCs of the
    zero-meaned embeddings of reference_type (e.g. 'EPG'), or of all embeddings if
    reference_type is None. Same arguments as utils.sort_embeddings.
    """
    reference_inds = None if reference_type is None else data.neuron_hash[reference_type]
    projection = projection_directions(embeddings, reference_inds)
    projected = (embeddings @ projection).detach().numpy()
    return np.arctan2(projected[:, 1], projected[:, 0])


def order_tau(theta_embedding, anatomy_deg, windings=(1, 0.5, 2)):
    """
    Kendall's tau between the order of the embedding angles and the anatomical order.

    Allowed freedoms: the embedding angles live on a circle, so the order depends on
    where the circle is cut and in which direction it is read (best cut and direction
    are used); and the anatomical angles (degrees) can be scaled by a winding factor k
    from windings (k * angle, wrapped to 360), best k is used.
    Pairs of neurons with equal anatomical values are ignored, so neurons at the same
    anatomical position can be in any order.

    Returns:
        tau: in [0, 1] (1 = same order; ~0.3 for unrelated orders), NaN if every
            neuron has the same anatomical value
        winding: the k of the best match
    """
    order = np.argsort(np.asarray(theta_embedding) % (2 * np.pi))
    anatomy_sorted = np.asarray(anatomy_deg)[order]
    best_tau, best_k = np.nan, windings[0]
    for k in windings:
        a = np.round(k * anatomy_sorted, 6) % 360
        sign = np.sign(a[None, :] - a[:, None])   # sign[i, j] = sign(a_j - a_i)
        n_pairs = np.count_nonzero(sign) / 2
        if n_pairs == 0:
            continue
        # D = concordant - discordant pairs for the cut before neuron 0; moving the
        # first neuron to the end (moving the cut) changes D by -2 * sum_j sign[0, j]
        D0 = sign[np.triu_indices(len(a), 1)].sum()
        D = D0 - 2 * np.concatenate([[0], np.cumsum(sign.sum(1))])
        tau = np.abs(D).max() / n_pairs
        if np.isnan(best_tau) or tau > best_tau:
            best_tau, best_k = tau, k
    return best_tau, best_k


def null_order_tau(theta_embedding, anatomy_deg, n_shuffles, rng, windings=(1, 0.5, 2)):
    """order_tau for each of n_shuffles shuffles of the anatomical angles across neurons."""
    return np.array([order_tau(theta_embedding, rng.permutation(anatomy_deg), windings)[0]
                     for _ in range(n_shuffles)])


def score_groups(labels, angle_column, rng, windings=(1, 0.5, 2),
                 min_neurons=4, min_positions=3, n_shuffles=1000):
    """
    Score every group of neurons with anatomical labels:
      - PB glomerulus: one group per cell type and PB side
      - FB column:     one group per cell type
    Returns one row per group, with tau (see order_tau), the winding factor of the
    best match, and the median and 2.5/97.5th percentiles of tau after shuffling the
    anatomical labels (NaN if n_shuffles is 0). A label kind is skipped if its anatomy
    column is not in labels (e.g. no fb_angle_deg when add_fb_angles was not called).
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
            anatomy_deg = group[anatomy_column].values
            tau, winding = order_tau(theta_embedding, anatomy_deg, windings)
            if n_shuffles > 0:
                null = null_order_tau(theta_embedding, anatomy_deg, n_shuffles, rng, windings)
            else:
                null = np.full(1, np.nan)
            rows.append({
                'type': keys[0],
                'pb_side': keys[1] if len(keys) > 1 else None,
                'label': label_kind,
                'n_neurons': len(group),
                'n_positions': n_positions,
                'tau': tau,
                'tau_null_median': np.median(null),
                'tau_null_lo': np.percentile(null, 2.5),
                'tau_null_hi': np.percentile(null, 97.5),
                'winding': winding,
            })
    return pd.DataFrame(rows)


def mean_tau_per_type(embeddings, data, reference_type='EPG', pb_period=8,
                      windings=(1, 0.5, 2), min_neurons=4, min_positions=3):
    """
    Agreement of each cell type's embedding angles with its anatomical order: tau of
    score_groups (no shuffles), averaged over the type's groups (PB sides, FB columns).
    Angles from pca_angles with reference_type. Types without anatomical labels are
    left out.

    Returns a DataFrame with columns type, anatomy_tau, anatomy_n_groups.
    """
    labels = add_fb_angles(anatomical_labels(data.neurons, pb_period=pb_period))
    labels['angle_pca'] = pca_angles(embeddings, data, reference_type)
    scores = score_groups(labels, 'angle_pca', rng=None, windings=windings,
                          min_neurons=min_neurons, min_positions=min_positions,
                          n_shuffles=0)
    if len(scores) == 0:
        return pd.DataFrame(columns=['type', 'anatomy_tau', 'anatomy_n_groups'])
    return (scores.groupby('type')['tau'].agg(['mean', 'size'])
            .rename(columns={'mean': 'anatomy_tau', 'size': 'anatomy_n_groups'})
            .reset_index())


def type_groups(types):
    """Group of each cell type (one of GROUPS), as a dict type -> group."""
    types = list(types)
    cx = types_matching(types, CX_TYPES)
    cx_columnar = types_matching(types, CX_COLUMNAR_TYPES)
    ring_and_tubu = types_matching(types, RING_AND_TUBU_TYPES) - cx
    groups = {}
    for t in types:
        if t in cx_columnar:
            groups[t] = 'CX columnar'
        elif t in cx:
            groups[t] = 'CX other'
        elif t in ring_and_tubu:
            groups[t] = 'ring / TuBu'
        else:
            groups[t] = 'other'
    return groups
