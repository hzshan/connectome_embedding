#%%
"""
Score the shape of every cell type's embedding, for a list of embeddings.

For each embedding in EMBEDDINGS, each cell type with at least MIN_NEURONS neurons is
scored by each function in SCORERS:
  circularity  best 2D projection of the type's embeddings onto a circle and its
               circularity score (geometry_scores.circularity_scores)
  homology     longest H1 (loop) lifetime from persistent homology
               (geometry_scores.h1_scores), optionally relative to matched Gaussian
               clouds (H1_N_NULL)
  dimension    intrinsic dimension, Gride estimator at neighbour scale GRIDE_N1
               (geometry_scores.dimension_scores)
  anatomy      Kendall's tau between embedding angles and PB glomerulus / FB column
               order, for CX types with anatomical labels (anatomy.mean_tau_per_type)
Every table also has the type's number of neurons and group (CX columnar, CX other,
ring / TuBu, other).

Each score of each embedding is saved as its own CSV,
RESULTS_DIR/<embedding>/<score>.csv, so one score can be recomputed without the
others: delete its CSV and rerun. Existing CSVs are skipped, and missing embedding
files are reported and skipped. ring_detection_auc.py and fig4_circle_screen.py read
the CSVs.

Embeddings are made by embed_hemibrain.py (the null models with the
'whole_hemibrain_null' config, one per SHUFFLE_SEED).
Needs ripser (pip install ripser). Run from code_for_paper_figures/.
"""
import os
import pickle
import numpy as np
import pandas as pd
import torch
from connectome_embedding import geometry_scores, anatomy

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
EMBEDDING_DIR = 'embeddings/'
RESULTS_DIR = 'results/type_geometry/'
NULL_SEEDS = list(range(20))
EMBEDDINGS = (['whole_hemibrain_embedding.pkl']
              + [f'whole_hemibrain_embedding_block_shuffled_seed{s}.pkl' for s in NULL_SEEDS])

MIN_NEURONS = 15     # score types with at least this many neurons

# Circularity: settings of the circle search (utils.solve_shadowmatic2)
CIRCLE_SEARCH = {'nseeds_to_try': 50, 'max_iter': 10}

# Homology
MAX_POINTS = 300     # subsample larger types to keep ripser fast
H1_N_NULL = 100      # matched Gaussian clouds per type for h1_relative (0: none)
SEED = 0             # seed for subsampling and the Gaussian clouds

# Intrinsic dimension
GRIDE_N1 = 4         # Gride neighbour scale (n2 = 2 * n1 = 8 < MIN_NEURONS). At n1 = 1
                     # (TwoNN) the estimate exceeds D when nearest neighbours are more
                     # regularly spaced than at random

# Anatomy
ANATOMY_REFERENCE_TYPE = 'EPG'   # embeddings are projected onto the top two PCs of this type


#%% Scoring functions: (embeddings, data, types) -> one row per type
def score_circularity(embeddings, data, types):
    return geometry_scores.circularity_scores(embeddings, data, types, CIRCLE_SEARCH)


def score_homology(embeddings, data, types):
    return geometry_scores.h1_scores(embeddings, data, types, np.random.default_rng(SEED),
                                     max_points=MAX_POINTS, n_null=H1_N_NULL)


def score_dimension(embeddings, data, types):
    return geometry_scores.dimension_scores(embeddings, data, types, GRIDE_N1)


def score_anatomy(embeddings, data, types):
    if ANATOMY_REFERENCE_TYPE not in data.neuron_hash:
        return pd.DataFrame({'type': types})
    tau = anatomy.mean_tau_per_type(torch.as_tensor(embeddings), data,
                                    reference_type=ANATOMY_REFERENCE_TYPE)
    return tau[tau['type'].isin(types)]


SCORERS = {
    'circularity': score_circularity,
    'homology': score_homology,
    'dimension': score_dimension,
    'anatomy': score_anatomy,
}


#%% Score every embedding; each score is skipped if its CSV exists
for name in EMBEDDINGS:
    embedding = name.replace('.pkl', '')
    outputs = {score: os.path.join(RESULTS_DIR, embedding, f'{score}.csv') for score in SCORERS}
    to_do = [score for score, out in outputs.items() if not os.path.isfile(out)]
    if not to_do:
        print(f'all scores exist, skipped: {embedding}')
        continue
    path = os.path.join(EMBEDDING_DIR, name)
    if not os.path.isfile(path):
        print(f'missing embedding, skipped: {path}')
        continue

    saved = pickle.load(open(path, 'rb'))
    data = saved['data']
    embeddings = saved['model'].embeddings.detach().numpy()
    types = geometry_scores.types_to_score(data, MIN_NEURONS)
    type_info = pd.DataFrame({'type': types,
                              'n_neurons': [len(data.neuron_hash[t]) for t in types]})
    type_info['group'] = type_info['type'].map(anatomy.type_groups(data.types))

    os.makedirs(os.path.join(RESULTS_DIR, embedding), exist_ok=True)
    for score in to_do:
        print(f'{embedding}: {score}')
        table = type_info.merge(SCORERS[score](embeddings, data, types), on='type', how='left')
        table.insert(0, 'embedding', embedding)
        table.to_csv(outputs[score], index=False)
        print(f'saved {outputs[score]} ({len(table)} types)')

# %%
