#%%
"""
How well does the embedding recover the anatomical order of central complex neurons?

Takes an embedding pickle (as saved by embed_hemibrain.py) and
  1. reads each neuron's anatomical position from its hemibrain instance name
     (data.neurons, the DataFrame of neuron info saved with the embedding; row i of
     data.neurons is row i of model.embeddings),
  2. reads each neuron's angle off the embedding, and
  3. compares the two per cell type, and plots the result.

Anatomical labels in the instance names (examples):
    EPG(PB08)_L4          PB glomerulus L4
    PEN_a(PB06a)_R7       PB glomerulus R7
    PFNa(PB03)_L2_C1      PB glomerulus L2, FB column C1
    PFL1(PB12a)_R1_C2_irreg   PB glomerulus R1, FB column C2, flagged irregular
    FC1A(FQ6)_C3_L        FB column C3, soma on the left
    hDeltaA_05_C5         FB column C5
    hDeltaK_07_C6b        FB column C6 (sub-column letter ignored)
Not parsed (left as NaN): neurons spanning several glomeruli, such as
Delta7(PB15)_L4R5_R or IbSpsP(PB17)_L2/4, and types with no glomerulus or column
in the name (e.g. ER, FB tangential, EL).

Anatomical angles:
  - PB glomerulus, separately for each side of the PB: glomeruli 1..8 tile the full
    circle in 45 degree steps, and glomerulus 9 has the same phase as glomerulus 1
    (PB_PERIOD). The two sides are compared separately, so no offset between them
    is assumed.
  - FB column: the columns of a cell type are taken to tile the full circle evenly,
    angle = 360 * (column - 1) / (number of columns of that type).

Angles from the embedding: project all embeddings onto the top two principal
components of the embeddings of a reference cell type (REFERENCE_TYPE, EPG) and take
the arctangent.

Comparison: connectivity alone cannot fix where 0 degrees is or which way is
clockwise, so each comparison allows one global rotation and one reflection.
The two PB sides of a cell type can also sit on the embedding circle in two ways:
as two copies of the full circle (e.g. EPG: glomeruli 1..8 of each side go once
around), or as the two halves of one circle (e.g. PEN: glomeruli 1..8 of each side
cover half of it). So each comparison also allows the anatomical angle to be scaled
by a winding factor k in WINDINGS (1 = full circle per side, 0.5 = half circle,
2 = twice around the circle).
The score is the circular alignment
    R = max over reflection s and winding k of
        | mean_i exp(i (s * theta_embedding - k * theta_anatomy)) |,
which is 1 when the two orders agree up to rotation/reflection and near 0 when
they are unrelated. The chosen k is reported. Also reported: the median absolute angle error after the best
rotation/reflection, and the null distribution of R (anatomical labels shuffled
within the group, with the same maximization over s and k): its median and 2.5th/97.5th
percentiles. The null is not centered at 0, especially for small groups.

Run from code_for_paper_figures/. Saves a CSV of per-neuron labels and angles and a
CSV of per-type scores, and plots the scores.
"""
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from connectome_embedding import plots, anatomy

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
EMBEDDING_NAME = 'CX_embedding.pkl'

# Reference cell type: all embeddings are projected on the top two PCs of its embeddings
REFERENCE_TYPE = 'EPG'

# Number of distinct phases per side of the PB: glomeruli 1..8, with 9 the same as 1
PB_PERIOD = 8

# Winding factors allowed between anatomical and embedding angles:
# 1 = each group's positions go once around the embedding circle,
# 0.5 = they cover half of it (the other half belongs to the other PB side)
# 2 = they go around the circle twice.
WINDINGS = (1, 0.5, 2)

# Only score groups (cell type, and PB side for glomerulus labels) with at least
# this many labelled neurons and at least this many distinct anatomical positions
MIN_NEURONS = 4
MIN_POSITIONS = 3

# Shuffles of the anatomical labels used to estimate the null distribution of R
N_SHUFFLES = 1000

embedding_tag = EMBEDDING_NAME.split('/')[-1].replace('.pkl', '')
LABELS_SAVE_NAME = f'anatomical_order_neurons_{embedding_tag}.csv'
SCORES_SAVE_NAME = f'anatomical_order_scores_{embedding_tag}.csv'


#%% Load the embedding and its neuron info
plots.set_rc_params()
rng = np.random.default_rng(0)

model_and_data = pickle.load(open(EMBEDDING_NAME, 'rb'))
model = model_and_data['model']
data = model_and_data['data']
embeddings = model.embeddings.detach()

# row i of data.neurons is row i of the embeddings
assert len(data.neurons) == data.N, 'data.neurons does not match the embedding size'

#%% Anatomical labels
labels = anatomy.anatomical_labels(data.neurons, pb_period=PB_PERIOD)
labels = anatomy.add_fb_angles(labels)

summary = anatomy.summarize_by_type(labels)
pd.set_option('display.max_rows', 200)
print(summary)
n_labelled = (labels['pb_glomerulus'].notna() | labels['fb_column'].notna()).sum()
print(f'{n_labelled} of {len(labels)} neurons have a PB glomerulus or FB column')

#%% Angles from the embedding
labels['angle_pca'] = anatomy.pca_angles(embeddings, data, REFERENCE_TYPE)

#%% Compare, per cell type
scores = anatomy.score_groups(labels, 'angle_pca', rng, windings=WINDINGS,
                              min_neurons=MIN_NEURONS, min_positions=MIN_POSITIONS,
                              n_shuffles=N_SHUFFLES)
print(scores.sort_values('R', ascending=False).to_string(index=False))

#%% Plots
# dot plot of R per group, sorted by R; shuffle null as median with 2.5-97.5 percentile bars
plot_data = scores.sort_values('R', ascending=False)
group_names = (plot_data['type']
               + plot_data['pb_side'].fillna('').map(lambda s: f' ({s})' if s else '')
               + ' ' + plot_data['label'].map({'PB glomerulus': 'PB', 'FB column': 'FB'}))
x = np.arange(len(plot_data))

plt.figure(figsize=(max(3, 0.12 * len(x)), 2), dpi=300)
plt.scatter(x, plot_data['R'], s=6, color='r', label='R', zorder=3)
plt.errorbar(x, plot_data['R_null_median'],
             yerr=[plot_data['R_null_median'] - plot_data['R_null_lo'],
                   plot_data['R_null_hi'] - plot_data['R_null_median']],
             fmt='o', color='k', ecolor='k', elinewidth=0.5, capsize=1,
             label='shuffled (median, 95%)', zorder=2, markersize=2)
plt.xticks(x, group_names, rotation=90, fontsize=4)
plt.ylim(0, 1.05)
plt.ylabel('alignment with\nanatomical order (R)')
plt.legend(fontsize=5, frameon=False)
plt.tight_layout()

#%% Save
labels.to_csv(LABELS_SAVE_NAME, index=False)
scores.to_csv(SCORES_SAVE_NAME, index=False)
print(f'Saved {LABELS_SAVE_NAME} and {SCORES_SAVE_NAME}')
