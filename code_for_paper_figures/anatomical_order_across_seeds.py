#%%
"""
How much does the recovered anatomical order of central complex neurons depend on
the random seed and the embedding dimension D?

Uses the embeddings fitted by fig2_model_hyperparam_comparison.py (model with cell
types, no rotations; each seed has its own random 80/20 split of neuron pairs and
its own initialization). Each embedding is scored against anatomy as in
anatomical_order.py: Kendall's tau between embedding order and PB glomerulus / FB
column order, per cell type and PB side.

Run from code_for_paper_figures/. Plots tau across seeds and D.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from connectome_embedding import plots, anatomy

plots.set_rc_params()

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
SAVE_DIR = 'embeddings/fig2_model_hyperparam_comparison'   # where fig2_model_hyperparam_comparison.py saves its fits
MODEL_KIND = 'with_types'
DS = [3, 5, 7, 9]
SEEDS = [0, 1, 2, 3, 4, 5]

# Scoring against anatomy (same settings as anatomical_order.py)
REFERENCE_TYPE = 'EPG'
PB_PERIOD = 8
WINDINGS = (1, 0.5, 2)
MIN_NEURONS = 4
MIN_POSITIONS = 3


def embedding_filename(D, seed, model_kind):
    """Same file names as fig2_model_hyperparam_comparison.py."""
    return os.path.join(SAVE_DIR, f'CX_D{D}_seed{seed}_{model_kind}.pkl')


#%% Score every embedding against anatomy
rng = np.random.default_rng(0)
labels = None
score_rows = []
for D in DS:
    for seed in SEEDS:
        filename = embedding_filename(D, seed, MODEL_KIND)
        if not os.path.isfile(filename):
            print(f'missing: {filename}')
            continue
        saved = pickle.load(open(filename, 'rb'))
        data = saved['data']
        embeddings = saved['model'].embeddings.detach()

        # anatomical labels are the same for every embedding (same neurons, same order)
        if labels is None:
            labels = anatomy.add_fb_angles(
                anatomy.anatomical_labels(data.neurons, pb_period=PB_PERIOD))

        labels['angle'] = anatomy.pca_angles(embeddings, data, REFERENCE_TYPE)
        scores = anatomy.score_groups(labels, 'angle', rng, windings=WINDINGS,
                                      min_neurons=MIN_NEURONS, min_positions=MIN_POSITIONS,
                                      n_shuffles=0)
        score_rows.append(scores.assign(D=D, seed=seed))
        print(f'D = {D}, seed {seed}: median tau {scores["tau"].median():.3f}')

scores = pd.concat(score_rows, ignore_index=True)
scores = scores.drop(columns=['tau_null_median', 'tau_null_lo', 'tau_null_hi'])
scores['group'] = (scores['type']
                   + scores['pb_side'].fillna('').map(lambda s: f' ({s})' if s else '')
                   + ' ' + scores['label'].map({'PB glomerulus': 'PB', 'FB column': 'FB'}))

#%% Summary per D: median tau over groups, mean and s.d. across seeds
per_run = scores.groupby(['D', 'seed']).agg(median_tau=('tau', 'median'))
print(per_run.groupby('D').agg(['mean', 'std']).round(3))

# spread across seeds for each group and D
per_group = scores.groupby(['group', 'D'])['tau'].agg(['mean', 'std', 'min']).reset_index()
print(per_group.sort_values('std', ascending=False).head(20).round(3).to_string(index=False))

#%% Plot: tau of every group, mean and range across seeds, one color per D
order = (scores[scores['D'] == 5].groupby('group')['tau'].mean()
         .sort_values(ascending=False).index.tolist())
x_of = {g: i for i, g in enumerate(order)}
colors = plt.cm.viridis(np.linspace(0, 0.9, len(DS)))

plt.figure(figsize=(max(3, 0.12 * len(order)), 2), dpi=300)
for c, D in zip(colors, DS):
    sub = per_group[(per_group['D'] == D) & per_group['group'].isin(x_of)]
    x = sub['group'].map(x_of).values + (DS.index(D) - (len(DS) - 1) / 2) * 0.15
    plt.errorbar(x, sub['mean'], yerr=sub['std'], fmt='o', markersize=1.5, color=c,
                 elinewidth=0.5, capsize=0, label=f'D = {D}')
plt.xticks(range(len(order)), order, rotation=90, fontsize=4)
plt.ylim(0, 1.05)
plt.ylabel("Kendall's tau with\nanatomical order")
plt.title('mean ± s.d. across seeds', fontsize=5)
plt.legend(fontsize=4, frameon=False, ncol=len(DS))
plt.tight_layout()

#%% Plot: median tau over groups vs D, one dot per seed
plt.figure(figsize=(1.8, 1.5), dpi=300)
per_run_reset = per_run.reset_index()
plt.scatter(per_run_reset['D'], per_run_reset['median_tau'], s=4, color='k', alpha=0.5, lw=0)
means = per_run_reset.groupby('D')['median_tau'].mean()
plt.plot(means.index, means.values, color='r', marker='o', markersize=2, lw=1)
plt.xticks(DS)
plt.xlabel('embedding dimension D')
plt.ylabel("median Kendall's tau\nover cell types")
plt.ylim(0, 1.05)
plt.tight_layout()

# %%
