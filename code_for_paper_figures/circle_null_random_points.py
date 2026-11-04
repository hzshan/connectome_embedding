#%%
"""
Null distribution of the circularity score for random points (Fig. 4).

For each N = 1..MAX_N, draw N i.i.d. Gaussian points
in D dimensions, run the SAME circle search used on the real embeddings
(utils.solve_shadowmatic2), and record the circularity score. Repeat N_SAMPLES
times per N. fig4_circle_screen.py compares each cell type's score with the null
for its N; types with more than MAX_N neurons use the MAX_N null (see the note
there on why that is conservative).

Run from code_for_paper_figures/. Saves the null to SAVE_NAME.
"""
import pickle, datetime
import numpy as np
import torch
import tqdm
from connectome_embedding import utils

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
D = 5             # embedding dimension; must match the embeddings being screened
N_SAMPLES = 100   # random point clouds per N
SEED = 0          # seed for generating the random points

# Settings of the circle search. fig4_circle_screen.py checks that it uses the same.
CIRCLE_SEARCH = {'nseeds_to_try': 50, 'max_iter': 10}

# Type sizes to compute the null for: N = 1..MAX_N.
# Larger types reuse the MAX_N null in fig4_circle_screen.py.
MAX_N = 149

SAVE_NAME = f'circle_null_random_points_D{D}.pkl'

results = {
    'D': D,
    'max_N': MAX_N,
    'n_samples': N_SAMPLES,
    'seed': SEED,
    'circle_search': CIRCLE_SEARCH,
    'point_distribution': 'i.i.d. standard normal',
    'scores': {},   # N -> array of N_SAMPLES circularity scores
}

#%% Compute the null, one type size at a time
for N in tqdm.tqdm(range(1, MAX_N + 1)):
    # Separate random stream per N, so results don't depend on which sizes ran before
    generator = torch.Generator().manual_seed(SEED * 1_000_003 + N)

    scores = np.zeros(N_SAMPLES)
    for k in range(N_SAMPLES):
        # N random points in D dimensions, the null stand-in for one cell type's embeddings
        x = torch.randn(N, D, generator=generator)

        # same circle search as used on the real embeddings
        x_2d, _, _ = utils.solve_shadowmatic2(x, **CIRCLE_SEARCH)
        scores[k] = float(utils.circ_score_via_norm(x_2d, skip_alignment=True))

    results['scores'][N] = scores

results['time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
pickle.dump(results, open(SAVE_NAME, 'wb'))

print(f'Saved {SAVE_NAME}')

#%% Quick look: null mean and spread vs N
import matplotlib.pyplot as plt
Ns_done = sorted(results['scores'])
means = [results['scores'][n].mean() for n in Ns_done]
stds = [results['scores'][n].std() for n in Ns_done]
plt.figure(figsize=(3, 2))
plt.errorbar(Ns_done, means, yerr=stds, fmt='.', ms=2)
plt.xscale('log')
plt.xlabel('number of neurons N')
plt.ylabel('circularity score\n(random points)')
plt.title(f'D = {D}')
plt.tight_layout()
