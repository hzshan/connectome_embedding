#%%
"""
Fig. 4: circularity screen of the cell types in the whole-hemibrain embedding.

The circularity scores are computed by score_type_geometry.py (circle search and
circularity score per cell type); this script plots them. How well the circularity
score separates CX from non-CX types is in ring_detection_auc.py.
Run from code_for_paper_figures/.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from connectome_embedding import plots

plots.set_rc_params()

# Per-type tables made by score_type_geometry.py
RESULTS_DIR = 'results/type_geometry/'
HEMIBRAIN_EMBEDDING = 'whole_hemibrain_embedding'

hemibrain_table = pd.read_csv(os.path.join(RESULTS_DIR, HEMIBRAIN_EMBEDDING, 'circularity.csv'))


def plot_circularity(table, colors=None, title=''):
    """Bar per type, sorted by circularity score. colors: one per row of table, or None (gray)."""
    table = table.assign(color='gray' if colors is None else list(colors))
    table = table.sort_values('circularity', ascending=False).reset_index(drop=True)
    plt.figure(figsize=(12, 1.5), dpi=300)

    plt.xticks(rotation=90, fontsize=5)
    plt.bar(table['type'], table['circularity'] - 0.6, color=table['color'])
    plt.yticks([0, 0.4], ['0.6', '1.0'])
    plt.ylim([0, 0.5])
    plt.xlim(-1, len(table) - 0.5)
    plt.ylabel('circularity\nscore', fontsize=6)
    plt.title(title, fontsize=6)
    plt.tight_layout()


#%% Whole-hemibrain embedding; CX types in gold
is_cx = hemibrain_table['group'].isin(['CX columnar', 'CX other'])
plot_circularity(hemibrain_table, colors=np.where(is_cx, 'goldenrod', 'gray'),
                 title=f'Whole-hemibrain embedding ({HEMIBRAIN_EMBEDDING}); gold: CX types')
