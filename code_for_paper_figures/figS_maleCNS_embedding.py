#%%
"""
Connectivity among central complex cell types in the maleCNS brain embedding
(embed_malecns_brain.py), with the neurons of each type sorted by their angle in
the embedding.

Angles are read off the top two principal components of the embeddings of a
reference cell type (REFERENCE_TYPE, EPG), as in anatomical_order.py
(utils.sort_embeddings). The embedding file has no connectivity matrix, so the
connections among the plotted neurons are read from the maleCNS weights file.

Run from code_for_paper_figures/.
"""
import os
import pickle
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import matplotlib.pyplot as plt
from connectome_embedding import plots, utils, data_utils

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
EMBEDDING_NAME = 'embeddings/malecns_brain_D7_reg0.01_seed0.pkl'
DATAPATH = os.path.expanduser('~/Dropbox/Data/data_maleCNS/')
WEIGHTS_FILE = 'connectome-weights-male-cns-v0.9-minconf-0.5.feather'

# Reference cell type: embeddings are sorted by angle in the top two PCs of its embeddings
REFERENCE_TYPE = 'EPG'

# Cell types shown in the connectivity plot, in this order
PLOT_TYPES = ['EPG', 'Delta7', 'PEG']


#%% Load the embedding and its neuron info
plots.set_rc_params()

model_and_data = pickle.load(open(EMBEDDING_NAME, 'rb'))
model = model_and_data['model']
data = model_and_data['data']
embeddings = model.embeddings.detach()

# row i of data.neurons is row i of the embeddings
assert len(data.neurons) == data.N, 'data.neurons does not match the embedding size'

#%% Connectivity among PLOT_TYPES, each type sorted by embedding angle
# sort_embeddings sorts within each cell type and keeps the types in place, so the
# neurons of a type are the same positions before and after sorting
assert (data.neurons['type'].values == np.repeat(data.types, data.N_per_type)).all(), \
    'neurons must be grouped by cell type, in the order of data.types'
sorted_inds = utils.sort_embeddings(embeddings, data, reference_type=REFERENCE_TYPE)
plot_inds = np.concatenate([sorted_inds[data.neuron_hash[t]] for t in PLOT_TYPES])
plot_neurons = data.neurons.iloc[plot_inds].reset_index(drop=True)

# connections among the plotted neurons; the filter runs on the memory-mapped file,
# so the 150M-row table is not loaded into memory
table = feather.read_table(DATAPATH + WEIGHTS_FILE, memory_map=True)
body_ids = pa.array(plot_neurons['bodyId'].values)
among_plotted = pc.and_(pc.is_in(table['body_pre'], value_set=body_ids),
                        pc.is_in(table['body_post'], value_set=body_ids))
conns = table.filter(among_plotted).to_pandas()
del table

# J[i, j]: synapses from plotted neuron i to plotted neuron j, in plot order
sorted_data = data_utils.ConnectivityData(
    data_utils.sparse_connectivity(plot_neurons['bodyId'], conns).toarray(),
    plot_neurons, 'type')
print(f'{sorted_data.N} neurons, {int(sorted_data.J.sum())} synapses among them')

plt.figure(figsize=(2.5, 2.5), dpi=300)
plots.plot_J(sorted_data.J, sorted_data, PLOT_TYPES, ax=plt.gca())
cbar = plt.colorbar(ticks=[0, np.log(2), np.log(11), np.log(101)])
cbar.ax.set_yticklabels([0, 1, 10, 100])
cbar.set_label('# synapses')
plt.tight_layout()

# %%
