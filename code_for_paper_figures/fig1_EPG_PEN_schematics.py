#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
from connectome_embedding import utils, models
import pickle
from connectome_embedding import plots

plots.set_rc_params()

embedding_name = '../Learned_embeddings/' + \
'embedding_AllCX_D5_LRsplit_woDelta7_031225.pkl'
# embedding_name = 'Learned_embeddings/embedding_AllCX_D5_noLRsplit_030625.pkl'
model_and_data = pickle.load(open(embedding_name, 'rb'))
model = model_and_data['model']
data = model_and_data['data']

sorted_inds = utils.sort_embeddings(model.embeddings, data)
sorted_Adj = data.J[sorted_inds][:, sorted_inds]

#%%
EPG_inds = np.hstack((data.neuron_hash['EPG_L'], 
                     data.neuron_hash['EPG_R']))
PEN_left_inds = np.hstack((data.neuron_hash['PEN_a(PEN1)_L'],
                           data.neuron_hash['PEN_b(PEN2)_L'],))
PEN_right_inds = np.hstack((data.neuron_hash['PEN_a(PEN1)_R'],
                            data.neuron_hash['PEN_b(PEN2)_R'],))

combined_inds = np.hstack((EPG_inds, PEN_left_inds, PEN_right_inds))
N_per_type = np.array([
    len(arr) for arr in [EPG_inds, PEN_left_inds, PEN_right_inds]])

embeddings = model.embeddings[combined_inds]
_, _, V = torch.svd(embeddings)
projected_all_embeddings = embeddings @ V

sorted_inds = []

for i in range(3):
    start_ind = int(N_per_type[:i].sum())
    end_ind = int(N_per_type[:i+1].sum())

    angle = np.arctan2(projected_all_embeddings[start_ind:end_ind, 1],
                    projected_all_embeddings[start_ind:end_ind, 0])
    sorted_inds += list(np.argsort(angle) + len(sorted_inds))
sorted_inds = np.array(sorted_inds)

np.random.seed(0)
total_shuffle_inds = np.random.permutation(len(combined_inds))
total_shuffled_J = data.J[combined_inds][:, combined_inds]

epg_inds_shuffled = EPG_inds[np.random.permutation(len(EPG_inds))]
PEN_left_inds_shuffled = PEN_left_inds[np.random.permutation(len(PEN_left_inds))]
PEN_right_inds_shuffled = PEN_right_inds[np.random.permutation(len(PEN_right_inds))]
within_type_shuffle_inds = np.concatenate([epg_inds_shuffled, PEN_left_inds_shuffled, PEN_right_inds_shuffled])
within_type_shuffled_J = data.J[within_type_shuffle_inds][:, within_type_shuffle_inds]

plt.figure(figsize=(2, 2))
plt.imshow(np.log(1 + total_shuffled_J[total_shuffle_inds][:, total_shuffle_inds].numpy()),
           aspect='equal', cmap='gray_r', interpolation='none')
cbar = plt.colorbar(ticks=[0, np.log(2), np.log(11), np.log(101)])
cbar.ax.set_yticklabels(labels=[0, 1, 10, 100], fontsize=6)
plt.tight_layout()
# plt.savefig('figures/epg_pen_fig1_full_shuffle.svg')

plt.figure(figsize=(2, 2))
plt.imshow(np.log(1 + within_type_shuffled_J),
           aspect='equal', cmap='gray_r', interpolation='none')
plt.xticks([N_per_type[0] // 2,
            N_per_type[0] + N_per_type[1] // 2,
            N_per_type[0] + N_per_type[1] + N_per_type[2] // 2],
           ['EPG', 'PEN\n(left)', 'PEN\n(right)'], fontsize=6)
plt.yticks([N_per_type[0] // 2,
            N_per_type[0] + N_per_type[1] // 2,
            N_per_type[0] + N_per_type[1] + N_per_type[2] // 2],
           ['EPG', 'PEN\n(left)', 'PEN\n(right)'], fontsize=6)
plt.axhline(y=N_per_type[0] - 0.5, color='gray', lw=0.5)
plt.axhline(y=N_per_type[0] + N_per_type[1] - 0.5, color='gray', lw=0.5)
plt.axvline(x=N_per_type[0] - 0.5, color='gray', lw=0.5)
plt.axvline(x=N_per_type[0] + N_per_type[1] - 0.5, color='gray', lw=0.5)

cbar = plt.colorbar(ticks=[0, np.log(2), np.log(11), np.log(101)])
cbar.ax.set_yticklabels(labels=[0, 1, 10, 100], fontsize=6)
plt.tight_layout()
plt.show()

sorted_J = data.J[combined_inds][:, combined_inds][sorted_inds][:, sorted_inds]

plt.figure(figsize=(2, 2))
plt.imshow(np.log(1 + sorted_J), cmap='gray_r', interpolation='none')
plt.xticks([N_per_type[0] // 2,
            N_per_type[0] + N_per_type[1] // 2,
            N_per_type[0] + N_per_type[1] + N_per_type[2] // 2],
           ['EPG', 'PEN\n(left)', 'PEN\n(right)'], fontsize=6)
plt.yticks([N_per_type[0] // 2,
            N_per_type[0] + N_per_type[1] // 2,
            N_per_type[0] + N_per_type[1] + N_per_type[2] // 2],
           ['EPG', 'PEN\n(left)', 'PEN\n(right)'], fontsize=6)
plt.axhline(y=N_per_type[0] - 0.5, color='gray', lw=0.5)
plt.axhline(y=N_per_type[0] + N_per_type[1] - 0.5, color='gray', lw=0.5)
plt.axvline(x=N_per_type[0] - 0.5, color='gray', lw=0.5)
plt.axvline(x=N_per_type[0] + N_per_type[1] - 0.5, color='gray', lw=0.5)

cbar = plt.colorbar(ticks=[0, np.log(2), np.log(11), np.log(101)])
cbar.ax.set_yticklabels(labels=[0, 1, 10, 100], fontsize=6)
plt.tight_layout()
plt.show()
# plt.savefig('figures/epg_pen_fig1.svg')