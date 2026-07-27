#%%
import numpy as np
# import matplotlib as mpl
import matplotlib.pyplot as plt
import torch, pickle
from connectome_embedding import utils, models, plots

plots.set_rc_params()

# embedding_name = 'Learned_embeddings/embedding_AllHemibrain_D5_noLRsplit_022525.pkl'
# embedding_name = 'Learned_embeddings/embedding_AllHemibrain_D5_noLRsplit_022525.pkl
# embedding_name = 'Learned_embeddings/embedding_Allhemibrain_D5_04172025.pkl'
embedding_name = '../Learned_embeddings/embedding_AllCX_D5_noLRsplit_030625.pkl'
model_and_data = pickle.load(open(embedding_name, 'rb'))
model = model_and_data['model']
data = model_and_data['data']
print(model_and_data['hyperparams'])
sort_ind = utils.sort_embeddings(model.embeddings, data)
sorted_J = data.J[sort_ind][:, sort_ind]
_ = plots.plot_J(sorted_J, data, tick_fontsize=8)
cbar = plt.colorbar(ticks=[0, np.log(2), np.log(11), np.log(101)])
cbar.set_ticklabels(['0', '1', '10', '100'])
# plt.savefig('Figures/full_CX.svg', dpi=300)

print(len(data.types))


# make goodness-of-fit figures of different model setups
# everything was run with LR=0.1 and lambda=0.1. Best loss out of 20k updates.
# the 4 elements correspond to D=3,5,7,9
Ds = [3, 5, 7, 9, 11]
with_cell_type_num_params = []
without_cell_type_num_params = []
for D in Ds:
    num_z = D * data.N
    type_params = data.Ntype**2 * 3
    with_cell_type_num_params.append(num_z + type_params)
    without_cell_type_num_params.append(num_z + 3)
with_cell_type_training_loss = [-0.2014, -0.2273, -0.2398, -0.2454]
without_cell_type_training_loss = [0.1722, 0.0714, 0.0131, -0.0214, -0.045]


plt.figure(dpi=300, figsize=(2, 1.5))

plt.plot(with_cell_type_num_params[:-1], with_cell_type_training_loss, label='with cell types', marker='o', markersize=3, color='r')
plt.plot(without_cell_type_num_params, without_cell_type_training_loss, label='without cell type', marker='o', markersize=3, color='k')
plt.xticks(rotation=0)
plt.xlabel('Number of parameters')
plt.ylabel('log LL')
plt.legend(fontsize=6)
plt.tight_layout()


# for the numbers below, a random 80% of the cell pairs were used to train the embeddings.
# the rest was used to test the model. The recorded test loss was taken at the checkpoint
# where the training loss was the lowest over 20k updates.
# the 4 elements correspond to D=3,5,7,9
with_cell_type_test_loss = [-0.1624, -0.2012, -0.2057, -0.1946]
without_cell_type_test = [0.2239, 0.155, 0.11, 0.102, 0.11]

plt.figure(dpi=300, figsize=(2, 1.5))

plt.plot(with_cell_type_num_params[:-1], with_cell_type_test_loss, label='with cell types', marker='o', markersize=3, color='r')
plt.plot(without_cell_type_num_params, without_cell_type_test, label='without cell type', marker='o', markersize=3, color='k')
plt.xticks(rotation=0)
plt.xlabel('Number of parameters')
plt.ylabel('log LL (test)')
plt.legend(fontsize=6)
plt.tight_layout()