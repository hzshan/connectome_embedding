#%%

import numpy as np
import matplotlib.pyplot as plt
import torch
from connectome_embedding import utils, models, plots, models, grid_cell

plots.set_rc_params()

n_e = 30  # the network has n^2 excitatory neurons and n^2 inhibitory neurons
n_i = 20
N_e = n_e**2  # size of each module
N_i = n_i**2
full_N = N_e + N_i  # size of the full network

sheet_width = n_e

lambda_w = n_e / 128 * 13  # 128 is the n in Burak and Fiete. 
beta = 3.
noise_strength = 1.0

range_e = np.linspace(-sheet_width / 2, sheet_width / 2, n_e, endpoint=False)
range_e -= range_e.mean()  # center the grid around (0,0)
X_e, Y_e = np.meshgrid(range_e, range_e)
X_e = X_e.flatten()
Y_e = Y_e.flatten()

range_i = np.linspace(-sheet_width / 2, sheet_width / 2, n_i, endpoint=False)
range_i -= range_i.mean()  # center the grid around (0,0)
X_i, Y_i = np.meshgrid(range_i, range_i)
X_i = X_i.flatten()
Y_i = Y_i.flatten()


full_W = grid_cell.make_synthetic_grid_cell_weights(
    n_excitatory=n_e,
    n_inhibitory=n_i,
    lambda_w=lambda_w,
    beta=beta,
)

W_to_use = full_W

#%% Schematics of excitatory and inhibitory neurons on a sheet
plt.figure(figsize=(2, 2))
plt.scatter(X_e, Y_e, s=1, c='k')
plt.scatter(X_i, Y_i, s=1, c='r')
plt.xlabel('sheet dim 1')
plt.ylabel('sheet dim 2')
plt.title('Grids of Exc. and Inh. Neurons')
plt.xlim(-sheet_width / 2, sheet_width / 2)
plt.ylim(-sheet_width / 2, sheet_width / 2)
plt.tight_layout()
# plt.savefig('../Figures/grids.svg', dpi=300)

#%% More schematics
W_EE_sheet = W_to_use[:N_e, :N_e].reshape(n_e, n_e, n_e, n_e)
W_EI_sheet = W_to_use[:N_e, N_e:].reshape(n_e, n_e, n_i, n_i)
plt.figure(figsize=(2, 2))
plt.imshow(np.log(1+W_EE_sheet[2, 2]), cmap='gray_r', vmax=np.log(11), vmin=0)
plt.xlabel('exc. coordinate 1')
plt.ylabel('exc. coordinate 2')
plt.colorbar()
plt.title('W_EE Connectivity from Single Neuron')
plt.tight_layout()

#%% Grid-cell circuit connectivity
plt.figure(figsize=(2, 2), dpi=300)
plt.imshow(np.log(1 + W_to_use), cmap='gray_r', vmin=0, vmax=np.log(101))
plt.axhline(N_e - 0.5, color='k', lw=0.25)
plt.axvline(N_e - 0.5, color='k', lw=0.25)
plt.xticks([N_e / 2, N_i / 2 + N_e], ['Exc.', 'Inh.'])
plt.yticks([N_e / 2, N_i / 2 + N_e], ['Exc.', 'Inh.'])

cbar = plt.colorbar(ticks=[0, np.log(2), np.log(11), np.log(101)])
cbar.ax.set_yticklabels([0, 1, 10, 100])
plt.title('True Circuit Connectivity Matrix')
plt.tight_layout()

# plt.savefig('../Figures/grid_cell.svg', dpi=300)


plt.figure(figsize=(2, 2), dpi=300)
perm = np.concatenate([torch.randperm(N_e), N_e + torch.randperm(N_i)])
plt.imshow(np.log(1 + W_to_use[perm][:, perm]), cmap='gray_r', vmin=0, vmax=np.log(101))
plt.axhline(N_e - 0.5, color='k', lw=0.25)
plt.axvline(N_e - 0.5, color='k', lw=0.25)
plt.xticks([N_e / 2, N_i / 2 + N_e], ['Exc.', 'Inh.'])
plt.yticks([N_e / 2, N_i / 2 + N_e], ['Exc.', 'Inh.'])
cbar = plt.colorbar(ticks=[0, np.log(2), np.log(11), np.log(101)])
cbar.ax.set_yticklabels([0, 1, 10, 100])
plt.tight_layout()
plt.title('Permuted Circuit Connectivity Matrix')
# plt.savefig('../figures/grid_cell_W_perm.svg', dpi=300)

#%% learn an embedding and use it sort the permuted matrix

type_onehot = torch.zeros((full_N, 2))
type_onehot[:N_e, 0] = 1
type_onehot[N_e:, 1] = 1

def reg_fn(model: models.InteractionModel):
    return torch.mean(model.rotation_params**2) * 1000 +\
             torch.mean(model.translation_vecs**2) +\
             torch.mean(model.embeddings**2) * 0.1

model = models.InteractionModel(N=W_to_use.shape[0],
                                Ntype=2,
                                D=5,
                                onehot_types=type_onehot)
model.use_transforms = False
models.train_model(model, W_to_use, lr=0.05, print_every=500, reg_fn=reg_fn, steps=4000,
                   loss_type='poisson',
                   )

#%% Show embedding
model.embeddings = model.embeddings.detach()
embeddings = model.embeddings.numpy()
u, s, v = np.linalg.svd(embeddings)
#project on top 3 pcs
embeddings = (embeddings @ v)


fig = plt.figure(figsize=(2, 2), dpi=200)
ax = plt.axes(projection='3d')
ax.scatter3D(embeddings[:N_e, 0],
                embeddings[:N_e, 1],
                embeddings[:N_e, 2], s=0.3, color='k')
ax.set_box_aspect((1, 1, 1))
plt.xlabel('embedding PC 1')
plt.ylabel('embedding PC 2')
ax.set_zlabel('embedding PC 3')
plt.title('Excitatory Neurons Embeddings')

# set viewing angle
ax.view_init(elev=30, azim=105)

plt.tight_layout()
# plt.savefig('figures/grid_cell_embeddings_e.svg', dpi=300)


fig = plt.figure(figsize=(2, 2), dpi=200)
ax = plt.axes(projection='3d')
ax.scatter3D(embeddings[N_e:, 0],
                embeddings[N_e:, 1],
                embeddings[N_e:, 2], s=0.3, color='r')
ax.set_box_aspect((1, 1, 1))
plt.xlabel('embedding PC 1')
plt.ylabel('embedding PC 2')
ax.set_zlabel('embedding PC 3')
plt.title('Inhibitory Neurons Embeddings')
ax.view_init(elev=30, azim=105)
plt.tight_layout()


#%% Recover sorting


torus_e = grid_cell.make_clifford_torus(n_e)
torus_i = grid_cell.make_clifford_torus(n_i)


rand_perm = np.concatenate([np.random.permutation(N_e), np.random.permutation(N_i) + N_e])


emb, proj, perm, _ = grid_cell.solve_shadowmatic_torus(model.embeddings[rand_perm].detach()[:N_e], torus_e.float(), 50)

projed_i_neurons = model.embeddings[rand_perm].detach()[N_e:] @ proj
i_perm = utils.find_perm(torus_i.float(), projed_i_neurons)[0]

shuffled_W = W_to_use[rand_perm][:, rand_perm]

recovered_W = torch.zeros_like(W_to_use)
recovered_W[:N_e, :N_e] = perm @ shuffled_W[:N_e, :N_e] @ perm.T
recovered_W[:N_e, N_e:] = perm @ shuffled_W[:N_e, N_e:] @ i_perm.T
recovered_W[N_e:, :N_e] = i_perm @ shuffled_W[N_e:, :N_e] @ perm.T
recovered_W[N_e:, N_e:] = i_perm @ shuffled_W[N_e:, N_e:] @ i_perm.T


plt.figure(figsize=(2, 2), dpi=300)
plt.imshow(np.log(1 + recovered_W), cmap='gray_r', vmin=0, vmax=np.log(101))
plt.axhline(N_e - 0.5, color='k', lw=0.25)
plt.axvline(N_e - 0.5, color='k', lw=0.25)
plt.xticks([N_e / 2, N_i / 2 + N_e], ['Exc.', 'Inh.'])
plt.yticks([N_e / 2, N_i / 2 + N_e], ['Exc.', 'Inh.'])
plt.title('Recovered Connectivity Matrix')
cbar = plt.colorbar(ticks=[0, np.log(2), np.log(11), np.log(101)])
cbar.ax.set_yticklabels([0, 1, 10, 100])
plt.tight_layout()
# plt.savefig('figures/recovered_W.svg', dpi=300)


print(torch.norm(recovered_W - full_W) / torch.norm(full_W))
print(torch.norm(W_to_use - full_W) / torch.norm(full_W))
