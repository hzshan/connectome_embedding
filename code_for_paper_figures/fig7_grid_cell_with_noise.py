#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
from connectome_embedding import utils, models, grid_cell, plots

plots.set_rc_params()

n_e = 30
n_i = 20
N_e = n_e**2
N_i = n_i**2
full_N = N_e + N_i

lambda_w = n_e / 128 * 13
beta = 3.0

full_W = grid_cell.make_synthetic_grid_cell_weights(
    n_excitatory=n_e,
    n_inhibitory=n_i,
    lambda_w=lambda_w,
    beta=beta,
)

type_onehot = torch.zeros((full_N, 2))
type_onehot[:N_e, 0] = 1
type_onehot[N_e:, 1] = 1

def reg_fn(model):
    return (
        torch.mean(model.rotation_params**2) * 1000
        + torch.mean(model.translation_vecs**2)
        + torch.mean(model.embeddings**2) * 0.1
    )


# noise_levels = np.linspace(0, 2.0, 9)
noise_level = 2.0
train_steps = 4000
nseeds_torus = 50

torus_e = grid_cell.make_clifford_torus(n_e).float()
torus_i = grid_cell.make_clifford_torus(n_i).float()

np.random.seed(0)
torch.manual_seed(0)

W_noisy = torch.clamp(
    full_W + noise_level* (
        torch.randn_like(full_W) * full_W + torch.randn_like(full_W) * full_W.mean()),
    min=0,
)

model = models.InteractionModel(
    N=W_noisy.shape[0],
    Ntype=2,
    D=5,
    onehot_types=type_onehot,
)
model.use_transforms = False

models.train_model(
    model,
    W_noisy,
    lr=0.05,
    print_every=500,
    reg_fn=reg_fn,
    steps=train_steps,
    loss_type='poisson',
)

rand_perm = np.concatenate([
    np.random.permutation(N_e),
    np.random.permutation(N_i) + N_e,
])

_, proj, perm, score = grid_cell.solve_shadowmatic_torus(
    model.embeddings[rand_perm].detach()[:N_e],
    torus_e,
    nseeds_torus,
)
projected_i_neurons = model.embeddings[rand_perm].detach()[N_e:] @ proj
i_perm = utils.find_perm(torus_i, projected_i_neurons)[0]

shuffled_W = W_noisy[rand_perm][:, rand_perm]
recovered_W = torch.zeros_like(W_noisy)
recovered_W[:N_e, :N_e] = perm @ shuffled_W[:N_e, :N_e] @ perm.T
recovered_W[:N_e, N_e:] = perm @ shuffled_W[:N_e, N_e:] @ i_perm.T
recovered_W[N_e:, :N_e] = i_perm @ shuffled_W[N_e:, :N_e] @ perm.T
recovered_W[N_e:, N_e:] = i_perm @ shuffled_W[N_e:, N_e:] @ i_perm.T

print(f"Noise level: {noise_level}")

#%%
plt.figure(figsize=(2, 2), dpi=300)
plt.imshow(np.log(1+recovered_W), cmap='gray_r', vmin=0)
plt.xlim(0, 50)
plt.ylim(50, 0)
cbar = plt.colorbar(ticks=[0, np.log(2), np.log(11), np.log(101)])
# cbar =plt.colorbar()
cbar.ax.set_yticklabels([0, 1, 10, 100])
plt.tight_layout()

plt.figure(figsize=(2, 2), dpi=300)
plt.imshow(np.log(1+recovered_W), cmap='gray_r', vmin=0)
plt.axhline(N_e - 0.5, color='k', lw=0.25)
plt.axvline(N_e - 0.5, color='k', lw=0.25)
plt.xticks([N_e / 2, N_i / 2 + N_e], ['Exc.', 'Inh.'])
plt.yticks([N_e / 2, N_i / 2 + N_e], ['Exc.', 'Inh.'])
cbar = plt.colorbar(ticks=[0, np.log(2), np.log(11), np.log(101)])
# cbar =plt.colorbar()
cbar.ax.set_yticklabels([0, 1, 10, 100])
plt.tight_layout()

#%%
