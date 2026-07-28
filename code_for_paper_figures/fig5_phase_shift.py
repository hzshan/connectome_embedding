

#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
from connectome_embedding import utils, plots
import pickle

plots.set_rc_params()

embedding_name = '../Learned_embeddings/embedding_AllCX_D5_LRsplit_woDelta7_031225.pkl'
model_and_data = pickle.load(open(embedding_name, 'rb'))
model = model_and_data['model']
data = model_and_data['data']

sorted_inds = utils.sort_embeddings(model.embeddings, data)
sorted_Adj = data.J[sorted_inds][:, sorted_inds]


#%% Summary plot of rotations
anti_sym = model.get_rotation_anti_sym().detach().numpy()
rot_forward_and_back_params = anti_sym + anti_sym.transpose(1, 0, 2, 3)


angles = np.zeros((data.Ntype, data.Ntype))

for i in range(data.Ntype):
    for j in range(data.Ntype):
        # angles[i, j] = np.arccos((np.trace(rot_mats[i, j]) - 1) / 2)
        angles[i, j] = np.linalg.norm(rot_forward_and_back_params[i, j]) / np.pi * 180


plt.figure(figsize=(8, 8))
plt.imshow(angles, cmap='Oranges')
plt.xticks(range(len(data.types)), data.types, rotation=90, fontsize=4)
plt.yticks(range(len(data.types)), data.types, rotation=0, fontsize=4)
plt.colorbar()
plt.title('A->B->A rotation')
plt.tight_layout()


#%% Connectivity between Delta7, EPG and PEN2
plt.figure(figsize=(2, 2))
plots.plot_J(sorted_Adj, data, ['Delta7', 'EPG_R', 'PEN_b(PEN2)_R'], ax=plt.gca())
cbar = plt.colorbar(ticks=[0, np.log(2), np.log(11), np.log(101)])
cbar.ax.set_yticklabels([0, 1, 10, 100])
plt.tight_layout()

#%% Phase plots of connectivity between cell types

epg_to_epg = sorted_Adj[data.neuron_hash['EPG_R'], :][:, data.neuron_hash['EPG_R']]
epg_to_pen = sorted_Adj[data.neuron_hash['EPG_R'], :][:, data.neuron_hash['PEN_b(PEN2)_R']]
pen_to_epg = sorted_Adj[data.neuron_hash['PEN_b(PEN2)_R'], :][:, data.neuron_hash['EPG_R']]
epg_to_delta7 = sorted_Adj[data.neuron_hash['EPG_R'], :][:, data.neuron_hash['Delta7']]
delta7_to_pfl = sorted_Adj[data.neuron_hash['Delta7'], :][:, data.neuron_hash['PFL3_R']]
epg_to_pfl = sorted_Adj[data.neuron_hash['EPG_R'], :][:, data.neuron_hash['PFL3_R']]
delta7_to_pen = sorted_Adj[data.neuron_hash['Delta7'], :][:, data.neuron_hash['PEN_b(PEN2)_R']]

epg_to_pen_to_epg = epg_to_pen @ pen_to_epg
epg_to_delta7_to_pen_to_epg = epg_to_delta7 @ delta7_to_pen @ pen_to_epg

epg_to_delta7_to_pfl = epg_to_delta7 @ delta7_to_pfl

def unroll_circulant(A):
    n = A.shape[0]
    unrolled_A = torch.zeros_like(A)
    for i in range(n):

        unrolled_A[i, :] = torch.roll(A[i, :], shifts=-i + int(n / 2))
    return unrolled_A

def plot_phase_diff(mat):
    mat = unroll_circulant(mat)
    mat /= mat.max()
    n = mat.shape[1] if mat.shape[0] != mat.shape[1] and mat.shape[0] > mat.shape[1] else mat.shape[0] 
    # Use logic for x based on how many phases we have
    if mat.shape[0] == mat.shape[1] or mat.shape[0] == 46:
        n = mat.shape[0]
    else:
        n = mat.shape[1]
        
    x = np.linspace(-np.pi, np.pi, n, endpoint=True)  # equally spaced angles for unrolling
    plt.figure(figsize=(2, 1), dpi=300)
    plt.errorbar(x, mat.mean(0), yerr=mat.std(0) / np.sqrt(n))
    plt.axvline(0, color='k')

    plt.xticks([-np.pi, -np.pi/2, 0, np.pi / 2, np.pi], ['$-\\pi$', '$-\\pi / 2$', '0', '$\\pi / 2$', '$\\pi$'])
    plt.xlabel('phase difference (rads)', fontsize=6)
    plt.ylabel('a.u.', fontsize=6)
    plt.tight_layout()

for mat in [epg_to_epg, epg_to_pen_to_epg, epg_to_delta7_to_pen_to_epg, epg_to_pfl, epg_to_delta7_to_pfl]:
    plot_phase_diff(mat)

# %%
