

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
# For each pair of cell types (A, B), how far does an embedding get rotated if we
# go A -> B -> A? The model has one rotation per (source type, target type),
# indexed rotation_mats[A, B], so the round trip is the product R_AB @ R_BA.
# (Composing rotations is not the same as adding their generators unless the
# rotations commute, so the angle is read off the product itself.)

# All rotation matrices, [source type, target type]: Ntype x Ntype x D x D
rot_mats = model.get_rotation_mats().detach().numpy()

def rotation_angles_deg(R):
    """
    Rotation angles of a rotation matrix R in SO(D), in degrees, largest first.

    A D-dim rotation turns vectors in up to floor(D/2) independent planes. Its
    eigenvalues come in pairs exp(+i theta_k), exp(-i theta_k) (plus a 1 when D
    is odd), so each plane's angle theta_k is the absolute phase of one pair.
    """
    eigvals = np.linalg.eigvals(R)
    # absolute phase of each eigenvalue, in [0, 180] degrees
    angles = np.degrees(np.abs(np.angle(eigvals)))
    # sort largest first; each angle appears twice (as +theta and -theta),
    # so take every other one to get one value per plane
    return np.sort(angles)[::-1][::2]

def total_rotation_deg(R):
    """
    One overall size for a rotation: sqrt(theta_1^2 + theta_2^2 + ...), in degrees.

    This is the geodesic distance from the identity to R in SO(D), i.e. the
    length of the shortest path of rotations that gets from "no rotation" to R.
    For a rotation in a single plane by theta it is just theta. (The eigenvalues
    themselves all have magnitude 1, so it is their phases that carry the size.)
    """
    return np.sqrt(np.sum(rotation_angles_deg(R)**2))

# Overall rotation angle of the round trip, and of the one-way rotation, per type pair
angles = np.zeros((data.Ntype, data.Ntype))          # A -> B -> A
one_way_angles = np.zeros((data.Ntype, data.Ntype))  # R_AB alone

for i in range(data.Ntype):
    for j in range(data.Ntype):
        # compose the two rotations (the order doesn't change the angles:
        # R_AB R_BA and R_BA R_AB have the same eigenvalues)
        round_trip = rot_mats[i, j] @ rot_mats[j, i]
        angles[i, j] = total_rotation_deg(round_trip)
        one_way_angles[i, j] = total_rotation_deg(rot_mats[i, j])


plt.figure(figsize=(8, 8))
plt.imshow(angles, cmap='Oranges')
plt.xticks(range(len(data.types)), data.types, rotation=90, fontsize=4)
plt.yticks(range(len(data.types)), data.types, rotation=0, fontsize=4)
plt.colorbar()
plt.title('A->B->A rotation angle (deg)')
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
