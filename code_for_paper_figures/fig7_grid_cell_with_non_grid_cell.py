#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
from connectome_embedding import utils, models, grid_cell, plots

plots.set_rc_params()


# Base synthetic network parameters
n_e = 25
n_i = 16
N_e = n_e**2
N_i = n_i**2
full_N = N_e + N_i
distractor_ratio = 2.0  # num distractors/num grid cells

lambda_w = n_e / 128 * 26
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

np.random.seed(0)
torch.manual_seed(0)

n_distractors = int(full_N * distractor_ratio)
# connections for distractors are random samples from the original W, which preserves the distribution of weights but breaks the structure
flat_W = full_W.flatten()

# all distractors are counted as excitatory neurons
type_onehot_distractors = torch.zeros((full_N + n_distractors, 2))
type_onehot_distractors[full_N:, 0] = 1
type_onehot_distractors[:full_N] = type_onehot

def reg_fn(model):
    return (
        torch.mean(model.rotation_params**2) * 1000
        + torch.mean(model.translation_vecs**2)
        + torch.mean(model.embeddings**2) * 0.1
    )

noise_levels = np.linspace(0.1, 5.0, 5)
n_repeats_per_level = 3
train_steps = 1000
nseeds_torus = 10

torus_e = grid_cell.make_clifford_torus(n_e).float()
torus_i = grid_cell.make_clifford_torus(n_i).float()

results = []
for noise_level in noise_levels:
    for repeat_idx in range(n_repeats_per_level):
        seed = int(round(noise_level * 1000)) * 100 + repeat_idx
        np.random.seed(seed)
        torch.manual_seed(seed)

        signal_mean = torch.mean(full_W.flatten())
        n_distractors = int(full_N * noise_level)
        flat_W = full_W.flatten()
        W_with_distractors = flat_W[torch.randint(0, len(flat_W), ((full_N + n_distractors), (full_N + n_distractors)))]
        W_with_distractors[:full_N, :full_N] = full_W

        type_onehot_distractors = torch.zeros((full_N + n_distractors, 2))
        type_onehot_distractors[full_N:, 0] = 1
        type_onehot_distractors[:full_N] = type_onehot

        model = models.InteractionModel(
            N=W_with_distractors.shape[0],
            Ntype=2,
            D=7,
            onehot_types=type_onehot_distractors,
        )
        model.use_transforms = False

        models.train_model(
            model,
            W_with_distractors,
            lr=0.01,
            print_every=1000,
            reg_fn=reg_fn,
            steps=train_steps,
            loss_type='poisson',
        )

        results.append({
            'noise_level': float(noise_level),
            'repeat': repeat_idx,
            'onehot': type_onehot_distractors,
            'full_W': full_W,
            'W_with_distractors': W_with_distractors,
            'pred_W': model.pred_synapses().detach(),
        })

#%% Plot histograms of relative error per neuron across noise levels
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

fig, axes = plt.subplots(5, 3, figsize=(6, 6), sharey=True)
axes = axes.flatten()

manual_thresholds = [1.0, 1.0, 1.0, 1.35, 1.35, 1.35, 1.45, 1.45, 1.45, 1.47, 1.47, 1.47, 1.49, 1.49, 1.49]
inferred_grid_cells_across_noise_levels = []
for i, result in enumerate(results):
    error = torch.abs(result['pred_W'] - result['W_with_distractors'])

    rel_error = error / (1e-10 + result['W_with_distractors'])

    relative_error_per_neuron = (rel_error.mean(0) + rel_error.mean(1)) / 2

    relative_error_per_neuron = (error.mean(0) + error.mean(1)) / 2 / result['W_with_distractors'].mean()
    inferred_grid_cells = np.where(relative_error_per_neuron < manual_thresholds[i])[0]

    inferred_grid_cells_across_noise_levels.append(inferred_grid_cells)

    ax = axes[i]
    ax.hist(relative_error_per_neuron, bins=50)
    ax.axvline(x=manual_thresholds[i], color='r', linestyle='--')
    ax.set_xticks(np.arange(1, 2, 0.2))

    _data = np.sort(relative_error_per_neuron.numpy())

    kde = gaussian_kde(_data, bw_method='scott')  # Adjust bw_method (e.g., 0.2) to control smoothness

    # Evaluate KDE over a dense grid
    grid = np.linspace(_data.min() - 1, _data.max() + 1, 1000)
    density = kde(grid)

    valley_indices, _ = find_peaks(-density)

    # Plot the KDE
    # ax.axvline(x=grid[valley_indices].max(), color='g', linestyle='--', label='KDE valley')

    if 'noise_level' in result:
        ax.set_title(f'Abundance of non-grid cells: {result["noise_level"]:.2f}', fontsize=8)

plt.tight_layout()


#%% calculate false positive/negative rates
fp_rates = []
fn_rates = []

noise_levels_list = []
for i, inferred_grid_cells in enumerate(inferred_grid_cells_across_noise_levels):
    noise_levels_list.append(results[i]['noise_level'])
    total_N = results[i]['W_with_distractors'].shape[0]
    distractors = total_N - full_N
    real_grid_cells = full_N
    false_positive_rate = np.sum(inferred_grid_cells >= full_N) / distractors
    inferred_non_grid_cells = set(np.arange(total_N)) - set(inferred_grid_cells)
    false_negative_rate = np.sum(np.array(list(inferred_non_grid_cells)) < full_N) / real_grid_cells
    fp_rates.append(false_positive_rate * 100)
    fn_rates.append(false_negative_rate * 100)

fp_rates = np.array(fp_rates)
fn_rates = np.array(fn_rates)
noise_levels_list = np.array(noise_levels_list)

unique_noise_levels = np.unique(noise_levels_list)
mean_fp = []
std_fp = []
mean_fn = []
std_fn = []

for nl in unique_noise_levels:
    mask = np.isclose(noise_levels_list, nl)
    mean_fp.append(fp_rates[mask].mean())
    std_fp.append(fp_rates[mask].std())
    mean_fn.append(fn_rates[mask].mean())
    std_fn.append(fn_rates[mask].std())

plt.figure(figsize=(2, 2))
plt.errorbar(unique_noise_levels, mean_fp, yerr=std_fp, fmt='o-', label='False positive rate', capsize=2, ms=3, color='k')
plt.errorbar(unique_noise_levels, mean_fn, yerr=std_fn, fmt='o-', label='False negative rate', capsize=2, ms=3, color='gray')
plt.xlabel('Noise level (distractor ratio)')
plt.ylabel('Rate (%)')
plt.legend(frameon=False, fontsize=6)
plt.yticks([0, 1, 2])
plt.tight_layout()

# plt.savefig(os.path.join(figure_path, 'grid_cell_distractor_analysis.svg'), dpi=300)