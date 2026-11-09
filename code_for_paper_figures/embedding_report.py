#%%
"""
Settings and computational cost of the embeddings.

1. Hyperparameter table: the settings stored in each embedding pickle used in the
   paper (as saved by embed_hemibrain.py / embed_flywire_cave.py), one row per
   embedding. Null models are summarized as one row with their seeds.
2. Benchmark (hemibrain only, one machine): time per training step and peak memory
   for the data of each hemibrain config, and for random subsets of the whole-hemibrain
   neurons of increasing size (scaling with the number of neurons N). Each run builds
   the data and model as embed_hemibrain.py does and trains for BENCHMARK_STEPS steps,
   after WARMUP_STEPS steps that are not timed (the first call compiles the loss).
   Configs that fit rotations also time the rotation stage. Total time of a full fit
   is estimated as time per step x the steps of the config.
   Peak memory: peak resident memory of this process from building the model to the
   end of the timed steps, minus the resident memory just before (sampled every
   MEMORY_SAMPLE_S seconds). Freed memory may stay with the process and be reused,
   so runs go from the smallest to the largest data; values are approximate.

Outputs (CSV) in RESULTS_DIR. Run from code_for_paper_figures/.
"""
import os, gc, glob, time, threading, platform, pickle, subprocess
import numpy as np
import pandas as pd
import psutil
import torch
import matplotlib.pyplot as plt
from connectome_embedding import models, data_utils, plots
import embedding_configs

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
EMBEDDING_DIR = 'embeddings/'
RESULTS_DIR = 'results/embedding_report/'
DATAPATH = 'hemibrain_data/'

# Embeddings used in the paper: label -> pickle in EMBEDDING_DIR
PAPER_EMBEDDINGS = {
    'hemibrain CX': 'CX_embedding.pkl',
    'hemibrain CX, L/R split': 'CX_LRsplit_embedding.pkl',
    'whole hemibrain': 'whole_hemibrain_embedding.pkl',
    'hemibrain visual projection': 'visual_projection_embedding.pkl',
    'FlyWire CX': 'flywire_CX_cave_embedding.pkl',
    'FlyWire CX, L/R split': 'flywire_CX_LRsplit_embedding.pkl',
}
# Null models: label -> glob pattern in EMBEDDING_DIR (one row per pattern)
NULL_EMBEDDINGS = {
    'hemibrain CX, block-shuffled': 'CX_embedding_block_shuffled_seed*.pkl',
    'whole hemibrain, block-shuffled': 'whole_hemibrain_embedding_block_shuffled_seed*.pkl',
}

# Benchmark
BENCHMARK_CONFIGS = ['central_complex', 'central_complex_LR_split', 'whole_hemibrain',
                     'visual_projection']
SUBSET_SIZES = [500, 1000, 2000, 4000]   # plus the full whole-hemibrain data
SUBSET_SEED = 0
BENCHMARK_STEPS = 200
WARMUP_STEPS = 20
N_THREADS = 8                  # torch CPU threads, fixed for comparable timings
MEMORY_SAMPLE_S = 0.05

os.makedirs(RESULTS_DIR, exist_ok=True)
plots.set_rc_params()


#%% 1. Hyperparameter table
def hyperparameter_row(label, path, n_fits=1, seeds=None):
    loaded = pickle.load(open(path, 'rb'))
    hp, data = loaded['hyperparams'], loaded['data']
    split = hp.get('split_LR')
    return {
        'embedding': label,
        'file': os.path.basename(path) if n_fits == 1 else os.path.basename(path).split('_seed')[0] + '_seed*',
        'fits': n_fits,
        'seeds': hp['seed'] if seeds is None else ', '.join(str(s) for s in seeds),
        'neurons': data.N,
        'cell types': data.Ntype,
        'min neurons per type': hp['min_num_per_type'],
        'L/R split': 'all types' if split is None else ('no' if len(split) == 0 else 'yes'),
        'D': hp['embedding_dim'],
        'loss': hp['loss_type'],
        'wrapper': hp['wrapper_fn'],
        'embedding regularizer': hp['embedding_regularizer'],
        'learning rate': hp['lr'],
        'steps': hp['steps'],
        'rotations': hp['fit_rotations'],
        'rotation steps': hp['rotation_steps'] if hp['fit_rotations'] else '',
        'rotation regularizer': hp['rotation_regularizer'] if hp['fit_rotations'] else '',
        'weights shuffled within blocks': hp.get('shuffle_within_blocks', False),
    }


rows = []
for label, filename in PAPER_EMBEDDINGS.items():
    path = os.path.join(EMBEDDING_DIR, filename)
    if os.path.isfile(path):
        rows.append(hyperparameter_row(label, path))
    else:
        print(f'missing: {label} ({filename})')
for label, pattern in NULL_EMBEDDINGS.items():
    paths = sorted(glob.glob(os.path.join(EMBEDDING_DIR, pattern)))
    if paths:
        seeds = [pickle.load(open(p, 'rb'))['hyperparams']['seed'] for p in paths]
        rows.append(hyperparameter_row(label, paths[0], n_fits=len(paths), seeds=seeds))
    else:
        print(f'missing: {label} ({pattern})')
hyperparameters = pd.DataFrame(rows)
pd.set_option('display.width', 250)
print(hyperparameters.T.to_string(header=False))
hyperparameters.to_csv(os.path.join(RESULTS_DIR, 'hyperparameters.csv'), index=False)


#%% 2. Benchmark functions
def machine_info():
    try:
        cpu = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        cpu = ''
    return {'cpu': cpu or platform.processor(), 'cores': os.cpu_count(),
            'RAM (GB)': round(psutil.virtual_memory().total / 1e9),
            'torch threads': torch.get_num_threads(), 'torch': torch.__version__,
            'device': 'cpu'}


class PeakMemory:
    """Peak resident memory of this process while in the with-block, above the
    resident memory at entry (bytes)."""
    def __enter__(self):
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.running = True
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _sample(self):
        while self.running:
            self.peak = max(self.peak, self.process.memory_info().rss)
            time.sleep(MEMORY_SAMPLE_S)

    def __exit__(self, *args):
        self.running = False
        self.thread.join()
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.extra = self.peak - self.baseline


def train(model, data, hp, steps, reg_fn):
    models.train_model(model, target_mat=data.J, loss_type=hp['loss_type'], lr=hp['lr'],
                       print_every=10 ** 9, reg_fn=reg_fn, steps=steps)


def timed_training(model, data, hp, reg_fn):
    """Seconds per step over BENCHMARK_STEPS steps, after WARMUP_STEPS untimed steps."""
    train(model, data, hp, WARMUP_STEPS, reg_fn)
    start = time.perf_counter()
    train(model, data, hp, BENCHMARK_STEPS, reg_fn)
    return (time.perf_counter() - start) / BENCHMARK_STEPS


def benchmark(data, hp, label):
    """Time the embedding stage (and the rotation stage if the config fits rotations)."""
    gc.collect()

    def reg_fn(m):
        return torch.mean(m.embeddings ** 2) * hp['embedding_regularizer']

    def reg_fn_rotations(m):
        return (torch.mean(m.rotation_params ** 2) * hp['rotation_regularizer']
                + torch.mean(m.translation_vecs ** 2) * hp['translation_regularizer']
                + torch.mean(m.embeddings ** 2) * hp['embedding_regularizer'])

    with PeakMemory() as memory:
        torch.manual_seed(hp['seed'])
        model = models.InteractionModel(N=data.N, Ntype=data.Ntype, D=hp['embedding_dim'],
                                        onehot_types=data.onehot_types, wrapper_fn=hp['wrapper_fn'])
        model.use_transforms = False
        per_step = timed_training(model, data, hp, reg_fn)
        if hp['fit_rotations']:
            model.embeddings.requires_grad = False
            model.use_transforms = True
            rot_per_step = timed_training(model, data, hp, reg_fn_rotations)
    row = {'run': label, 'neurons': data.N, 'cell types': data.Ntype,
           's per step': per_step, 'peak extra memory (GB)': memory.extra / 1e9,
           'steps': hp['steps'], 'estimated time (min)': per_step * hp['steps'] / 60}
    if hp['fit_rotations']:
        row.update({'rotation s per step': rot_per_step, 'rotation steps': hp['rotation_steps']})
        row['estimated time (min)'] += rot_per_step * hp['rotation_steps'] / 60
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
    return row


def config_data(hp):
    W_all, neuronsall = data_utils.get_W_all_neuronall(DATAPATH, min_num_per_type=hp['min_num_per_type'])
    return data_utils.prep_connectivity_data(W_all, neuronsall, hp['types_wanted'],
                                             split_LR=hp['split_LR'])


def neuron_subset(data, n, rng):
    """The connectivity among n randomly chosen neurons (types with no chosen neuron drop out)."""
    keep = np.sort(rng.choice(data.N, n, replace=False))
    J = data.J[keep][:, keep]
    neurons = data.neurons.iloc[keep].reset_index(drop=True)
    return data_utils.ConnectivityData(J, neurons, 'type')


#%% 2. Benchmark: data of each hemibrain config
torch.set_num_threads(N_THREADS)
machine = machine_info()
print(machine)
configs = {name: embedding_configs.get_config(name) for name in BENCHMARK_CONFIGS}
datasets = {name: config_data(hp) for name, hp in configs.items()}
config_rows = [benchmark(datasets[name], configs[name], name)        # smallest data first
               for name in sorted(datasets, key=lambda name: datasets[name].N)]
del datasets
config_table = pd.DataFrame(config_rows).assign(**machine)
print(config_table.round(3).to_string(index=False))
config_table.to_csv(os.path.join(RESULTS_DIR, 'benchmark_configs.csv'), index=False)

#%% 2. Benchmark: scaling with the number of neurons (subsets of the whole hemibrain)
# Memory of small subsets is underestimated if larger runs came before in the same
# session (the process keeps freed memory); for clean values, run this cell first in
# a fresh session (after the cells defining the functions).
hp = embedding_configs.get_config('whole_hemibrain')
whole = config_data(hp)
rng = np.random.default_rng(SUBSET_SEED)
scaling_rows = [benchmark(neuron_subset(whole, n, rng), hp, f'subset {n}')
                for n in SUBSET_SIZES if n < whole.N]
scaling_rows.append(benchmark(whole, hp, 'whole hemibrain'))
scaling = pd.DataFrame(scaling_rows).assign(**machine)
print(scaling.round(4).to_string(index=False))
scaling.to_csv(os.path.join(RESULTS_DIR, 'benchmark_scaling.csv'), index=False)

#%% Plot: time per step and memory vs number of neurons, with an N^2 reference line
fig, axes = plt.subplots(1, 2, figsize=(4, 2), dpi=300)
for ax, column, ylabel in [(axes[0], 's per step', 'time per step (s)'),
                           (axes[1], 'peak extra memory (GB)', 'peak memory (GB)')]:
    ax.loglog(scaling['neurons'], scaling[column], 'o-', color='k', ms=2, lw=0.8)
    reference = scaling[column].iloc[-1] * (scaling['neurons'] / scaling['neurons'].iloc[-1]) ** 2
    ax.loglog(scaling['neurons'], reference, color='gray', lw=0.5, ls='--', label='N$^2$')
    ax.set_xlabel('neurons N', fontsize=6)
    ax.set_ylabel(ylabel, fontsize=6)
    ax.legend(fontsize=5, frameon=False)
fig.suptitle(f"{machine['cpu']}, {N_THREADS} threads, CPU", fontsize=5)
plt.tight_layout()
