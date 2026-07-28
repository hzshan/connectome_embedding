#%%
#TODO: create code for generating the embeddings and the null circularity coefficient distribution

import numpy as np
import matplotlib.pyplot as plt
import torch
from connectome_embedding import utils, plots, models
import pickle
from scipy import stats

plots.set_rc_params()

embedding_name = '../Learned_embeddings/embedding_Allhemibrain_D5_09262025.pkl'
model_and_data = pickle.load(open(embedding_name, 'rb'))
model = model_and_data['model']
data = model_and_data['data']
model.embeddings = model.embeddings.detach()

sorted_inds = utils.sort_embeddings(model.embeddings, data)
sorted_Adj = data.J[sorted_inds][:, sorted_inds]

CXembedding_name = '../Learned_embeddings/embedding_AllCX_D5_09232025.pkl'
CXmodel_and_data = pickle.load(open(CXembedding_name, 'rb'))
CXmodel = CXmodel_and_data['model']
CXdata = CXmodel_and_data['data']

def evaluate_circularity(model, data):
    embeddings_2d = []
    best_projs = []
    best_biases = []
    best_perms = []

    for i, type in enumerate(data.types):
        print(type)
        x = model.embeddings[data.neuron_hash[type]]

        found_circle, projs, perms = utils.solve_shadowmatic2(x, nseeds_to_try=50, max_iter=10)
        embeddings_2d.append(found_circle)
        best_perms.append(perms)
        best_projs.append(projs)

    circularity_via_norm = np.zeros(len(data.types))
    for i in range(len(data.types)):
        circularity_via_norm[i] = utils.circ_score_via_norm(embeddings_2d[i], skip_alignment=True)

    baseline = pickle.load(open('../baseline_circ_coef_D5.pkl', 'rb'))
    assert baseline['D'] == model.embeddings.shape[1]

    p_values = np.zeros(len(data.types))
    for i in range(len(data.types)):
        scores_from_random_points_same_N = baseline['circ_scores'][:, np.min([int(data.N_per_type[i] - 1), 148])]
        p_values[i] = 1 - stats.norm.cdf(circularity_via_norm[i], np.mean(scores_from_random_points_same_N), np.std(scores_from_random_points_same_N))

    return circularity_via_norm, p_values

def plot_circularity(data, circularity_via_norm, p_values, colors=None):
    argsort = np.argsort(circularity_via_norm)[::-1]
    plt.figure(figsize=(12, 1.5), dpi=300)

    for i in range(len(data.types)):
        text = ''
        if p_values[argsort][i] < 0.001:
            text = '***'
        elif p_values[argsort][i] < 0.01:
            text = '**'
        elif p_values[argsort][i] < 0.05:
            text = '*'

        plt.text(i + 0.5, circularity_via_norm[argsort][i] + 0.003 - 0.6, text, ha='center', va='bottom', fontsize=8, rotation=90)

    plt.xticks(rotation=90, fontsize=5)
    
    if colors is None:
        colors = 'gray'
    else:
        colors = np.array(colors)[argsort]
        
    plt.bar(np.array(data.types)[argsort], circularity_via_norm[argsort] - 0.6, color=colors)
    plt.yticks([0, 0.4], ['0.6', '1.0'])
    plt.ylim([0, 0.5])
    plt.xlim(-1, len(data.types) - 0.5)
    plt.tight_layout()

#%% Process CX Model
CX_circ, CX_p_values = evaluate_circularity(CXmodel, CXdata)
plot_circularity(CXdata, CX_circ, CX_p_values)

#%% Process Whole-hemibrain Model
data_circ, data_p_values = evaluate_circularity(model, data)
CX_types = CXdata.type_hash.keys()
celltype_colors = ['goldenrod' if t in CX_types else 'gray' for t in data.types]
plot_circularity(data, data_circ, data_p_values, colors=celltype_colors)