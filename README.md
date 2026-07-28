This is the repository for [Graph embeddings for identifying symmetries in connectomes (Shan and Litwin-Kumar, 2026)](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=6CF1B8oAAAAJ&sortby=pubdate&citation_for_view=6CF1B8oAAAAJ:5nxA0vEk-isC). 

Created by [Haozhe Shan](hzshan.github.io). Some code for loading and processing connectomes is based on those from https://github.com/alitwinkumar/connectome_tools. 

* `/toolkit/` contains code for loading and analyzing connectome data.
* `/code_for_paper_figures/` contains Jupyter notebooks for recreating the main figures. 


### Setup

`cd` into `toolkit/` and run `pip install -e .` before using the notebooks.

### Data

The main figures make heavy use of the [hemibrain connectome](https://www.janelia.org/project-team/flyem/hemibrain) of the *Drosophila* brain. It can be downloaded [here](https://storage.cloud.google.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz). The files needed are `traced-neurons.csv` and `traced-total-connections.csv`, which contain information about neurons as well as the number of synapses between them.

In addition, the analysis of visual projection neurons (Fig. 6) makes use of spatial coordinates of individual synapses to estimate the receptive fields of neurons. This is a substantial amount of data and can be downloaded with NeuPrint (#TODO).

Finally, the analysis of grid cells makes use of a synthetic medial entorhinal cortex (MEC) connectome. The code for generating it is included in the files for making the relevant figures.


### Workflow

Obtaining the embedding is the most computationally heavy part of the analysis pipeline (but it is doable in a matter of minutes on a laptop for connectomes of thousands of neurons). Therefore, the code is organized such that one script learns the embedding and save it as a pickle file and the figure notebooks make use of it.