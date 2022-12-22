from dataset import *
from helper_functions import *
from behavior_extraction import *
from clustering_analysis import *
from umap_embeddings import *
from knn_graph import *


## ephys_data : list of dictionaries, each containing the processed neural activity and neuron locations of a mouse.
## behav_data : list of dictionaries, each containing the behavioral variables extracted from the video recording of a mouse.
## names      : list of names of mice.
ephys_data, behav_data, names = load_data()

## mice_colors   : list of colors, one for each mouse.
## region_colors : list of colors, one for each brain region.
## behav_colors  : list of colors, one for each behavioral state.
mice_colors, region_colors, behav_colors = generate_colors()

# Customize default matplotlib parameters
plot_config()

# dataset
plot_neuron_positions(ephys_data, region_colors)
plot_probe_positions(names, mice_colors)
plot_region_sample_sizes(ephys_data, names)
plot_time_wrt_behavior(ephys_data, behav_data, names, behav_colors)

# behavior variables
plot_behavior_variables(behav_data, 0, behav_colors, tstart=200)

# clustering analysis
plot_cophenetic_vs_behavior(ephys_data, behav_data, names)
plot_silhouette_kstest(ephys_data, behav_data, names, behav_colors)

# graph-theoretic metric
plot_modularity_vs_nneighbors(ephys_data, names, behav_colors, np.arange(5, 101, 5))

# dimensionality reduction
plot_umap_vs_nneighbors(ephys_data, names, region_colors)
plot_silhouette_vs_nneighbors(ephys_data, names, mice_colors)


for imouse in range(len(names)):
    # dataset
    plot_raster(ephys_data, imouse)
    
    # clustering analysis
    plot_hclust_vs_behavior(ephys_data, behav_data, names, region_colors, mouseID=imouse)
    plot_silhouette_vs_behavior(ephys_data, behav_data, names, imouse, region_colors)


