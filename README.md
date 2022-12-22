# NeuropixelSpont
Analyses of multi-brain Neuropixels recordings of mice during spontaneous behaviors.

All results were obtained using the publicly available [dataset](https://janelia.figshare.com/articles/dataset/Eight-probe_Neuropixels_recordings_during_spontaneous_behaviors/7739750) consisting of raw eloctrophysical recordings and video footages of three mice moving in the dark. To reproduce the figures, follow these steps:
1. Download the preprocessed data from [here](https://drive.google.com/drive/folders/1f75fpMnQdRJF6JPe0hYDVkou8hYJaR-4?usp=share_link).
2. Clone this repository.
3. In the local Python folder, create a Python file named `dependencies.py` then declare two variables `data_path` and `figure_path` that contain respetive paths to the preprocessed data (step 1) and a folder to hold all the figures.
4. Run `produce_all_plots.py` to produce all the figures. Details about the code are provided below.

`load_data` reads the processed data from the `.mat` files and package them into Python dictionaries for easy access. Specifically, `ephys_data` is a list of dictionaries containing the following information:
- `'spkmat'` : matrix of shape (n_neurons, n_time_points) in which each row contained the filtered spiking activity of a single neuron
- `'tpoints'` : time points of the filtered spiking activity (length=n_time_points)
- `'regIDs'` : brain region indices assigned to the neurons
- `'reglbs'` : names of brain regions corresponding to the indices
- `'coords'` : 3D anatomical location of neurons in Allen Common Coordinate Framework (CCF)
- `'probeIDs'` : ID of the probes used to record the neurons
- `'heights'` : vertical position of neurons on the associated probes

Behavior variables extracted from the videos are stored in `behav_data`, including:
- `'whisk_cont'`, `'lomot_cont'` : $L^2$-norms of pixel change within the regions of interest (ROIs)
- `'whisk_dist'`, `'lomot_disc'` : time series of behavioral event indicators
- `'time_pts'` : time points of the behavioral time series

Color assignments for mice, behavioral states and brain regions are generated by `generate_colors` to ensure consistency between plots. `ephys_data`, `behav_data` and the color variables are the main inputs into the functions which performed analyses on the processed data and plot the results. The functions are divided into following modules:

#### [About Dataset](https://github.com/trannttoan/NeuropixelSpont/blob/main/Python/dataset.py)
- `plot_neuron_positions` : plot anatomical locations of neurons in Allen Common Coordinnate Framework
- `plot_probe_positions` : plot positions Neuropixels probes in Allen Common Coordinnate Framework
- `plot_raster` : plot heatmaps of neural time series grouped by brain regions
- `plot_region_sample_sizes` : plot the sample sizes of brain regions for each mouse
- `plot_time_wrt_behavior` : plot percent of time spent in each behavioral state

#### [Auxiliary Functions](https://github.com/trannttoan/NeuropixelSpont/blob/main/Python/helper_functions.py)
- `plot_config` : set default parameters for matplotlib
- `generate_colors` : generate colors for mice, behavioral states and brain regions to be used across all plots
- `load_data` : load processed data into dictionaries for easy access

#### [Behavior Variables](https://github.com/trannttoan/NeuropixelSpont/blob/main/Python/behavior_extraction.py)
- `label_tpoints` : align behavioral event indicators with neural recording time and create a mask for each behavioral state
- `plot_behavior_variables` : plot example frames along with ROIs and corresponding behavior variables

#### [Clustering Analysis](https://github.com/trannttoan/NeuropixelSpont/blob/main/Python/clustering_analysis.py)
- `plot_hclust_vs_behavior` : perform correlation-distance-based hierarchical clustering on each of the behavior-based partition of population activity and plot distance matrices sorted based on the resulting dendrograms
- `plot_cophenetic_vs_behavior` : perform correlation-distance-based hierarchical clustering on each of the behavior-based partition of population activity and plot the cophenetic coefficients of the resulting dendrograms
- `compute_silhouette_vs_behavior` : compute Silhouette coefficients based on brain region labels and correlation distance between neural time series for each behavioral state
- `plot_silhouette_vs_behavior` : create Silhouette plots for the behavior-based partitions of population activity (Silhouette coefficients are computed based on brain region labels and correaltion distance between segmented neural time series)
- `plot_silhouette_kstest` : perform kolmogorov-smirnov test on pairs of Silhouette samples and plot the ECDFs along with the resulting KS statistics

#### [Graph-theoretic Metrics](https://github.com/trannttoan/NeuropixelSpont/blob/main/Python/knn_graph)
- `find_nearest_neighbors` : for each neuron, sort other neurons in ascending order based on the correlation distance between neural time series
- `plot_modularity_vs_nneighbors` : plot network modularity versus the number of nearest neighbor when generating k-nearest-neighbor graph based on correlation distance between neural time series (saved results of find_nearest_neighbors are required)

#### [Dimensionality Reduction](https://github.com/trannttoan/NeuropixelSpont/blob/main/Python/umap_embeddings.py)
- `create_umap_vs_nneighbors` : generate the given number of UMAP embeddings for each of the given numbers of nearest neighbors
- `plot_umap_vs_nneighbors` : plot the UMAP embeddings generated by the create_umap_vs_nneighbors function
- `plot_silhouette_vs_nneighbors` : plot mean Silhouette standard deviation of UMAP embeddings versus the number of nearest neighbors. For each neuron, the standard deviation of Silhouette coefficients is computed across UMAP embebdings with the same number of nearest neighbors (saved results of the create_umap_vs_nneighbors are required)



