import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from scipy.io import loadmat, savemat
from scipy.spatial.distance import pdist, squareform
from networkx.algorithms.community import modularity

from behavior_extraction import label_tpoints
from dependencies import root



def sort_by_distance(D):
    """


    Parameters
    ----------
    D : array-like of shape (n_neurons, n_neurons)
        Distance matrix

    Returns
    -------
    sort_idc : array-like of shape (n_neurons, n_neurons)
        Matrix in which the ith row is an array of indices
        that sort neurons other than neuron i from closest to farthest
        based on the distance between the neural time series.

    """

    seq_ids = np.arange(D.shape[0])
    np.fill_diagonal(D, D.max()+1)
    sort_idc = np.array([seq_ids[np.argsort(row)] for row in D])
    
    return sort_idc
        
def create_knn_graph(
    sort_idc,
    n_neighbors=10
):
    """


    Parameters
    ----------
    sort_idc : array-like of shape (n_neurons, n_neurons)
        Matrix in which the ith row is an array of indices
        that sort neurons other than neuron i from closest to farthest
        based on the distance between the neural time series.
        
    n_neighbors : int
        Number of nearest neighbors

    Returns
    -------
    adj_mat : array-like of shape (n_neurons, n_neurons)
        Adjacency matrix representing the undirected, unweighted knn graph
    """

    if n_neighbors < sort_idc.shape[0]-1:
        adj_mat = np.zeros(sort_idc.shape)
        for row_ids, row_adj in zip(sort_idc, adj_mat):
            row_adj[row_ids[:n_neighbors]] = 1
    else:
        adj_mat = np.ones(sort_idc.shape)
        np.fill_diagonal(adj_mat, 0)
    
    return adj_mat

def save_nearest_neighbors(
    ephys_data,
    behav_data,
    names,
    path
):
    """


    Parameters
    ----------
    ephys_data : list
        Dictionaries containing the processed neural activity and neuron locations
    behav_data : list 
        Dictionaries containing the behavioral variables extracted from videos
    names : list 
        Names of mice
    path : str
        Path to where the results are saved

    """

    dat = dict()
    for imouse in range(len(names)):
        spkmat = ephys_data[imouse]["spkmat"]
        T_neither, T_whisk_only, T_lomot_only, T_both = label_tpoints(ephys_data, behav_data, mouseID=imouse)
        sort_ids = [sort_by_distance(squareform(pdist(spkmat[:, T], metric="correlation"))) for T in [T_neither, T_whisk_only, T_both]]
        dat[names[imouse]] = sort_ids
    
    if path == '':
        path = f"{root}/Data/save/nearest_neighbors_sort_indices.mat"
    savemat(path, dat)


def plot_modularity_vs_knn(
    ephys_data,
    behav_colors,
    names,
    nn_vals,
    plot_width=4,
    plot_height=5,
    path=''
):
    """


    Parameters
    ----------
    ephys_data : list
        Dictionaries containing the processed neural activity and neuron locations
    behav_data : list 
        Dictionaries containing the behavioral variables extracted from videos
    names : list 
        Names of mice
    nn_vals : list
        List of values of number of nearest neighbors
    plot_width : float
        Width of indidividual plots
    plot_height : float
        Height of individual plots
    path : str
        Path to where the figure is saved

    """
    
    # load saved nearest-neighbor indices 
    nn_ids_dict = loadmat(f"{root}/Data/save/nearest_neighbors_sort_indices.mat")
    
    # plotting variables (e.g. ticks, axis limits, colormaps, ...)
    beh_lbs = ["Neither", "Whisking Only", "Both"]
    colors = [behav_colors[0], behav_colors[1], behav_colors[3]]
    ytks = np.round(np.arange(0, 0.61, 0.2), 1)

    # initialize and adjust axes
    n_mice = len(names)
    fig, axs = plt.subplots(1, n_mice, figsize=(n_mice*plot_width, plot_height))
    fig.subplots_adjust(wspace=0.1)
    
    # main
    for imouse, name, ax, ephys in zip(range(n_mice), names, axs, ephys_data):
        regIDs = ephys["regIDs"]
        nn_ids_imouse = nn_ids_dict[name]
        
        # partitions of graph nodes
        communities = [set(np.argwhere(regIDs==rid).flatten()) for rid in np.unique(regIDs)]

        for nn_ids, clr, lb in zip(nn_ids_imouse, colors, beh_lbs):
            modularities = [modularity(nx.from_numpy_array(create_knn_graph(sort_ids=nn_ids, n_neighbors=nn)), communities) for nn in nn_vals]
            ax.plot(nn_vals, modularities, color=clr, label=lb)
        ax.set_ylim((0, 0.6))
        ax.set_yticks(ytks)
        ax.set_yticklabels(ytks if imouse==0 else [])
        ax.set_title(name)

        ax.set_xscale("log")
        if name == names[0]: ax.set_ylabel("Modularity")
        if name == names[1]: ax.set_xlabel("Number of nearest neighbors")
    
    ax.legend(
        prop=dict(size=14),
        framealpha=0
    )
            
    if path == '':
        path = f"{root}/Plots/modularity.png"

    plt.savefig(path, bbox_inches="tight", transparent=True)




