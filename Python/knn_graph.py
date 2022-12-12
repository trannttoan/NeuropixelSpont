import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from scipy.io import loadmat, savemat
from scipy.spatial.distance import pdist, squareform
from networkx.algorithms.community import modularity

from behavior_extraction import label_tpoints
from dependencies import root



def sort_by_distance(D):
    seq_ids = np.arange(D.shape[0])
    np.fill_diagonal(D, D.max()+1)
    sort_ids = np.array([seq_ids[np.argsort(row)] for row in D])
    
    return sort_ids
        
def create_knn_graph(sort_ids=None, D=None, n_neighbors=10):
    if sort_ids is None and not D is None:
        sort_ids = sort_by_distance(D)        
    if not sort_ids is None:
        if n_neighbors < sort_ids.shape[0]:
            adj_mat = np.zeros(sort_ids.shape)
            for row_ids, row_adj in zip(sort_ids, adj_mat):
                row_adj[row_ids[:n_neighbors]] = 1
        else:
            adj_mat = np.ones(D.shape)
    
    return adj_mat

def save_nearest_neighbors(
    ephys_data,
    behav_data,
    names
):
    dat = dict()
    for imouse in range(len(names)):
        spkmat = ephys_data[imouse]["spkmat"]
        T_neither, T_whisk_only, T_lomot_only, T_both = label_tpoints(ephys_data, behav_data, mouseID=imouse)
        sort_ids = [sort_by_distance(squareform(pdist(spkmat[:, T], metric="correlation"))) for T in [T_neither, T_whisk_only, T_both]]
        dat[names[imouse]] = sort_ids
        
    savemat(f"{root}/Data/save/nearest_neighbors_sort_indices.mat", dat)


def plot_modularity_vs_knn(
    ephys_data,
    behav_colors,
    names,
    nn_vals,
    plot_width=4,
    plot_height=5,
    save_plot=False
):
    
    nn_ids_dict = loadmat(f"{root}/Data/save/nearest_neighbors_sort_indices.mat")
    beh_lbs = ["Neither", "Whisking Only", "Both"]
    colors = [behav_colors[0], behav_colors[1], behav_colors[3]]
    ytks = np.round(np.arange(0, 0.61, 0.2), 1)

    n_mice = len(names)
    fig, axs = plt.subplots(1, n_mice, figsize=(n_mice*plot_width, plot_height))
    fig.subplots_adjust(wspace=0.1)
    
    for imouse, name, ax, ephys in zip(range(n_mice), names, axs, ephys_data):
        regIDs = ephys["regIDs"]
        communities = [set(np.argwhere(regIDs==rid).flatten()) for rid in np.unique(regIDs)]
        nn_ids_imouse = nn_ids_dict[name]

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
            
    if save_plot:
        plt.savefig(f"{root}/Plots/modularity.png", bbox_inches="tight", transparent=True)


def compute_mean_correlations(
    ephys_data,
    behav_data,
    names,
    save_plot=False
):

    for imouse in range(len(names)):
        spkmat = ephys_data[imouse]["spkmat"]
        regIDs = ephys_data[imouse]["regIDs"][:, np.newaxis]
        T_neither, T_whisk_only, T_lomot_only, T_both = label_tpoints(ephys_data, behav_data, mouseID=imouse)
        
        print(names[imouse].upper())
        diff_region = squareform((regIDs - regIDs.T) > 0)
        print(diff_region.dtype)
        for T in [T_neither, T_whisk_only, T_both]:
            cormat = 1 - pdist(spkmat[:, T], metric="correlation")
            print(f"Intra:{cormat[diff_region].mean():.4f}, Inter:{cormat[1-diff_region].mean():.4f}")

