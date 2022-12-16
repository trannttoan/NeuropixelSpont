import numpy as np
import matplotlib.pyplot as plt

from umap import UMAP
from scipy.io import loadmat, savemat
from sklearn.metrics import silhouette_samples
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpecFromSubplotSpec

from dependencies import root
from helper_functions import load_data, generate_colors, plot_config


def create_umap_vs_nneighbors(
    ephys_data,
    names,
    metric="correlation",
    n_reps=10,
    embed_dim=2,
    n_neighbors_vals=np.round(np.logspace(np.log10(5), np.log10(500), 10) / 5).astype(int) * 5,
    path=''
):
    """


    Parameters
    ----------
    ephys_data : list
        Dictionaries containing the processed neural activity and neuron locations
    names : list 
        Names of mice
    metric : str
        Distance measure
    n_reps : int
        Number of embeddings to generate for each set of parameters
    embed_dim : int
        Embedding dimension
    n_neighbor_vals : array-like
        Array of values of number of nearest neighbors
    path : str
        Path to where the results are saved

    """

    embed_dict = dict()
    embed_dict["n_neighbor_vals"] = n_neighbors_vals
    embed_dict["embed_dim"] = embed_dim

    for imouse, dat in enumerate(ephys_data):
        print(names[imouse] + ':', end='\t')
        spkmat = dat["spkmat"]
        embeddings = np.zeros((n_neighbors_vals.size, spkmat.shape[0], embed_dim*n_reps))
        for i, nn in enumerate(n_neighbors_vals):
            print(nn, end=' ')
            for j in range(n_reps):
                embeddings[i, :, j*embed_dim:(j+1)*embed_dim] = UMAP(n_neighbors=nn, n_components=embed_dim, metric=metric).fit_transform(spkmat)

        embed_dict[names[imouse]] = embeddings
        print()

    if path == '':
        path = f"{root}/Data/save/umap_embeddings.mat"
    savemat(path, mdict=embed_dict)


def plot_umap_vs_nneighbors(
    ephys_data,
    names,
    region_colors,
    n_plots=5,
    plot_width=4,
    plot_height=4,
    path=''
):
    """


    Parameters
    ----------
    ephys_data : list
        Dictionaries containing the processed neural activity and neuron locations
    names : list 
        Names of mice
    region_colors : list
        List of colors assigned to brain regions
    n_plots : int
        Number of plots for each mouse/row
    plot_width : float
        Width of indidividual plots
    plot_height : float
        Height of individual plots
    path : str
        Path to where the figure is saved

    """

    # load saved UMAP embeddings
    embed_dict = loadmat(f"{root}/Data/save/umap_embeddings.mat")
    embed_dim = embed_dict["embed_dim"].item()
    n_neighbor_vals = embed_dict["n_neighbor_vals"].flatten()
   
    reglbs = ephys_data[0]["reglbs"]
    i_emb = 0
    
    # initialize and adjust axes
    nrows, ncols = len(names), n_plots
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*plot_width, nrows*plot_height))
    fig.subplots_adjust(wspace=0.1, hspace=0.15)

    # main
    for irow, row, ephys, name in zip(list(range(len(names))), axs, ephys_data, names):
        regIDs = ephys["regIDs"]
        cols = [region_colors[rid-1] for rid in regIDs]
        embed_imouse = embed_dict[name]
        for icol, ax, embed, nn in zip(list(range(n_neighbor_vals.size)), row, embed_imouse[::2], n_neighbor_vals[::2]):
            sils = silhouette_samples(embed[:, i_emb*embed_dim:(i_emb+1)*embed_dim], regIDs)
            ax.scatter(embed[:, 0], embed[:, 1], c=cols, s=6)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title((f"n_neighbors={nn}\n\n" if irow==0 else "") + f"s={sils.mean():.4f}", y=0.98)
            if icol==0:
                ax.set_ylabel(name, labelpad=60, rotation=0, ha="center")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.get_yaxis().set_label_position("right")
    
    h = [Line2D(
        [0], [0],
        marker='o',
        markerfacecolor=cl,
        color='w',
        markersize=16,
        label=lb,
    ) for cl, lb in zip(region_colors, reglbs)]

    fig.legend(
        handles=h,
        ncol=1,
        handletextpad=0.1,
        prop={"size":16},
        bbox_to_anchor=(1.02, 0.89),
        framealpha=0
    )

    if path == '':
        path = f"{root}/Plots/umap_vs_nn.png"
    plt.savefig(path, bbox_inches="tight")