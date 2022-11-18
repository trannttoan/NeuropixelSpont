import numpy as np
import matplotlib.pyplot as plt

from umap import UMAP
from scipy.io import loadmat, savemat

from dependencies import root
from helper_functions import load_data, generate_colors, plot_config


def create_umap_embeddings(
    ephys_data,
    names,
    metric="correlation",
    n_reps=10,
    embed_dim=2,
    n_neighbors_vals=np.round(np.logspace(np.log10(5), np.log10(500), 10) / 5).astype(int) * 5
):

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
    savemat(f"{root}/Data/save/umap_embeddings.mat", mdict=embed_dict)


def plot_umap_embeddings(
    save_plot=False
):

    embed_dict = loadmat(f"{root}/Data/save/umap_embeddings.mat")
    for name in names:
        embeddings = embed_dict[name]
        print(embeddings.shape)
        print(embeddings[0, :, :2])



    if save_plot:
        plt.savefig(f"{root}/Plots/umap_vs_nn.png", bbox_inches="tight")


ephys_data, behav_data, names = load_data()
mice_colors, region_colors, behav_colors = generate_colors()
# create_umap_embeddings(ephys_data, names)
plot_umap_embeddings()