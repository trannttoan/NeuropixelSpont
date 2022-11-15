import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples

from scipy.io import loadmat, savemat
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import average, single, cut_tree

from dependencies import root
from helper_functions import load_data, generate_colors, plot_config, label_tpoints



def compute_silhouette_vs_nclusters(
    ephys_data,
    behav_data,
    names,
    n_cluster_vals=np.arange(2, 13)
):
    
    sils_dict = dict()
    sils_dict["beh_states"] = ["Neither", "Whisking only", "Both"]
    sils_dict["n_cluster_vals"] = n_cluster_vals

    for imouse in range(3):
        spkmat = ephys_data[imouse]["spkmat"]
        regIDs = ephys_data[imouse]["regIDs"]
        T_neither, T_whisk_only, T_lomot_only, T_both = label_tpoints(ephys_data, behav_data, mouseID=imouse)
        T_splits = [T_neither, T_whisk_only, T_both]
        sil_samples = np.zeros((len(T_splits), n_cluster_vals.size, spkmat.shape[0]))

        for ibeh, T in enumerate(T_splits):
            D = squareform(pdist(spkmat[:, T], metric="correlation"))
            Z = average(squareform(D))
            sil_samples[ibeh, :, :] = np.array([silhouette_samples(D, cut_tree(Z, n_clusters=n_clusters).flatten(), metric="precomputed") for n_clusters in n_cluster_vals])
        sils_dict[names[imouse]] = sil_samples

    savemat(f"{root}/Data/save/silhouette_hclust.mat", mdict=sils_dict)


def plot_silhouette_vs_nclusters(
    micecols=None,
    save_plot=False
):

    sil_dict = loadmat(f"{root}/Data/save/silhouette_hclust.mat")
    nrows, ncols = len(names), len(sil_dict["beh_states"])
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 9))

    for imouse, row in enumerate(axs):
        sil_samples = sil_dict[names[imouse]]
        for ax, sils in zip(row, sil_samples):
            ax.errorbar(np.arange(2, 13), sils.mean(axis=1), yerr=sils.std(axis=1))
            ax.set_xticks(np.arange(2, 13, 2))
            # ax.set_ylim((-0.06, 0.045))

    if save_plot:
        plt.savefig(f"{root}/Plots/silhouette_hclust.png", bbox_inches="tight")

ephys_data, behav_data, names = load_data()
mice_colors, region_colors, behav_colors = generate_colors()
plot_config()
compute_silhouette_vs_nclusters(ephys_data, behav_data, names)
plot_silhouette_vs_nclusters(micecols=mice_colors, save_plot=True)