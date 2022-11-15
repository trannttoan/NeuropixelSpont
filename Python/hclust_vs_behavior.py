import numpy as np
import matplotlib.pyplot as plt
import h5py

from sklearn.metrics import rand_score, adjusted_rand_score
from scipy.io import loadmat, savemat
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import average, single, cut_tree, dendrogram
from matplotlib.lines import Line2D

from helper_functions import load_data, label_tpoints, generate_colors, plot_config
from dependencies import root

def compute_ari_vs_behavior(
    ephys_data,
    behav_data,
    names,
    n_cluster_vals=np.arange(2, 13)
):
    
    rand_dict = dict()
    rand_dict["beh_states"] = ["Neither", "Whisking only", "Both"]
    rand_dict["n_cluster_vals"] = n_cluster_vals

    for imouse in range(3):
        spkmat = ephys_data[imouse]["spkmat"]
        regIDs = ephys_data[imouse]["regIDs"]
        T_neither, T_whisk_only, T_lomot_only, T_both = label_tpoints(ephys_data, behav_data, mouseID=imouse)
        T_splits = [T_neither, T_whisk_only, T_both]
        rand_scores = np.zeros((len(T_splits), n_cluster_vals.size))

        for ibeh, T in enumerate(T_splits):
            D = squareform(pdist(spkmat[:, T], metric="correlation"))
            Z = average(squareform(D))
            rand_scores[ibeh, :] = np.array([rand_score(regIDs, cut_tree(Z, n_clusters=n_clusters).flatten()) for n_clusters in n_cluster_vals])
        rand_dict[names[imouse]] = rand_scores

    savemat(f"{root}/Data/save/rand_index.mat", mdict=rand_dict)


def plot_ari_vs_behavior(
    micecols=None,
    save_plot=False
):

    rand_dict = loadmat(f"{root}/Data/save/rand_index.mat")
    n_cluster_vals = rand_dict["n_cluster_vals"].flatten()
    nrows, ncols = len(names), len(rand_dict["beh_states"])
    fig, axs = plt.subplots(nrows, ncols, figsize=(15, 9))

    for imouse, row in enumerate(axs):
        rand_scores = rand_dict[names[imouse]]
        for ax, rands in zip(row, rand_scores):
            ax.plot(n_cluster_vals, rands)
            ax.set_xticks(np.arange(2, 13, 2))
            ax.set_ylim((0, 0.65))

    if save_plot:
        plt.savefig(f"{root}/Plots/rand_index.png", bbox_inches="tight")

    
def plot_hclust_vs_behavior(
    ephys_data,
    behav_data,
    names,
    mouseID=0,
    colors=None,
    dpi=100,
    save_plot=False
):

    f = h5py.File(f"{root}/Data/save/pairwise_minfo.mat")
    minfos_all = {k:np.array(v) for k, v in f.items()}
    minfos = minfos_all[names[mouseID]]
    fig, axs = plt.subplots(4, 4, figsize=(14, 15), dpi=dpi,
                           gridspec_kw=dict(height_ratios=[4, 4, 4, 1],
                                            width_ratios=[4, 4, 4, 1]))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    
    T_neither, T_whisk_only, T_lomot_only, T_both = label_tpoints(ephys_data, behav_data, mouseID=mouseID)
    T_splits = [T_neither, T_whisk_only, T_both]
    beh_lbs = ["Neither", "Whisking Only", "Both"]
    spkmat = ephys_data[mouseID]["spkmat"]
    coords = ephys_data[mouseID]["coords"]
    dist = squareform(pdist(coords, metric="euclidean"))
    regIDs = ephys_data[mouseID]["regIDs"]
    xreg = np.arange(spkmat.shape[0])
    ticks = np.arange(0, spkmat.shape[0], 500)

    cor_dict  = dict(
        lims = (-0.25, 0.75),#(-0.5, 1),
        cmap = plt.cm.viridis,
        label = "Pearson's\ncorrelation",
        ticks = np.arange(-0.25, 1, 0.25)#np.arange(-0.5, 1.1, 0.5)
    )
    
    mi_dict = dict(
        lims = (-18, 2),
        cmap = plt.cm.viridis,
        label = "Mutual\ninformation\n (log)",
        ticks = np.arange(-15, 1, 5)
    )
        
    pdist_dict  = dict(
        lims = (0, 6000),
        cmap = plt.cm.viridis,
        label = "Physical\ndistance\n" +  r"($\mu$m)",
        ticks = np.arange(0, 6001, 1500)
    )
    
    plot_config = [cor_dict, mi_dict, pdist_dict]
    
    
    for i in range(len(beh_lbs)):
        # correlation
        D = squareform(pdist(spkmat[:, T_splits[i]], metric="correlation"))
        Z = average(squareform(D))
        dn = dendrogram(Z, no_plot=True)
        sort_ids = dn["leaves"]
        cor_mat = 1 - D
        
        # mutual information
        mi_mat = np.log(minfos[:, :, i])
        
        # physical distance
        pd_mat = dist
        
        j = 0
        for ax, mat, dt in zip(axs[:-1, i], [cor_mat, mi_mat, pd_mat], plot_config):        
            mat = mat[:, sort_ids]
            mat = mat[sort_ids, :]
            pcm = ax.matshow(mat, aspect="auto", vmin=dt["lims"][0], vmax=dt["lims"][1], cmap=dt["cmap"])
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            if i > 0: ax.set_yticklabels([])
            if j > 0: ax.set_xticklabels([])
            if i == len(beh_lbs)-1: fig.colorbar(pcm, cax=axs[j, -1], ticks=dt["ticks"])
            
            j += 1

        
        # regions
        ax = axs[3, i]
        regIDs_sorted = regIDs[sort_ids]
        for ireg in np.unique(regIDs):
            ax.fill_between(xreg, 0, 1, where=regIDs_sorted==ireg, color=colors[ireg-1], transform=ax.get_xaxis_transform())
        ax.margins(x=0)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_title(beh_lbs[i], y=-0.5)
        
        
    for ax, dt in zip(axs[:-1, -1], plot_config):
        ax.set_ylabel(dt["label"], rotation=0, labelpad=70, va="center")
        
    reglbs = ephys_data[mouseID]["reglbs"]
    axs[-1, -1].axis("off")
    h = [Line2D(
        [0], [0],
        marker='o',
        markersize=16,
        markerfacecolor=colors[i],
        color='w',
        markeredgecolor='w',
        label=reglbs[i]
    ) for i in range(len(reglbs))]
        
    axs[-1, -1].legend(
        handles=h,
        ncol=3,
        prop={'size':14},
        framealpha=0,
        columnspacing=0.5,
        labelspacing=0.4,
        handletextpad=0.1,
        loc="center",
        bbox_to_anchor=(2, 0.3))
    
    fig.text(0, 0.5, names[mouseID], fontsize=18, ha="center")
    
    if save_plot:
        plt.savefig(f"{root}/Plots/hclust_wrt_behaviors_{names[mouseID]}.png", dpi=dpi, bbox_inches="tight")

ephys_data, behav_data, names = load_data()
mice_colors, region_colors, behav_colors = generate_colors()
plot_config()
# compute_ari_vs_behavior(ephys_data, behav_data, names)
plot_ari_vs_behavior(micecols=mice_colors, save_plot=True)

# for imouse in range(3):
#     plot_hclust_vs_behavior(ephys_data, behav_data, names, mouseID=imouse, colors=region_colors, dpi=1000, save_plot=True)