import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat, savemat
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import average, cut_tree
from sklearn.metrics import silhouette_samples, rand_score
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpecFromSubplotSpec

from helper_functions import label_tpoints
from dependencies import root


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
    names,
    micecols,
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
    names,
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


def plot_silhouette_by_region(
    ephys_data,
    names,
    mice_colors,
    behavior_indices=[[0, 2], [1, 2]],
    standard_error=False,
    path=''
):
    """


    Parameters
    ----------
    ephys_data : list
        Dictionaries containing the processed neural activity and neuron locations
    names : list 
        Names of mice
    mice_colors : list
        List of colors assigned to mice
    behavior_indices : list
        Pairs of indices
        (0-Neither, 1-Whisking Only, 2-Running Only, 3-Both)
    standard_error : bool
        If true, plot standard error of the mean instead of standard deviation
    path : str
        Path to where figure is saved
    """
    
    n_mice = len(names)
    silhouette_all_mice = loadmat(f"{root}/Data/save/silhouette.mat")
    
    reglbs = ephys_data[0]["reglbs"]
    n_regs = len(reglbs)
    behlbs = ["Neither", "Whisking\nOnly", "Both"]
    n_behs = len(behlbs)
    
    xtks = [0, 1]
    ytks = np.arange(0, 0.15, 0.05)
    nrows = 4; ncols = 3
    fig, axs = plt.subplots(nrows, ncols, figsize=(17, 7))
    fig.subplots_adjust(wspace=0.15)
    
    total_neuron_count = 0
    neuron_counts = []
    silhouette_dict = {}
    for imouse, name in enumerate(names):
        regIDs = ephys_data[imouse]["regIDs"]
        total_neuron_count += regIDs.size
        neuron_counts.append(regIDs.size)
        
        silhouette_all_behaviors = silhouette_all_mice[name]
        silhouette_dict[name] = {}
        silhouette_dict[name]["neurons_per_region"] = [(regIDs==rid+1).sum() for rid in range(n_regs)]
        for beh_idx in np.arange(len(behlbs)):
            sil_samples = silhouette_all_behaviors[beh_idx]
            silhouette_dict[name][behlbs[beh_idx]] = [sil_samples[regIDs==rid+1] for rid in range(n_regs)]
        
    weights = np.array(neuron_counts) / total_neuron_count
    weights = weights[:, np.newaxis]
    
    for i, row in enumerate(axs):
        for j, ax in enumerate(row):
            rid = i*ncols + j
            gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=ax, wspace=0.05)
            for k, cell, index_pair in zip([0, 1], gs, behavior_indices):
                ax = plt.subplot(cell)
                mus = np.zeros((n_mice, len(index_pair)))
                sigmas = np.zeros((n_mice, len(index_pair)))
                for ii, name in enumerate(names):
                    if silhouette_dict[name]["neurons_per_region"][rid] > 0:
                        for jj, beh_idx in enumerate(index_pair):
                            sils = np.array(silhouette_dict[name][behlbs[beh_idx]][rid])
                            mus[ii, jj] = sils.mean()
                            sigmas[ii, jj] = sils.std() if not standard_error else sils.std() / np.sqrt(sils.size)

                        ax.errorbar(xtks, mus[ii, :], yerr=sigmas[ii, :], marker='o', markersize=1, color=mice_colors[ii])

                ax.errorbar(xtks, (weights*mus).sum(axis=0), yerr=(weights*sigmas).sum(axis=0), marker='o', markersize=1, color="black")
                                        
                if k==1:
                    ax.set_title(reglbs[rid] + ' ', y=0.7, loc="right", fontdict=dict(ha="right"), fontsize=16)
                ax.set_ylabel("Silhouette" if j==0 and k==0 else "", fontsize=16)
                ax.set_xlim((-0.3, 1.3))
                ax.set_ylim((-0.05, 0.15))                    
                ax.set_xticks(xtks)
                ax.set_xticklabels([behlbs[idx] for idx in index_pair] if i==nrows-1 else [])
                ax.set_yticks(ytks)
                ax.set_yticklabels(ytks if j==0 and k==0 else [])
                ax.tick_params(axis='both', direction="in")


    if path:
        path = f"{root}/Plots/silhouette_by_region.png"
    plt.savefig(path, bbox_inches="tight")



def plot_silhouette_by_mouse(
    ephys_data,
    names,
    regcols=None,
    standard_error=False,
    behavior_indices=[[0, 2], [1, 2]],
    save_plot=True
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
    mouseID : int
        Index number of a mouse
    region_colors : list
        List of colors assigned to brain regions for plotting
    path : str
        Path to where figure is saved

    """

    fig, axs = plt.subplots(3, 2, figsize=(8, 8))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)
    
    n_mice = len(names)
    silhouette_all_mice = loadmat(f"{root}/Data/save/silhouette.mat")
    
    reglbs = ephys_data[0]["reglbs"]
    n_regs = len(reglbs)
    behlbs = ["Neither", "Whisking\nOnly", "Both"]
    n_behs = len(behlbs)
    
    xtks = [0, 1]
    ytks = np.arange(0, 0.15, 0.05)
    
    y = 0.75
    for irow, row in enumerate(axs):
        fig.text(1, y, names[irow], fontsize=17, ha="center")
        silhouette_all_behaviors = silhouette_all_mice[names[irow]]
        regIDs = ephys_data[irow]["regIDs"]
        icol = 0
        
        for ax, index_pair in zip(row, behavior_indices):
            mus = np.zeros((n_regs, 2))
            sigmas = np.zeros((n_regs, 2))
            weights = np.array([(regIDs == i).mean() for i in range(1, n_regs+1)])
            weights = weights[:, np.newaxis]
            for j, ind in enumerate(index_pair):
                for i in range(n_regs):
                    cond = regIDs == i + 1
                    if cond.any():
                        sils = silhouette_all_behaviors[ind][cond]
                        mus[i, j] = sils.mean()
                        sigmas[i, j] = sils.std() if not standard_error else sils.std() / np.sqrt(sils.size)

            # for each region
            for mu_pair, sigma_pair, color in zip(mus, sigmas, regcols):
                if sigma_pair[0] != 0:
                    ax.errorbar(xtks, mu_pair, yerr=sigma_pair, marker='o', markersize=1, color=color)
                    
            # weighted average
            ax.errorbar(xtks, (mus*weights).sum(axis=0), yerr=(sigmas*weights).sum(axis=0), marker='o', markersize=1, color="black")

            ax.set_ylabel("Silhouette" if icol==0 else "")
            ax.set_xlim((-0.3, 1.3))
            ax.set_ylim((-0.05, 0.15))                    
            ax.set_xticks(xtks)
            ax.set_xticklabels([behlbs[idx] for idx in index_pair] if irow==n_mice-1 else [])
            ax.set_yticks(ytks)
            ax.set_yticklabels(ytks if icol==0 else [])
            ax.tick_params(axis='both', direction="in")
            
            icol += 1
        y -= 0.25
        
    h = [Line2D(
        [0], [0],
        marker='o',
        color=regcols[i],
        markerfacecolor=regcols[i],
        markersize=12,
        linewidth=2,
        label=reglbs[i]
    ) for i in range(n_regs)]
    h.append(Line2D(
        [0], [0],
        marker='o',
        color="black",
        markerfacecolor="black",
        markersize=12,
        linewidth=2,
        label="Average")
    )

    fig.legend(
        handles=h,
        prop={"size":14},
        ncol=7,
        framealpha=0,
        labelspacing=0.1,
        columnspacing=1.2,
        handletextpad=0.5,
        loc="center",
        bbox_to_anchor=(0.55, .92)
    )

    if save_plot:
        plt.savefig(f"{root}/Plots/silhouette_by_mouse.png", bbox_inches="tight")
