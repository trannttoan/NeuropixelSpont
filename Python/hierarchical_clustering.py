import numpy as np
import matplotlib.pyplot as plt
import h5py

from scipy.io import loadmat
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet

from matplotlib.lines import Line2D

from behavior_extraction import label_tpoints
from dependencies import root


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
        Z = linkage(squareform(D), method="average")
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

        
        # brain regions
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

        


def plot_cophenetic_vs_behavior(
    ephys_data,
    behav_data,
    names,
    methods=["single", "complete", "average"],
    plot_width=4,
    plot_height=5,
    save_plot=False
):
    
    fig, axs = plt.subplots(1, len(names), figsize=(len(names)*plot_width, plot_height))
    beh_lbs = ["Neither", "Whisking Only", "Both"]
    colors = [plt.cm.Accent(v) for v in np.linspace(0, 1, len(methods))]
    xtks = list(range(len(beh_lbs)))
    ytks = np.round(np.arange(0, 0.81, 0.2), 1)
    
    for imouse, ax in enumerate(axs):
        spkmat = ephys_data[imouse]["spkmat"]
        T_neither, T_whisk_only, T_lomot_only, T_both = label_tpoints(ephys_data, behav_data, mouseID=imouse)
        T_masks = [T_neither, T_whisk_only, T_both]
        cophenet_coeffs = []

        for T in T_masks:
            Y = pdist(spkmat[:, T], metric="correlation")
            cophenet_coeffs.append([cophenet(linkage(Y, method=m), Y)[0] for m in methods])

        ax.set_prop_cycle(color=colors)
        ax.plot(cophenet_coeffs, label=methods)
        ax.set_xticks(xtks)
        ax.set_xticklabels(beh_lbs)
        ax.set_ylim((0, 0.8))
        ax.set_yticks(ytks)
        ax.set_yticklabels(ytks if imouse==0 else [])
        ax.set_title(names[imouse])

        if imouse == 0:
            ax.set_ylabel("Cophenetic Correltion")


    ax.legend(
        prop=dict(size=14),
        framealpha=0,
        labelspacing=0.15,
        bbox_to_anchor=(0, 0),
        loc="lower left"
    )

    if save_plot:
        plt.savefig(f"{root}/Plots/cophenetic_vs_behavior.png", bbox_inches="tight", transparent=True)