import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat, savemat
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import silhouette_samples
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

from helper_functions import label_tpoints
from dependencies import root



def compute_silhouette_vs_behavior(
    ephys_data,
    behav_data,
    names
):
    
    sils = dict()
    sils["beh_states"] = ["Neither", "Whisking only", "Both"]
    for imouse in range(len(names)):
        spkmat = ephys_data[imouse]["spkmat"]
        regIDs = ephys_data[imouse]["regIDs"]
        T_neither, T_whisk_only, T_lomot_only, T_both = label_tpoints(behav_data=behav_data, mouseID=imouse)
        T_splits = [T_neither, T_whisk_only, T_both]
        sils[names[imouse]] = [silhouette_samples(squareform(pdist(spkmat[:, T], metric="correlation")), regIDs, metric="precomputed") for T in T_splits]
        
    savemat(f"{root}/Data/save/silhouette.mat", sils)


def _plot_sil_samples(
    sil_samples=None,
    clust_ids=None,
    unq_ids=None,
    unq_lbs=None,
    ax=None,
    title=None,
    colors=None,
    sort_idx=[],
    pad=20,
    uplim=0.2,
    lowlim=-0.2
):
    
    xpos = [lowlim, uplim]
    algs = ["left", "right"]
    xtks = np.arange(lowlim, uplim+0.05, 0.05).round(2)
    xtklbs = [str(xtks[i]) if i%2==0 else "" for i in range(len(xtks))]

    y_lower = pad
    for i, cid in enumerate(unq_ids):
        ireg_sil_samples = sil_samples[clust_ids == cid]
        
        if len(sort_idx) < len(unq_ids):
            sort_idx.append(np.argsort(ireg_sil_samples))
            
        ireg_sil_samples = ireg_sil_samples[sort_idx[i]]
            

        ireg_size = ireg_sil_samples.size
        y_upper = y_lower + ireg_size

        clr = colors[cid-1]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ireg_sil_samples,
            facecolor=clr,
            edgecolor=clr,
            linewidth=1.
        )

        ax.text(x=xpos[i%2],
                y=y_lower + 0.5*ireg_size,
                s=unq_lbs[cid-1],
                fontsize=16,
                c=clr,
                ha=algs[i%2],
                va="top"
        )
        y_lower = y_upper + pad

    ax.set_title(title, fontsize=20)
    ax.set_xlim((lowlim, uplim))
    ax.set_xlabel("Silhouette", fontsize=18)
    ax.set_xticks(xtks)
    ax.set_xticklabels(xtklbs, fontsize=17)
    ax.set_yticks([])
    ax.grid(visible=True, axis='x', which="major")
    ax.set_axisbelow(True)
        
    return sort_idx

def plot_silhouette_vs_behavior(
    ephys_data,
    behav_data,
    names,
    mouseID=0,
    colors=None,
    save_plot=False
):
    spkmat = ephys_data[mouseID]["spkmat"]
    regIDs = ephys_data[mouseID]["regIDs"]
    unq_regIDs= np.unique(regIDs)
    reglbs = ephys_data[mouseID]["reglbs"]
    n_regs = len(reglbs)

    T_neither, T_whisk_only, T_lomot_only, T_both = label_tpoints(behav_data, mouseID=mouseID)
    T_splits = [T_neither, T_whisk_only, T_both]
    n_splits = len(T_splits)
    
    silhouette_all_mice = loadmat(f"{root}/Data/save/silhouette.mat")
    beh_labels = ["Neither", "Whisking Only", "Both"] #silhouette_all_mice["beh_states"]
    silhouette_all_behs = silhouette_all_mice[names[mouseID]]
    
    fig, (pax_left, pax_right) = plt.subplots(1, 2, figsize=(21, 10),
                                              gridspec_kw=dict(width_ratios=[8, 3]))
    
    sort_indices = []
    gs = GridSpecFromSubplotSpec(1, n_splits, subplot_spec=pax_left)
    for isplit, cell in enumerate(gs):
        ax = plt.subplot(cell)
        sort_indices = _plot_sil_samples(
            sil_samples=silhouette_all_behs[isplit],
            clust_ids=regIDs,
            unq_ids=unq_regIDs,
            unq_lbs=reglbs,
            ax=ax,
            title=beh_labels[isplit],
            colors=colors,
            sort_idx=sort_indices
        )
        
    xtks = np.arange(n_splits)
    xtklbs = [s.replace(' ', '\n') for s in beh_labels]
    ytks = np.arange(0, 0.15, 0.05)
    nrows = 6
    ncols = 2
    gs = GridSpecFromSubplotSpec(nrows, ncols, subplot_spec=pax_right, hspace=0.15, wspace=0.15)
    for itr, cell in enumerate(gs):
        if itr < unq_regIDs.size:
            i = itr // 2
            j = itr % 2
            rid = unq_regIDs[(unq_regIDs.size-1)-itr]
            cond = regIDs==rid
            mus = [silhouette_all_behs[i][cond].mean() for i in range(n_splits)]
            sigmas = [silhouette_all_behs[i][cond].std() for i in range(n_splits)]
            
            ax = plt.subplot(cell)
            ax.errorbar(xtks, mus, yerr=sigmas, color=colors[rid-1], linewidth=2.5)
            ax.set_title(reglbs[rid-1] + " ", y=0.7, fontsize=16, loc="right", fontdict=dict(ha="right"))
            ax.set_xlim((-0.3, 2.3))
            ax.set_ylim((-0.05, 0.15))                    
            ax.set_xticks(xtks)
            ax.set_xticklabels(xtklbs if itr==unq_regIDs.size-1 or itr==unq_regIDs.size-2 else [], fontdict=dict(rotation=60), fontsize=17)
            ax.set_yticks(ytks)
            ax.set_yticklabels(ytks if j==0 else [], fontsize=17)
            ax.tick_params(axis='both', direction="in")
            
    fig.text(0.67, 0.5, "Silhouette", fontsize=18, ha="center", rotation=90)
    fig.text(0.08, 0.5, names[mouseID], fontsize=20, ha="center")
    
    if save_plot:
        plt.savefig(f"{root}/Plots/silhouette_{names[mouseID]}.png", bbox_inches="tight")


def plot_silhouette_vs_nneighbors(
    ephys_data,
    names,
    mice_colors,
    save_plot=False
):

    embed_dict = loadmat(f"{root}/Data/save/umap_embeddings.mat")
    embed_dim = embed_dict["embed_dim"].item()
    n_neighbor_vals = embed_dict["n_neighbor_vals"].flatten()

    fig, ax = plt.subplots(figsize=(15, 5))

    for name, ephys, color in zip(names, ephys_data, mice_colors):
        regIDs = ephys["regIDs"]
        embed_imouse = embed_dict[name]
        
        sil_mean_stds = np.zeros(n_neighbor_vals.size)
        for inn, embed_nn in enumerate(embed_imouse):
            sil_samples = np.array([
                silhouette_samples(
                    X=embed_nn[:, i*embed_dim:(i+1)*embed_dim],
                    labels=regIDs,
                    metric="euclidean"
                    )
                for i in range(n_neighbor_vals.size)])

            sil_mean_stds[inn] = sil_samples.std(axis=0).mean()

        ax.plot(n_neighbor_vals, sil_mean_stds, color=color, label=name)

    ax.set_ylim((0, 0.15))
    ax.set_yticks(np.arange(0, 0.16, 0.05))
    ax.set_ylabel(r"$E[\sigma_{s(i)}]$")
    ax.set_xlabel("Number of nearest neighbors")
    ax.legend(
        loc="upper right",
        prop=dict(size=15),
        framealpha=0
    )

    if save_plot:
        plt.savefig(f"{root}/Plots/silhouette_vs_nn.png", bbox_inches="tight")


def plot_silhouette_by_region(
    ephys_data,
    names,
    behavior_indices=[[0, 2], [1, 2]],
    micecols=None,
    standard_error=False,
    save_plot=False
):
    
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

                        ax.errorbar(xtks, mus[ii, :], yerr=sigmas[ii, :], marker='o', markersize=1, color=micecols[ii])

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


    if save_plot:
        plt.savefig(f"{root}/Plots/silhouette_by_region.png", bbox_inches="tight")



def plot_silhouette_by_mouse(
    ephys_data,
    names,
    regcols=None,
    standard_error=False,
    behavior_indices=[[0, 2], [1, 2]],
    save_plot=True
):
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

    

def silhouette_kstest(
    ephys_data,
    behav_data,
    names,
    behav_colors,
    behav_indices=[[0, 1], [0, 2]],
    plot_width=4,
    plot_height=5,
    save_plot=False
):
    nrows, ncols = len(names), len(behav_indices)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*plot_width, nrows*plot_height))
    fig.subplots_adjust(wspace=0.1, hspace=0.2)
    colors = [behav_colors[0], behav_colors[1], behav_colors[3]]
    behlbs = ["Neither", "Whisking\nOnly", "Both"]
    xtks = np.round(np.arange(-0.2, 0.21, 0.1), 1)
    ytks = np.arange(0, 1.1, 0.25)

    x = np.arange(-0.2, 0.21, 0.005)
    pad = 1

    for irow, row in enumerate(axs):
        spkmat = ephys_data[irow]["spkmat"]
        regIDs = ephys_data[irow]["regIDs"]
        T_neither, T_whisk_only, T_lomot_only, T_both = label_tpoints(ephys_data, behav_data, mouseID=irow)
        ecdfs = [ECDF(silhouette_samples(spkmat[:, T], regIDs, metric="correlation"))(x) for T in [T_neither, T_whisk_only, T_both]]

        for icol, indice_pair, ax in zip(range(2), behav_indices, row):
            for idx in indice_pair:
                ax.plot(x, ecdfs[idx], c=colors[idx], label=behlbs[idx])

            y1, y2 = ecdfs[indice_pair[0]], ecdfs[indice_pair[1]]
            ecdf_diffs = np.abs(y1 - y2)
            stat_idx = np.argmax(ecdf_diffs)
            ax.fill_between(x[stat_idx-pad:stat_idx+pad], y1[stat_idx-pad:stat_idx+pad], y2[stat_idx-pad:stat_idx+pad], color="black")
            ax.set_title(f"statistics={ecdf_diffs[stat_idx]:.4f}")
            ax.set_xticks(xtks)
            ax.set_xticklabels(xtks if irow==nrows-1 else [])
            ax.set_yticks(ytks)
            ax.set_yticklabels(ytks if icol==0 else [])
            
            if irow == nrows-1:
                ax.set_xlabel("Silhouette")
                ax.legend(
                    loc="lower right",
                    prop=dict(size=13),
                    framealpha=0
                )
            if icol == 0:
                ax.set_ylabel("Probability")

    text_ys = [0.75, 0.5, 0.25]
    for y, name in zip(text_ys, names):
        fig.text(0.98, y, name, fontdict=dict(ha="center"), fontsize=17)

                
    if save_plot:
        plt.savefig(f"{root}/Plots/silhouette_kstest.png", bbox_inches="tight", transparent=True)