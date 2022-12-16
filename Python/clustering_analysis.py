import numpy as np
import matplotlib.pyplot as plt
import h5py

from scipy.io import loadmat, savemat
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from sklearn.metrics import silhouette_samples
from statsmodels.distributions.empirical_distribution import ECDF
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

from behavior_extraction import label_tpoints
from dependencies import root


def plot_hclust_vs_behavior(
    ephys_data,
    behav_data,
    names,
    region_colors,
    mouseID=0,
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
    mouseID : int
        Index number of a mouse
    region_colors : list
        List of colors assigned to brain regions for plotting
    path : str
        Path to where the figure is saved
    """

    # load saved pairwise mutual information between neural time series
    f = h5py.File(f"{root}/Data/save/pairwise_minfo.mat")
    minfos_all = {k:np.array(v) for k, v in f.items()}
    minfos = minfos_all[names[mouseID]]

    # generate masks to extract neural activity based on behavioral state
    T_neither, T_whisk_only, T_lomot_only, T_both = label_tpoints(ephys_data, behav_data, mouseID=mouseID)
    masks = [T_neither, T_whisk_only, T_both]
    behlbs = ["Neither", "Whisking Only", "Both"]

    # store relevant data into variables
    spkmat = ephys_data[mouseID]["spkmat"]
    regIDs = ephys_data[mouseID]["regIDs"]
    coords = ephys_data[mouseID]["coords"]
    reglbs = ephys_data[mouseID]["reglbs"]
    anatom_dist = squareform(pdist(coords, metric="euclidean"))

    # initialize and adjust axes
    fig, axs = plt.subplots(4, 4, figsize=(14, 15), dpi=600,
                           gridspec_kw=dict(height_ratios=[4, 4, 4, 1],
                                            width_ratios=[4, 4, 4, 1]))
    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    # plotting variables (e.g. ticks, axis limits, colormaps, ...)
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
    
    # main
    for i in range(len(behlbs)):
        # hierarchical clustering
        D = squareform(pdist(spkmat[:, masks[i]], metric="correlation"))
        Z = linkage(squareform(D), method="average")
        dn = dendrogram(Z, no_plot=True)
        sort_ids = dn["leaves"]
        
        # correlation
        cor_mat = 1 - D
        
        # mutual information
        mi_mat = np.log(minfos[:, :, i])
        
        # anatomical distance
        pd_mat = anatom_dist
        
        j = 0
        for ax, mat, dt in zip(axs[:-1, i], [cor_mat, mi_mat, pd_mat], plot_config):        
            mat = mat[:, sort_ids]
            mat = mat[sort_ids, :]
            pcm = ax.matshow(mat, aspect="auto", vmin=dt["lims"][0], vmax=dt["lims"][1], cmap=dt["cmap"])
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            if i > 0: ax.set_yticklabels([])
            if j > 0: ax.set_xticklabels([])
            if i == len(behlbs)-1: fig.colorbar(pcm, cax=axs[j, -1], ticks=dt["ticks"])
            
            j += 1

        # brain regions
        ax = axs[3, i]
        regIDs_sorted = regIDs[sort_ids]
        for ireg in np.unique(regIDs):
            ax.fill_between(xreg, 0, 1, where=regIDs_sorted==ireg, color=region_colors[ireg-1], transform=ax.get_xaxis_transform())
        
        ax.margins(x=0)
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)
        ax.set_title(behlbs[i], y=-0.5)
        

    for ax, dt in zip(axs[:-1, -1], plot_config):
        ax.set_ylabel(dt["label"], rotation=0, labelpad=70, va="center")
        
    axs[-1, -1].axis("off")

    h = [Line2D(
        [0], [0],
        marker='o',
        markersize=16,
        markerfacecolor=cl,
        color='w',
        markeredgecolor='w',
        label=lb
    ) for cl, lb in range(region_colors, reglbs)]
        
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
    
    # fig.text(0, 0.5, names[mouseID], fontsize=18, ha="center")
    
    if path == '':
        path = f"{root}/Plots/hclust_wrt_behaviors_{names[mouseID]}.png"
    plt.savefig(path, dpi=600, bbox_inches="tight", transparent=True)

        


def plot_cophenetic_vs_behavior(
    ephys_data,
    behav_data,
    names,
    linkages=["single", "complete", "average"],
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
        Names (strings) of mice
    linkages : list
        Linkage methods (strings)
    region_colors : list
        List of colors assigned to brain regions for plotting
    plot_width : float
        Width of indidividual plots
    plot_height : float
        Height of individual plots
    path : str
        Path to where the figure is saved
    """
    
    # intialize and ajust axes
    fig, axs = plt.subplots(1, len(names), figsize=(len(names)*plot_width, plot_height))
    
    # plotting variables (e.g. ticks, axis limits, colormaps, ...)
    beh_lbs = ["Neither", "Whisking Only", "Both"]
    colors = [plt.cm.Accent(v) for v in np.linspace(0, 1, len(linkages))]
    xtks = list(range(len(beh_lbs)))
    ytks = np.round(np.arange(0, 0.81, 0.2), 1)
    
    # main
    for imouse, ax in enumerate(axs):
        spkmat = ephys_data[imouse]["spkmat"]
        T_neither, T_whisk_only, T_lomot_only, T_both = label_tpoints(ephys_data, behav_data, mouseID=imouse)
        T_masks = [T_neither, T_whisk_only, T_both]
        cophenet_coeffs = []

        for T in T_masks:
            Y = pdist(spkmat[:, T], metric="correlation")
            cophenet_coeffs.append([cophenet(linkage(Y, method=m), Y)[0] for m in linkages])

        ax.set_prop_cycle(color=colors)
        ax.plot(cophenet_coeffs, label=linkages)
        ax.set_xticks(xtks)
        ax.set_xticklabels(beh_lbs)
        ax.set_ylim((0, 0.8))
        ax.set_yticks(ytks)
        ax.set_yticklabels(ytks if imouse==0 else [])
        ax.set_title(names[imouse])

        if imouse == 0:
            ax.set_ylabel("Cophenetic Correlation")


    ax.legend(
        prop=dict(size=14),
        framealpha=0,
        labelspacing=0.15,
        bbox_to_anchor=(0, 0),
        loc="lower left"
    )

    if path == '':
        path = f"{root}/Plots/cophenetic_vs_behavior.png"

    plt.savefig(path, bbox_inches="tight", transparent=True)






def compute_silhouette_vs_behavior(
    ephys_data,
    behav_data,
    names,
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
    path : str
        Path to where the results are saved

    """

    sil_dict = dict()
    sil_dict["beh_states"] = ["Neither", "Whisking only", "Both"]
    for imouse in range(len(names)):
        spkmat = ephys_data[imouse]["spkmat"]
        regIDs = ephys_data[imouse]["regIDs"]
        T_neither, T_whisk_only, T_lomot_only, T_both = label_tpoints(ephys_data, behav_data, mouseID=imouse)
        masks = [T_neither, T_whisk_only, T_both]
        sil_dict[names[imouse]] = [silhouette_samples(squareform(pdist(spkmat[:, T], metric="correlation")), regIDs, metric="precomputed") for T in masks]
        
    if path == '':
        path = f"{root}/Data/save/silhouette.mat"
    savemat(path, sil_dict)


def _plot_sil_samples(
    sil_samples,
    clust_ids,
    unq_ids,
    unq_lbs,
    colors,
    ax,
    title,
    sort_idc=[],
    pad=20,
    lims=(0.2, -0.2)
):
    """
    (Helper function) Draw a Silhouette plot based on the brain region labels
    and correlation distance between neural time series

    Parameters
    ----------
    sil_samples : array-like
        Silhouette coefficients for all neurons
    clust_ids : array-like 
        Brain region indices
    unq_ids : array-like 
        Unique set of the brain region indices
    unq_lbs : array-like
        Unique set of the brain region labels
    colors : list
        List of colors corresponding to brain regions
    ax : list
        Axis to draw in
    title : str
        Title of the plot
    sort_idc : list
        List of indices that sorts the Silhouette coefficents
        within each brain region
    pad : int
        Vertical space between Sihouette plots
    lims : tuple
        Limits for the x-axis

    Returns
    -------
    sort_idc : list
        List of indices that sorts the Silhouette coefficents
        within each brain region
    """

    algs = ["left", "right"]
    xtks = np.arange(lims[0], lims[1]+0.05, 0.05).round(2)
    xtklbs = [str(xtks[i]) if i%2==0 else "" for i in range(len(xtks))]

    y_lower = pad
    for i, cid in enumerate(unq_ids):
        ireg_sil_samples = sil_samples[clust_ids == cid]
        
        if len(sort_idc) < len(unq_ids):
            sort_idc.append(np.argsort(ireg_sil_samples))
            
        ireg_sil_samples = ireg_sil_samples[sort_idc[i]]
            
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

        ax.text(x=lims[i%2],
                y=y_lower + 0.5*ireg_size,
                s=unq_lbs[cid-1],
                fontsize=16,
                c=clr,
                ha=algs[i%2],
                va="top"
        )
        y_lower = y_upper + pad

    ax.set_title(title, fontsize=20)
    ax.set_xlim(lims)
    ax.set_xlabel("Silhouette", fontsize=18)
    ax.set_xticks(xtks)
    ax.set_xticklabels(xtklbs, fontsize=17)
    ax.set_yticks([])
    ax.grid(visible=True, axis='x', which="major")
    ax.set_axisbelow(True)
        
    return sort_idc

def plot_silhouette_vs_behavior(
    ephys_data,
    behav_data,
    names,
    mouseID,
    region_colors,
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
    mouseID : int
        Index number of a mouse
    region_colors : list
        List of colors assigned to brain regions for plotting
    path : str
        Path to where the figure is saved

    """

    # store relevant data into variables
    spkmat = ephys_data[mouseID]["spkmat"]
    regIDs = ephys_data[mouseID]["regIDs"]
    unq_regIDs= np.unique(regIDs)
    reglbs = ephys_data[mouseID]["reglbs"]
    n_regs = len(reglbs)

    # generate masks to extract neural activity based on behavioral state
    T_neither, T_whisk_only, T_lomot_only, T_both = label_tpoints(behav_data, mouseID=mouseID)
    masks = [T_neither, T_whisk_only, T_both]
    n_splits = len(masks)
    
    # load saved Silhouette coefficients
    silhouette_all_mice = loadmat(f"{root}/Data/save/silhouette.mat")
    beh_labels = ["Neither", "Whisking Only", "Both"] #silhouette_all_mice["beh_states"]
    silhouette_all_behs = silhouette_all_mice[names[mouseID]]
    
    # initialize and adjust axes
    fig, (pax_left, pax_right) = plt.subplots(1, 2, figsize=(21, 10),
                                              gridspec_kw=dict(width_ratios=[8, 3]))
    
    # Silhouette plots (one for each behavioral state)
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
            colors=region_colors,
            sort_idx=sort_indices
        )
        
    # mean and standard deviation of Silhouette coefficient for each brain region 
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
            ax.errorbar(xtks, mus, yerr=sigmas, color=region_colors[rid-1], linewidth=2.5)
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
    
    if path == '':
        path = f"{root}/Plots/silhouette_{names[mouseID]}.png"
    plt.savefig(path, bbox_inches="tight", transparent=True)


def plot_silhouette_vs_nneighbors(
    ephys_data,
    names,
    mice_colors
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
    path : str
        Path to where the figure is saved

    """

    # load saved UMAP embeddings
    embed_dict = loadmat(f"{root}/Data/save/umap_embeddings.mat")
    embed_dim = embed_dict["embed_dim"].item()
    n_neighbor_vals = embed_dict["n_neighbor_vals"].flatten()

    # initialize axes
    fig, ax = plt.subplots(figsize=(15, 5))

    # main
    for name, ephys, color in zip(names, ephys_data, mice_colors):
        regIDs = ephys["regIDs"]
        embed_imouse = embed_dict[name]
        
        # calculate the standard deviation of Silhouette coefficients
        # across UMAP embeddings for each neuron then
        # the mean of those standard deviations 
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

    if path == '':
        path = f"{root}/Plots/silhouette_vs_nn.png"
    plt.savefig(path, bbox_inches="tight", transparent=True)
    

def silhouette_kstest(
    ephys_data,
    behav_data,
    names,
    behav_colors,
    behav_indices=[[0, 1], [0, 2]],
    plot_width=4,
    plot_height=5,

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
    behav_colors : list
        List of colors assigned to behavioral states
    behavior_indices : list
        Pairs of indices
        (0-Neither, 1-Whisking Only, 2-Running Only, 3-Both)
    plot_width : float
        Width of indidividual plots
    plot_height : float
        Height of individual plots
    path : str
        Path to where the figure is saved

    """

    # initialize and adjust axes
    nrows, ncols = len(names), len(behav_indices)
    fig, axs = plt.subplots(nrows, ncols, figsize=(ncols*plot_width, nrows*plot_height))
    fig.subplots_adjust(wspace=0.1, hspace=0.2)

    # plotting variables (e.g. ticks, axis limits, colormaps, ...)
    colors = [behav_colors[0], behav_colors[1], behav_colors[3]]
    behlbs = ["Neither", "Whisking\nOnly", "Both"]
    xtks = np.round(np.arange(-0.2, 0.21, 0.1), 1)
    ytks = np.arange(0, 1.1, 0.25)
    
    x = np.arange(-0.2, 0.21, 0.005)
    pad = 1

    # main
    for irow, row in enumerate(axs):
        # store relevant data into variables
        spkmat = ephys_data[irow]["spkmat"]
        regIDs = ephys_data[irow]["regIDs"]
        
        # generate masks to extract neural activity based on behavioral state
        T_neither, T_whisk_only, T_lomot_only, T_both = label_tpoints(ephys_data, behav_data, mouseID=irow)
        
        # compute the ECDFs of the Silhouette coefficients
        ecdfs = [ECDF(silhouette_samples(spkmat[:, T], regIDs, metric="correlation"))(x) for T in [T_neither, T_whisk_only, T_both]]

        # plot ECDFs and KS-statistics
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

                
    if path:
        path = f"{root}/Plots/silhouette_kstest.png"
    plt.savefig(path, bbox_inches="tight", transparent=True)