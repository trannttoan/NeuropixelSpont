import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpecFromSubplotSpec

from behavior_extraction import label_tpoints
from dependencies import data_path, figure_path


def plot_neuron_positions(
    ephys_data,
    region_colors,
    view_angles=(-125, 25),
    plot_size=(15, 15)
):
    """
    Plot anatomical locations of neurons in Allen Common Coordinnate Framework

    Parameters
    ----------
    ephys_data : list
        Dictionaries containing the processed neural activity and neuron locations
    region_colors : list
        List of colors assigned to brain regions for plotting
    view_angles : tuple, default=(-125, 25)
        3D view angle
    plot_size : tuple, default=(15, 15)
        Width and height of the plot

    """

    width, height = plot_size
    fig, ax = plt.subplots(figsize=(width, height), subplot_kw={"projection":"3d"})
    
    grid_coords = np.load(f"{data_path}/brainGridData.npy")
    grid_coords = grid_coords[grid_coords.sum(axis=1)!=0, :] * 10
    max_z = grid_coords[:, 2].max()
    hv, vv = view_angles
    
    ax.scatter(grid_coords[:, 0], grid_coords[:, 1], max_z-grid_coords[:, 2], s=1, c="black", alpha=0.3)
    ax.set_axis_off()
    ax.view_init(azim=hv, elev=vv)
    ax.set_xlim((0, 12000))
    
    reglbs = ephys_data[0]["reglbs"]
    
    for imouse in range(3):
        neuron_coords = ephys_data[imouse]["coords"]
        regIDs = ephys_data[imouse]["regIDs"]
        
        clr = [region_colors[i-1] for i in regIDs]
        ax.scatter(neuron_coords[:, 0], neuron_coords[:, 2], max_z-neuron_coords[:, 1], c=clr, s=15, alpha=1)
        
    h = [Line2D(
        [0], [0],
        marker='o',
        markersize=20,
        markerfacecolor=cl,
        color='w',
        markeredgecolor='w',
        label=lb
    ) for cl, lb in zip(region_colors, reglbs)]
    
    fig.legend(
        handles=h,
        ncol=4,
        prop={'size':20},
        framealpha=0,
        columnspacing=0.5,
        handletextpad=0.25,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.215))
    
    plt.savefig(f"{figure_path}/neuron_{abs(hv)}-{abs(vv)}.png", bbox_inches="tight", transparent=True)



def plot_probe_positions(
    names,
    mice_colors,
    view_angle=(-125, 25),
    plot_size=(14, 14)
):
    """
    Plot positions Neuropixels probes in Allen Common Coordinnate Framework

    Parameters
    ----------
    ephys_data : list
        Dictionaries containing the processed neural activity and neuron locations
    mice_colors : list
        List of colors assigned to mice for plotting
    view_angles : tuple, default=(-125, 25)
        3D view angle
    plot_size : tuple, default=(14, 14)
        Width and height of the plot
    """

    width, height = plot_size
    fig, ax = plt.subplots(figsize=(width, height), subplot_kw={"projection":"3d"})
    
    # plot brain grid
    grid_coords = np.load(f"{data_path}/brainGridData.npy")
    grid_coords = grid_coords[grid_coords.sum(axis=1)!=0, :] * 10
    max_z = grid_coords[:, 2].max()
    hv, vv = view_angle

    ax.scatter(grid_coords[:, 0], grid_coords[:, 1], max_z-grid_coords[:, 2], s=1, c="black", alpha=0.3)
    ax.set_axis_off()
    ax.view_init(azim=hv, elev=vv)
    ax.set_xlim((0, 12000))
    
    # plot probes
    loc_dict = loadmat(f"{data_path}/probeLocations.mat")
    
    for imouse in range(3):
        for locs in loc_dict[names[imouse]].flatten():
            ax.scatter(locs[:, 0], locs[:, 2], max_z-locs[:, 1],
                       color=mice_colors[imouse], s=10)

    h = [Line2D(
        [0], [0],
        marker='s',
        markersize=22,
        markerfacecolor=mice_colors[i],
        color='w',
        markeredgecolor='w',
        label=names[i]
    ) for i in range(len(names))]
    
    fig.legend(
        handles=h,
        ncol=len(names),
        prop={'size':22},
        framealpha=0,
        columnspacing=0.5,
        handletextpad=0.25,
        loc="lower center",
        bbox_to_anchor=(0.55, 0.26))

    plt.savefig(f"{figure_path}/probe_{abs(hv)}-{abs(vv)}.png", bbox_inches="tight", transparent="True")


def plot_raster(
    ephys_data,
    mouseID
):
    """
    Plot heatmaps of neural time series grouped by brain regions

    Parameters
    ----------
    ephys_data : list
        Dictionaries containing the processed neural activity and neuron locations
    mouseID : int
        Index number of a mouse (0-Krebs, 1-Waksman, 2-Robbins)

    """
    
    spkmat = ephys_data[mouseID]["spkmat"]
    tpoints = ephys_data[mouseID]["tpoints"]
    srate = 1 / ephys_data[mouseID]["timestep"]
    regIDs = ephys_data[mouseID]["regIDs"]
    unq_regIDs = np.unique(regIDs)
    reglbs = ephys_data[mouseID]["reglbs"]
    n_regs = len(reglbs)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    ratios = np.array([np.sum(regIDs == id) for id in unq_regIDs])
    ratios = np.round(ratios / ratios.min(), 2)
    xtks = np.arange(0, tpoints.size, 5e3)
    xtklbs = (xtks / srate).flatten().astype(int)
    
    gs = GridSpecFromSubplotSpec(unq_regIDs.size, 1, subplot_spec=ax, height_ratios=ratios, hspace=0.1)
    for i, cell in enumerate(gs):
        rid = unq_regIDs[i]
        cond = regIDs==rid
        count = cond.sum()
        ax = plt.subplot(cell)
        ax.imshow(spkmat[cond, :], aspect="auto", cmap="Greys", vmax=1)
        ax.get_yaxis().set_visible(False)
        ax.text(x=-300, y=int(count/2), s=f"{reglbs[rid-1]} ({count})",
                fontsize=15, ha="right", va="center")
        
        if i == unq_regIDs.size-1:
            ax.set_xticks(xtks)
            ax.set_xticklabels(xtklbs)
            ax.set_xlabel("Time(s)")
        else:
            ax.get_xaxis().set_visible(False)
            
    plt.savefig(f"{figure_path}/raster.png", bbox_inches="tight", transparent=True)



def plot_region_sample_sizes(
    ephys_data,
    names,
    plot_size=(6, 8)
):
    """
    Plot the sample sizes of brain regions for each mouse

    Parameters
    ----------
    ephys_data : list
        Dictionaries containing the processed neural activity and neuron locations
    names : list 
        Names of mice
    plot_size : tuple, default=(6, 8)
        Width and height of the plot

    """
    width, height = plot_size
    reglbs = ephys_data[0]["reglbs"]
    fig, axs = plt.subplots(1, len(names), figsize=(width*3, height))
    
    for imouse, ax in enumerate(axs):
        regIDs = ephys_data[imouse]["regIDs"]
        
        unq_regIDs = np.unique(regIDs)
        counts = [(regIDs == rid).sum() for rid in unq_regIDs]
    
        ax.bar(unq_regIDs, counts, color="blue")
        ax.set_title(names[imouse], fontsize=18)
        ax.set_xticks(np.arange(1, 13))
        ax.set_xticklabels(reglbs, rotation=90, fontsize=15)
        ax.set_yticks(np.arange(0, 1800, 500))
        ax.tick_params(axis='y', labelsize=15)
        ax.set_xlim((0, 13))
        ax.set_ylim((0, 1900))
        
        if imouse==0:
            ax.set_ylabel("Number of neurons", fontsize=18)

    plt.savefig(f"{figure_path}/regions_bar.png", bbox_inches="tight", transparent=True)


def plot_time_wrt_behavior(
    ephys_data,
    behav_data,
    names,
    behav_colors,
    plot_size=(6, 6),
    adjust=False
):
    """
    Plot percent of time spent in each behavioral state

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
    plot_size : tuple, default=(6, 6)
        Width and height of the plot
    adjust : bool
        "Running Only" is subsumed into "Both" if true
    """
    ncols = len(behav_data)
    width, height = plot_size
    fig, axs = plt.subplots(1, ncols, figsize=(width*ncols, height))
    fig.subplots_adjust(wspace=0.4)
    
    for imouse, ax in enumerate(axs):
        T_neither, T_whisk_only, T_lomot_only, T_both = label_tpoints(ephys_data, behav_data, imouse, adjust=adjust)
        
        cnts = np.array([T_neither.sum(), T_whisk_only.sum(), T_lomot_only.sum(), T_both.sum()])
        n_tpts = cnts.sum()
        behs = ["No whisking or running", "Whisking only", "Running only", "Both whisking and running"]
        lbs = [f"{(cnts[i]/n_tpts)*100:.2f}%" for i in range(len(behs))]
        exp = [0, 0, 0.1, 0]
        
        ax.pie(cnts,
               labels=lbs,
               labeldistance=1.07,
               textprops=dict(fontsize=15),
               startangle=0,
               colors=behav_colors
        )
        ax.set_title(names[imouse], fontsize=20)
        
    h = [Line2D(
        [0], [0],
        marker='s',
        linewidth=0,
        markerfacecolor=behav_colors[i],
        markersize=20,
        label=behs[i]
    ) for i in range(len(behs))]
    
    fig.legend(
        handles=h,
        ncol=len(behs),
        prop={"size":18},
        framealpha=0,
        columnspacing=1,
        handletextpad=0.25,
        loc="lower center",
        bbox_to_anchor=(0.52, 0.1)
    )

    plt.savefig(f"{figure_path}/behavior_percent.png", bbox_inches="tight", transparent=True)