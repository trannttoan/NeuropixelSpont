import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpecFromSubplotSpec

from helper_functions import load_data, label_tpoints
from dependencies import root



def plot_neuron_positions(
    ephys_data,
    region_colors,
    psz=15,
    msz=15,
    hv=-125,
    vv=25,
    a=1,
    show_legend=True,
    save_plot=False
):
    
    fig, ax = plt.subplots(figsize=(psz, psz), subplot_kw={"projection":"3d"})
    
    grid_coords = np.load("../../../Data/source/brainGridData.npy")
    grid_coords = grid_coords[grid_coords.sum(axis=1)!=0, :] * 10
    max_z = grid_coords[:, 2].max()
    
    ax.scatter(grid_coords[:, 0], grid_coords[:, 1], max_z-grid_coords[:, 2], s=1, c="black", alpha=0.3)
    ax.set_axis_off()
    ax.view_init(azim=hv, elev=vv)
    ax.set_xlim((0, 12000))
    
    reglbs = ephys_data[0]["reglbs"]
    
    for imouse in range(3):
        neuron_coords = ephys_data[imouse]["coords"]
        regIDs = ephys_data[imouse]["regIDs"]
        
        clr = [region_colors[i-1] for i in regIDs]
        ax.scatter(neuron_coords[:, 0], neuron_coords[:, 2], max_z-neuron_coords[:, 1], c=clr, s=msz, alpha=a)
        
    if show_legend:
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
    
    if save_plot:
        plt.savefig(f"{root}/Plots/neuron_pos_{abs(hv)}-{abs(vv)}.png", bbox_inches="tight", transparent=True)



def plot_probe_positions(
    names,
    colors,
    psz=14,
    msz=10,
    hv=-125,
    vv=25,
    a=1,
    show_legend=True,
    save_plot=False
):
      
    fig, ax = plt.subplots(figsize=(psz, psz), subplot_kw={"projection":"3d"})
    
    # plot brain grid
    grid_coords = np.load("../../../Data/source/brainGridData.npy")
    grid_coords = grid_coords[grid_coords.sum(axis=1)!=0, :] * 10
    max_z = grid_coords[:, 2].max()
    
    ax.scatter(grid_coords[:, 0], grid_coords[:, 1], max_z-grid_coords[:, 2], s=1, c="black", alpha=0.3)
    ax.set_axis_off()
    ax.view_init(azim=hv, elev=vv)
    ax.set_xlim((0, 12000))
    
    # plot probes
    loc_dict = loadmat("../../../Data/source/probeLocations.mat")
    
    for imouse in range(3):
        for locs in loc_dict[names[imouse]].flatten():
            ax.scatter(locs[:, 0], locs[:, 2], max_z-locs[:, 1],
                       color=colors[imouse], s=msz)

    
    if show_legend:
        h = [Line2D(
            [0], [0],
            marker='s',
            markersize=22,
            markerfacecolor=colors[i],
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
    
    if save_plot:
        plt.savefig(f"{root}/Plots/probe_pos_{abs(hv)}-{abs(vv)}.png", bbox_inches="tight", transparent="True")


def plot_raster(
    ephys_data,
    mouseID=0,
    save_plot=False
):
    
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
            
    if save_plot:
        plt.savefig(f"{root}/Plots/raster.png", bbox_inches="tight")



def plot_neuron_dist_wrt_region(
    ephys_data,
    names,
    psz=6,
    fontsize=12,
    cmap=plt.cm.Paired,
    save_plot=False
):
    ncols = len(ephys_data)
    fig, axs = plt.subplots(1, ncols, figsize=(psz*ncols, psz))
    reglbs = ephys_data[0]["reglbs"]
    n_regs = len(reglbs)
    colors = [cmap(v) for v in np.linspace(0, 1, n_regs)]
    
    for imouse, ax in enumerate(axs):
        regIDs = ephys_data[imouse]["regIDs"]
        unq_regIDs = np.unique(regIDs)
        cnts = [(regIDs==id).sum() for id in unq_regIDs]
        lbs = [reglbs[id-1] for id in unq_regIDs]
        clrs = [colors[id-1] for id in unq_regIDs]
        
        ax.pie(cnts,
               labels=lbs,
               labeldistance=1.05,
               rotatelabels=True,
               colors=clrs,
               startangle=90,
               textprops={"fontsize":fontsize}
        )
        ax.axis("equal")
        ax.set_title(names[imouse], y=-0.1, fontsize=20)
        
    if save_plot:
        plt.savefig(f"{root}/Plots/regions.png", bbox_inches="tight")


def plot_neurons_wrt_region(
    ephys_data,
    names,
    psz=8,
    cmap=plt.cm.Paired,
    save_plot=False
):
    reglbs = ephys_data[0]["reglbs"]
    n_regs = len(reglbs)
    fig, axs = plt.subplots(1, 3, figsize=(psz*3, psz))
    
    for imouse, ax in enumerate(axs):
        regIDs = ephys_data[imouse]["regIDs"]
        
        unq_regIDs = np.unique(regIDs)
        counts = [(regIDs == rid).sum() for rid in unq_regIDs]
        clr = cmap((unq_regIDs-1) / n_regs-1)
    
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

    if save_plot:
        plt.savefig(f"{root}/Plots/regions_bar.png", bbox_inches="tight")


def plot_time_wrt_behavior(
    behav_data,
    names,
    colors,
    psz=6,
    adjust=False,
    save_plot=False
):
    ncols = len(behav_data)
    fig, axs = plt.subplots(1, ncols, figsize=(psz*ncols, psz))
    fig.subplots_adjust(wspace=0.4)
    
    for imouse, ax in enumerate(axs):
        T_neither, T_whisk_only, T_lomot_only, T_both = label_tpoints(behav_data, imouse, adjust=adjust)
        
        cnts = np.array([T_neither.sum(), T_whisk_only.sum(), T_lomot_only.sum(), T_both.sum()])
        n_tpts = cnts.sum()
        behs = ["No whisking or locomotion", "Whisking only", "Locomotion only", "Both whisking and locomotion"]
        lbs = [f"{(cnts[i]/n_tpts)*100:.2f}%" for i in range(len(behs))]
        exp = [0, 0, 0.1, 0]
        
        ax.pie(cnts,
               labels=lbs,
               labeldistance=1.07,
               textprops=dict(fontsize=15),
               startangle=0,
               colors=colors
        )
        ax.set_title(names[imouse], fontsize=20)
        
    h = [Line2D(
        [0], [0],
        marker='s',
        linewidth=0,
        markerfacecolor=colors[i],
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
        
    if save_plot:
        plt.savefig(f"{root}/Plots/beh_percent.png", bbox_inches="tight")