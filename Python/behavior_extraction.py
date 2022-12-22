import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy.interpolate import interp1d

from matplotlib.gridspec import GridSpecFromSubplotSpec
from dependencies import data_path, figure_path


def label_tpoints(
    ephys_data,
    behav_data,
    mouseID,
    adjust=True
):
    """
    Align behavioral event indicators with neural recording time and create a mask for each behavioral state. 

    Parameters
    ----------
    ephys_data : list
        Dictionaries containing the processed neural activity and neuron locations
    behav_data : list
        Dictionaries containing the behavioral variables extracted from videos
    mouseID : int
        Index number of a mouse (0-Krebs, 1-Waksman, 2-Robbins)
    adjust: bool, default=True
        "Running Only" is subsumed into "Both" if true

    Returns
    -------
    T_neither : array-like
        1D mask for "Both" behavioral state (no whisking or running).
    T_whisk_only : array-like
        1D mask for "Whisking Only" behavioral state (only whisking).
    T_lomot_only : array-like
        1D mask for "Locomotion Only" behavioral state (only running).
    T_both : array-like
        1D mask for "Both" behavioral state (whisking and running).

    """
    
    lomot_disc_raw = behav_data[mouseID]["lomot_disc"]
    whisk_disc_raw = behav_data[mouseID]["whisk_disc"]

    video_times = behav_data[mouseID]["time_pts"]
    ephys_times = ephys_data[mouseID]["tpoints"]

    # interpolate behavioral labels to align to neural recording time
    f = interp1d(video_times, lomot_disc_raw, kind="nearest")
    lomot_disc = f(ephys_times)
    f = interp1d(video_times, whisk_disc_raw, kind="nearest")
    whisk_disc = f(ephys_times)

    # mask for each behavioral state
    T_both = (lomot_disc * whisk_disc) == 1
    T_neither = (lomot_disc + whisk_disc) == 0
    T_lomot_only = (lomot_disc - whisk_disc) == 1
    T_whisk_only = (whisk_disc - lomot_disc) == 1
    
    if adjust:
        T_both = T_both + T_lomot_only
        T_lomot_only = np.zeros(T_both.size).astype(bool)  
    
    return T_neither, T_whisk_only, T_lomot_only, T_both



def plot_behavior_variables(
    behav_data,
    mouseID,
    behav_colors,
    tstart=0,
    duration=50,
    interval=10
):
    """
    Plot example frames along with ROIs and corresponding behavior variables

    Parameters
    ----------
    behav_data : list 
        Dictionaries containing the behavioral variables extracted from videos
    mouseID : int
        Index number of a mouse (0-Krebs, 1-Waksman, 2-Robbins)
    behav_colors : list
        List of colors assigned to behavioral states
    tstart : int, default=0
        Start time
    duration : int, default=50
        Length of time to display
    interval : int, interval=10
        Size of interval between ticks
    """

    # frame and ROIs
    roi_dict = loadmat(f"{data_path}/rois.mat")
    frame = roi_dict["frames"][mouseID, 0].astype(int)

    # time window to show
    srate = 40
    start = tstart * srate
    end = start + srate*duration
    
    T = behav_data[mouseID]["time_pts"][start:end]
    T = T - T[0]

    # create behavioral state indicators
    lomot_disc = behav_data[mouseID]["lomot_disc"][start:end]
    whisk_disc = behav_data[mouseID]["whisk_disc"][start:end]
    
    T_both = (lomot_disc * whisk_disc) == 1
    T_neither = (lomot_disc + whisk_disc) == 0
    T_lomot_only = (lomot_disc - whisk_disc) == 1
    T_whisk_only = (whisk_disc - lomot_disc) == 1


    # initialize and adjust axes
    fig, (ax_top, ax_bot, ax_extra) = plt.subplots(3, 1, figsize=(15, 8),
                                         gridspec_kw=dict(height_ratios=[7, 7, 1]))
    fig.subplots_adjust(hspace=0.5)
    
    # main
    for pax, beh_str, mask_str, clr in zip([ax_top, ax_bot], ["whisk", "lomot"], ["nose", "wheel"], behav_colors[1:3]): 
        beh_cont = behav_data[mouseID][f"{beh_str}_cont"][start:end]
        beh_cont = beh_cont / beh_cont.max()
        beh_disc = behav_data[mouseID][f"{beh_str}_disc"][start:end]
        roi = roi_dict[f"{mask_str}_bounds"][mouseID, 0].astype(int)

        gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=pax, width_ratios=[1, 2], wspace=0.25)
        
        # frame
        ax = plt.subplot(gs[0])
        ax.imshow(frame, cmap="gray", vmin=0, vmax=255)
        ax.scatter(*np.fliplr(np.argwhere(roi)).T, s=5, color=clr)
        ax.set_axis_off()
        
        # behavior variables
        ax = plt.subplot(gs[1])
        ax.plot(T, beh_cont, color="black")
        ax.fill_between(T, 0, 1, where=beh_disc>0, color=clr, alpha=0.8, transform=ax.get_xaxis_transform())
        
        ax.margins(x=0)
        ax.set_xlabel("Time(s)", fontsize=16)
        ax.set_ylabel(r"$L^2$-norm (ROI)")
        ax.set_xticks(np.arange(0, duration+1, interval))
        ax.set_yticks(np.arange(0, 1.1, 0.5))
        ax.tick_params(labelsize=15)
    
    fig.text(0.072, 0.725, "Whisking", fontsize=17, ha="center")
    fig.text(0.072, 0.375, "Running", fontsize=17, ha="center")
    
    fig.text(0.31, 0.26, "ROI", fontsize=17, ha="center", color="white")
    fig.text(0.295, 0.69, "ROI", fontsize=17, ha="center", color="white")
    
    # behavioral state indicators
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=ax_extra, width_ratios=[1, 2], wspace=0.25)
    ax = plt.subplot(gs[1])
    for T_label, clr in zip([T_neither, T_whisk_only, T_lomot_only, T_both], behav_colors):
        ax.fill_between(T, 0, 1, where=T_label>0, color=clr, transform=ax.get_xaxis_transform())

    ax.margins(x=0)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("Time(s)", fontsize=16)
    ax.set_xticks(np.arange(0, duration+1, interval))
    ax.tick_params(labelsize=15)
    
    plt.savefig(f"{figure_path}/behavior_{tstart}-{tstart+duration}.png", bbox_inches="tight", transparent=True)