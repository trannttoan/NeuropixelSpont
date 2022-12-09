import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from matplotlib.gridspec import GridSpecFromSubplotSpec
from dependencies import root

def plot_behav_labels(
    behav_data,
    mouseID=0,
    colors=None,
    tstart=0,
    duration=50,
    tstep=10,
    roi_lw=4,
    save_plot=False

):
    
    # frame and ROIs
    roi_dict = loadmat(root + "/Data/source/rois.mat")
    frame = roi_dict["frames"][mouseID, 0].astype(int)
    wheel_mask = roi_dict["wheel_masks"][mouseID, 0].astype(bool)
    wheel_bound = roi_dict["wheel_bounds"][mouseID, 0].astype(int)
    nose_mask = roi_dict["nose_masks"][mouseID, 0].astype(bool)
    nose_bound = roi_dict["nose_bounds"][mouseID, 0].astype(int)
    
    wheel_frame = frame.copy()
    # wheel_frame[wheel_mask] = 255-wheel_frame[wheel_mask]
    nose_frame = frame.copy()
    # nose_frame[nose_mask] = 255-nose_frame[nose_mask]

    
    # behavior indices/labels
    srate = 40
    start = tstart * srate
    end = start + srate*duration
    
    T = behav_data[mouseID]["time_pts"][start:end]
    T = T - T[0]
    

    lomot_cont = behav_data[mouseID]["lomot_cont"][start:end]
    lomot_norm_coef = lomot_cont.max()
    lomot_cont = lomot_cont / lomot_norm_coef
    lomot_disc = behav_data[mouseID]["lomot_disc"][start:end]
    lomot_thresh = behav_data[mouseID]["lomot_thresh"] / lomot_norm_coef

    whisk_cont = behav_data[mouseID]["whisk_cont"][start:end]
    whisk_norm_coef = whisk_cont.max()
    whisk_cont = whisk_cont / whisk_norm_coef
    whisk_disc = behav_data[mouseID]["whisk_disc"][start:end]
    whisk_thresh = behav_data[mouseID]["whisk_thresh"] / whisk_norm_coef
    
    T_both = (lomot_disc * whisk_disc) == 1
    T_neither = (lomot_disc + whisk_disc) == 0
    T_lomot_only = (lomot_disc - whisk_disc) == 1
    T_whisk_only = (whisk_disc - lomot_disc) == 1
    
    lomot_color = colors[2]
    whisk_color = colors[1]
    fig, (ax_top, ax_bot, ax_extra) = plt.subplots(3, 1, figsize=(15, 8),
                                         gridspec_kw=dict(height_ratios=[7, 7, 1]))
    fig.subplots_adjust(hspace=0.5)
    
    # locomotion
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=ax_top, width_ratios=[1, 2], wspace=0.25)
    ax = plt.subplot(gs[0])
    ax.imshow(wheel_frame, cmap="gray", vmin=0, vmax=255)
    ax.scatter(*np.fliplr(np.argwhere(wheel_bound)).T, s=roi_lw, color=lomot_color)
    ax.set_axis_off()
    
    ax = plt.subplot(gs[1])
#     ax.axhline(lomot_thresh, color='red')
    ax.plot(T, lomot_cont, color="black")
    ax.fill_between(T, 0, 1, where=lomot_disc>0, color=lomot_color, alpha=0.8, transform=ax.get_xaxis_transform())
#     ax.set_yscale("log")
#     ax.set_ylim((0.1, 10.1))
    ax.margins(x=0)
    ax.set_xlabel("Time(s)", fontsize=16)
    ax.set_ylabel(r"$L^2$-norm (ROI)")
    ax.set_xticks(np.arange(0, duration+1, tstep))
    ax.set_yticks(np.arange(0, 1.1, 0.5))
    ax.tick_params(labelsize=15)
    
    # whisking
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=ax_bot, width_ratios=[1, 2], wspace=0.25)
    ax = plt.subplot(gs[0])
    ax.imshow(nose_frame, cmap="gray", vmin=0, vmax=255)
    ax.scatter(*np.fliplr(np.argwhere(nose_bound)).T, s=roi_lw, color=whisk_color)
    ax.set_axis_off()
    
    ax = plt.subplot(gs[1])
#     ax.axhline(whisk_thresh, color='red')
    ax.plot(T, whisk_cont, color="black")
    ax.fill_between(T, 0, 1, where=whisk_disc>0, color=whisk_color, alpha=0.8, transform=ax.get_xaxis_transform())
#     ax.set_yscale("log")
#     ax.set_ylim((0.6, 10.6))
    ax.margins(x=0)
    ax.set_ylabel(r"$L^2$-norm (ROI)")
    ax.set_xlabel("Time(s)", fontsize=16)
    ax.set_xticks(np.arange(0, duration+1, tstep))
    ax.set_yticks(np.arange(0, 1.1, 0.5))
    ax.tick_params(labelsize=15)
    
    fig.text(0.06, 0.725, "Locomotion", fontsize=17, ha="center")
    fig.text(0.06, 0.375, "Whisking", fontsize=17, ha="center")
    # fig.text(0.3, 0.13, "Behavior Labels", fontsize=17)
    
    fig.text(0.31, 0.63, "ROI", fontsize=17, ha="center", color="white")
    fig.text(0.295, 0.34, "ROI", fontsize=17, ha="center", color="white")
    
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=ax_extra, width_ratios=[1, 2], wspace=0.25)
    ax = plt.subplot(gs[1])

    for T_label, clr in zip([T_neither, T_whisk_only, T_lomot_only, T_both], colors):
        ax.fill_between(T, 0, 1, where=T_label>0, color=clr, transform=ax.get_xaxis_transform())
    ax.margins(x=0)
    ax.get_yaxis().set_visible(False)
    ax.set_xlabel("Time(s)", fontsize=16)
    ax.set_xticks(np.arange(0, duration+1, tstep))
    ax.tick_params(labelsize=15)
    
    if save_plot:
        plt.savefig(f"{root}/Plots/behavior_{tstart}-{tstart+duration}.png", bbox_inches="tight")