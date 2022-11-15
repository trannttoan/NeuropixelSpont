import numpy as np
import matplotlib.pyplot as plt
import h5py

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import average, dendrogram
from matplotlib.lines import Line2D

from helper_functions import load_data, label_tpoints, generate_colors, plot_config
from dependencies import root





ephys_data, behav_data, names = load_data()
mice_colors, region_colors, behav_colors = generate_colors()
plot_config()
