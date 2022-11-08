import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.graph_objects as go
import ipywidgets as widgets
import networkx as nx
import h5py

from scipy.io import loadmat, savemat
from os.path import isfile

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import make_blobs

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import average, dendrogram, cophenet
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
plt.style.use("seaborn-paper")

# from umap import UMAP, AlignedUMAP

from networkx.algorithms.community import modularity

