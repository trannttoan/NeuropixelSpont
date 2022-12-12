# NeuropixelSpont
Analyses of multi-brain Neuropixels recordings of mice during spontaneous behaviors.

The dataset is publicly available at [link](https://janelia.figshare.com/articles/dataset/Eight-probe_Neuropixels_recordings_during_spontaneous_behaviors/7739750), consisting of electrophysiological recordings (plus neuron locations, brain regions, etc.) and video recordings.

The MATLAB scripts `processNeuralActivity.m` and `createBehaviorLabels.m` process the neural data and videos respectively. Make sure to change the path variables in the scripts for them to properly load the downloaded data and save the processed data at desired locations on the local machine. The results include `ephys_[mouse_name].mat` (neural data) and `behav_[mouse_name].mat` (behavioral data) for every mouse which should all be in one folder. Finally, all processed data can be loaded into workspace using the function `load_data` from `helper_functions.py`.

```
from helper_functions import load_data

# ephys_data : list of dictionaries, each containing the processed neural activity and supplementary information of a mice.
# behav_data : list of dictionaries, each containing the behavioral variables extracted from videos.
# names      : list of names of mice.
ephys_data, behav_data, names = load_data()

# mice_colors   : list of colors, one for each mouse.
# region_colors : list of colors, one for each brain region.
# behav_colors  : list of colors, one for each behavioral state.
mice_colors, region_colors, behav_colors = generate_colors()

# Customize default matplotlib parameters
plot_config()

```

The variable described previously are used as inputs for the documented plotting functions in the python files.



