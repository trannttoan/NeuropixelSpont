# NeuropixelSpont
Collaborative space for analysis of datasets containing eight-probe Neuropixels recordings of spontaneous neural activity in mice. Current primary programming language is Julia.
### Load data using [MAT.jl](https://github.com/JuliaIO/MAT.jl)
```Julia
# Load all data in a three-element array of dictionaries
# Each element corresponding to each mouse maps variable names to their contents  
mice_names = ["Krebs", "Waksman", "Robbins"]
data = [MAT.matread("./Data/$(mice_names[i])withFaces_KS2.mat") for i in 1:3];
```
