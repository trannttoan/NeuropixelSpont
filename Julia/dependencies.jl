include("dataIO.jl")
using .DataIO: load_result, load_mouse_data, save_result
include("analysis.jl")
using .Analysis
include("visualization.jl")
using .Visualization


using MAT
using Plots, Measures; gr()
using LinearAlgebra, Random
using Statistics, StatsBase
using Clustering, Distances
using Ripserer
using InformationMeasures
