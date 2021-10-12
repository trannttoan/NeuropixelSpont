module Visualization

using Plots, Measures; gr()
using Statistics: mean, std, cor
using Distances: pairwise, Euclidean, CorrDist
using Clustering: hclust
using Ripserer: ripserer

include("dataIO.jl")
using .DataIO: load_mouse_data, load_result
include("analysis.jl")
using .Analysis: neuron_sort

"""
    data_dist(dtname::Symbol)

Plot histograms of specified data, one for each mouse. Options include :region, :probe, :rate, :count.
"""
function data_dist(dtname::Symbol, save::Bool=false)
    names = ["Krebs", "Waksman", "Robbins"]
    plot_holder = Array{Plots.Plot{Plots.GRBackend}}(undef, 3)

    # distribution of brain region labels
    if dtname == :region
        reglbs = load_mouse_data(1, "areaLabels")
        for i=1:length(names)
            plot_holder[i] = histogram(load_mouse_data(i, "brainLoc"), nbin=1:15, title=names[i])
        end

        plot(plot_holder..., xticks=(1:14, reglbs), xrotation=90, xlabel="Brain region", yaxis=("Number of neurons", (0, 2000)),
             guidefontsize=10, layout=(1, 3), size=(900, 400), grid=false, legend=false, left_margin=2mm, bottommargin=10mm)

    # distribution of probe IDs
    elseif dtname == :probe
        for i=1:length(names)
            plot_holder[i] = histogram(load_mouse_data(i, "iprobe"), nbin=8, title=names[i])
        end

        plot(plot_holder..., xlabel="Probe", yaxis=("Number of neurons", (0, 700)),
             guidefontsize=10, layout=(1, 3), size=(900, 400), grid=false, legend=false, left_margin=2mm, bottom_margin=5mm)

    # distribution of spike-count rate
    elseif dtname == :rate
        for i=1:length(names)
            period = load_mouse_data(i, "tspont")
            spkcounts = load_mouse_data(i, "stall")
            plot_holder[i] = histogram(sum(spkcounts, dims=2)/(period[end]-period[1]), nbin=0:1:60, title=names[i])
        end

        plot(plot_holder..., xaxis=("Firing Rate", (0, 60)), yaxis=("Number of neurons", (0, 600)),
             guidefontsize=10, layout=(1, 3), size=(900, 400), grid=false, legend=false, left_margin=2mm, bottom_margin=5mm)

    # distribution of number of unique spike count
    elseif dtname == :count
        for i=1:length(names)
            spkcounts = load_mouse_data(i, "stall")
            unq_spkcounts = [length(unique(spkcounts[j, :])) for j in 1:size(spkcounts, 1)]
            plot_holder[i] = histogram(unq_spkcounts, nbin=0:2:30, title=names[i])
        end

        plot(plot_holder..., xaxis=("Firing Range", (0, 30)), yaxis=("Number of neurons", (0, 600)),
             guidefontsize=10, layout=(1, 3), size=(900, 400), grid=false, legend=false, left_margin=2mm, bottom_margin=5mm)
    
    elseif dtname == :fluct
        for i=1:length(names)
            spkcounts = load_mouse_data(i, "stall")
            avgdiff = mean(abs.(spkcounts[:, 2:end] - spkcounts[:, 1:end-1]), dims=2)
            plot_holder[i] = histogram(avgdiff, nbin=0:0.1:1.5, title=names[i])
        end
        
        plot(plot_holder..., xaxis=("Fluctuation", (0, 2)), yaxis=("Number of neurons", (0, 1000)),
        guidefontsize=10, layout=(1, 3), size=(900, 400), grid=false, legend=false, left_margin=2mm, bottom_margin=5mm)
    elseif dtname == :stdv
        for i=1:length(names)
            spkcounts = load_mouse_data(i, "stall")
            stdv = std(spkcounts, dims=2)
            plot_holder[i] = histogram(stdv, nbin=0:0.1:2.5, title=names[i])
        end

        plot(plot_holder..., xaxis=("Standard Deviation", (0, 2.5)), yaxis=("Number of neurons", (0, 500)),
        guidefontsize=10, layout=(1, 3), size=(900, 400), grid=false, legend=false, left_margin=2mm, bottom_margin=5mm)
    end

    if save
        savefig("C:/Users/trann/Google Drive/Research/NeuropixelSpont/Plots/$(dtname).png")
    end
end


"""
    height_plot(mouseID::Int)

Plot relative heights of neurons on probes.
"""
function height_plot(mouseID::Int)
    hop_all = load_mouse_data(mouseID, "Wh")              # relative height of each neuron on a probe
    probe_ids = load_mouse_data(mouseID, "iprobe")        # array of corresponding probe IDs

    plot_array = Array{Plots.Plot{Plots.GRBackend}}(undef, 8*2)

    for i=1:8
        hop_iprobe = hop_all[probe_ids .== i]
        n_neuron_iprobe = length(hop_iprobe)
        rep_mat = repeat(hop_iprobe, 1, n_neuron_iprobe)
        dist_mat = abs.(rep_mat - transpose(rep_mat))
        prox_mat = 1 .- (dist_mat ./ maximum(dist_mat))

        plot_array[i*2-1] = heatmap(prox_mat, title="M$mouseID-P$i", yflip=true)
        plot_array[i*2] = plot(hop_iprobe, title="M$mouseID-P$i", xlabel="Neuron", ylabel="Height", legend=false)
    end

    display(plot(plot_array[1:8]..., layout=(4, 2), size=(1250, 1800), left_margin=8mm))
    display(plot(plot_array[9:16]..., layout=(4, 2), size=(1250, 1800), left_margin=8mm))
end


"""
    raster_plot(mouseID::Int, clust::Bool=false, binary::Bool=false)

Create raster plots of neural activity in a mouse grouped by brain regions.

...
# Argument
- mouseID : ID number of mouse
- clust   : determine whether neurons within each group is clustered.
- binary  : convert the non-binary spike count vectors to binary before plotting (all positive values are considered the same).
"""
function raster_plot(mouseID::Int; clust::Bool=false, binary::Bool=false)
    spkcounts = load_mouse_data(mouseID, "stall")           # spike count matrix
    regIDs = load_mouse_data(mouseID, "brainLoc")           # ID of region each neuron is in
    reglbs = load_mouse_data(mouseID, "areaLabels")         # corresponding region labels

    # extract region IDs
    unique_regIDs = convert(Array{Int64, 1}, sort(unique(regIDs)))
    n_regs = length(unique_regIDs)

    # for plotting
    plot_holder = Array{Plots.Plot{Plots.GRBackend}}(undef, n_regs)
    plot_heights = Array{Float64, 1}(undef, n_regs)
    N, T = size(spkcounts)      # N : number of neurons, T : number of samples/spike-train length
    cap = binary ? 1 : 36       # 36 is the highest number of spikes for a single sample and single neuron

    for i=1:n_regs
        # extract spike count arrays based on region
        fltr = regIDs .== unique_regIDs[i]
        N_reg = sum(fltr)
        plot_heights[i] = round(N_reg/N; digits=3)
        spkcounts_reg = spkcounts[fltr, :]

        # hierarchical clustering (if specified)
        if clust
            D = pairwise(CorrDist(), spkcounts_reg, dims=1)
            res = hclust(D, linkage=:average)
            sort_idx = res.order
        else
            sort_idx = 1:N_reg
        end

        plot_holder[i] = heatmap(spkcounts_reg[sort_idx, :], title=reglbs[unique_regIDs[i]], clims=(0, cap),
                                 color=cgrad(:grays, rev=true), colorbar=false, axis=false, ticks=false)
    end

    plot(plot_holder..., titleposition=:left, titlefontsize=30, top_margin=5mm,
         layout=grid(n_regs, 1, heights=plot_heights), size=(round(T/10), N))

    savefig("../Plots/spktrain_all_M$(mouseID).png")
end


"""
    regavg_plot(mouseID::Int, interval::Tuple{Int, Int}=(0, 0))

Plot average spike count array of each brain region during a specific period.
"""
function regavg_plot(mouseID::Int, interval::Tuple{Int, Int}=(0, 0))
    spkcounts = load_mouse_data(mouseID, "stall")       # spike count matrix
    regIDs = load_mouse_data(mouseID, "brainLoc")       # ID of region each neuron is in
    reglbs = load_mouse_data(mouseID, "areaLabels")     # corersponding region labels

    N, T = size(spkcounts)      # N : number of neurons, T : number of samples

    # input handling
    if interval == (0, 0)
        lowb = 1
        upb = T
    else
        if interval[1] >= 1 && interval[2] <= T && interval[1] < interval[2]
            lowb = interval[1]
            upb = interval[2]
        else
            lowb = 1
            upb = T
        end
    end

    # extract region IDs
    unique_regIDs = convert(Array{Int64, 1}, sort(unique(regIDs)))
    n_regs = length(unique_regIDs)

    # for plotting
    plot_holder = Array{Plots.Plot{Plots.GRBackend}}(undef, n_regs)
    plot_heights = ones(n_regs) * round(1/n_regs, digits=3)

    for i=1:n_regs
        fltr = regIDs .== unique_regIDs[i]
        spkcounts_reg = spkcounts[fltr, lowb:upb]
        spktrain_avg = transpose(mean(spkcounts_reg, dims=1))
        spktrain_std = transpose(std(spkcounts_reg, dims=1, corrected=false))
        plot_holder[i] = plot(spktrain_avg, ribbon=spktrain_std, fillalpha=0.5, title=reglbs[unique_regIDs[i]], ylim=(-1,4),
                              ticks=false, legend=false, grid=false, widen=false)
    end

    plot(plot_holder..., titleposition=:left, titlefontsize=20,
         layout=grid(n_regs, 1, heights=plot_heights), size=(round((upb-lowb+2)/10), N))
    savefig("../Plots/spktrain_avg_M$(mouseID).png")
end


function _reg_diff(regIDs::Array{Float64, 1}, identify_region::Bool)
    rep = repeat(regIDs, 1, length(regIDs))
    regsim = (rep - transpose(rep)) .== 0
    output = identify_region ? rep .* regsim  : regsim

    return output
end

function _locate_borders(regIDs)
    N_reg = 14
    pos = zeros(N_reg-1)
    pos[1] = sum(regIDs .== 1)

    for i=2:N_reg-1
        pos[i] = pos[i-1] + sum(regIDs .== i)
    end

    return pos
end

function _heatmap_mod(M, lim, irow, icol, borpos=nothing)
    row_titles = ["Krebs", "Waksman", "Robbins"]
    col_titles = ["Pearson's r", "Mutual Info(log)", "Physical Distance"]

    l_margin = icol == 1 ? 90mm : 10mm 
    p = heatmap(M, clim=lim, left_margin=l_margin)
    
    if irow==1
        title!(col_titles[icol])
    end

    if borpos !== nothing
        p = vline!(borpos, linecolor=:black, legend=false)
        p = hline!(borpos, linecolor=:black, legend=false)
    end

    return p
end

"""
    neuron_relate(mouseID::Int, sortSchemeName::String, sortScheme::Array{Int, 1}, includeBorder::Bool=false, identifyRegion::Bool=false)

Plot matrices of pairwise correlation, mutual information and physical distance. Plots arranged in 2 rows each corresponding to an ordering of neurons.
In the top row, neurons are grouped by brain region whereas hierarchical clustering is applied to those in the bottom row.
"""
function neuron_relate(mouseID::Int, sort_name::String, sort_scheme::Array{Int, 1}; include_border::Bool=false, identify_region::Bool=false)
    # load data
    regIDs = load_mouse_data(mouseID, "brainLoc")
    coords = load_mouse_data(mouseID, "ccfCoords")

    # compute "relation" matrices
    corM = load_result(mouseID, "cor")
    logmiM = log.(load_result(mouseID, "minfo"))
    pdistM = pairwise(Euclidean(), coords, dims=1)
    regM = _reg_diff(regIDs, identify_region)

    # plot configuration
    border_pos =  _locate_borders(regIDs)
    # clr = identify_region ? cgrad(:gist_earth, 15, rev=true, categorical=true) : cgrad(:heat, 2, categorical=true)
    # clm = identify_region ? (-1, 14) : (0, 1)

    regsort_name, regsort_scheme = neuron_sort(regIDs)

    # default
    p1_dflt = _heatmap_mod(corM[regsort_scheme, regsort_scheme], (-0.3, 1), "M$(mouseID)-cor-" * regsort_name, border_pos)
    p2_dflt = _heatmap_mod(logmiM[regsort_scheme, regsort_scheme], (-20, 1.1), "M$(mouseID)-logmi-" * regsort_name, border_pos)
    p3_dflt = _heatmap_mod(pdistM[regsort_scheme, regsort_scheme], (0, 6050), "M$(mouseID)-pdist-" * regsort_name, border_pos)
    # p4_dflt = heatmap(regM[regsort_scheme, regsort_scheme], clim=clm, title="M$(mouseID)-reg-" * regsort_name, color=clr)

    border_pos = sort_name != "HClustAll" && include_border ? _locate_borders(regIDs) : nothing

    # clustered
    p1_clst = _heatmap_mod(corM[sort_scheme, sort_scheme], (-0.3, 1), "M$(mouseID)-cor-" * sort_name, border_pos)
    p2_clst = _heatmap_mod(logmiM[sort_scheme, sort_scheme], (-20, 1.1), "M$(mouseID)-logmi-" * sort_name, border_pos)
    p3_clst = _heatmap_mod(pdistM[sort_scheme, sort_scheme], (0, 6050), "M$(mouseID)-pdist-" * sort_name, border_pos)
    # p4_clst = heatmap(regM[sort_scheme, sort_scheme], clim=clm, title="M$(mouseID)-reg-" * sort_name, color=clr)

    plot(p1_dflt, p2_dflt, p3_dflt, p1_clst, p2_clst, p3_clst,
         layout=(2, 3), size=(1700*3, 1200*2), margin=5mm,
         tickfontsize=40, ticks=false, yflip=true)
    annotate!(-1.8, 200, text("Before", :left, 10), subplot=6)
    savefig("C:/Users/trann/Google Drive/Research/NeuropixelSpont/Plots/M$(mouseID)_mat.png")
end


"""
    common_region_PD(maxDim::Int)

Plot persistence diagrams constructed from neural activity in common regions between the mouse datasets.
"""
function common_region_PD(maxDim::Int)
    N_mouse = 3
    N_reg = 14
    reglbs = load_mouse_data(1, "areaLabels")       # all region labels (same for three mice)

    # load data
    spkcounts_all = [load_mouse_data(i, "stall") for i in 1:N_mouse]
    regIDs_all = [load_mouse_data(i, "brainLoc") for i in 1:N_mouse]

    # find common regions between mouse datasets
    common_reg_set = Set(1:N_reg)
    for i=1:N_mouse
        intersect!(common_reg_set, Set(unique(regIDs_all[i])))
    end
    common_regs = sort([i for i in common_reg_set])
    N_common_reg = length(common_regs)

    # create and plot persistence diagrams
    plot_holder = Array{Plots.Plot{Plots.GRBackend}}(undef, N_common_reg)
    shapes = [:circle, :star4, :rect, :diamond, :star5]
    colors = palette(:default, N_common_reg)

    for i=1:N_common_reg
        plot_temp = plot()
        for j=1:N_mouse
            flt = regIDs_all[j] .== common_regs[i]
            D = sqrt.(pairwise(CorrDist(), spkcounts_all[j][flt, :], dims=1))
            ph = ripserer(D, dim_max=maxDim)
            for k=1:length(ph)
                if length(ph[k]) > 0
                    plot_temp = plot!(ph[k], markershape=shapes[k], markercolor=colors[j], title=reglbs[common_regs[i]])
                end
            end
        end
        plot_holder[i] = plot_temp
    end

    plot(plot_holder..., layout=(1,N_common_reg), size=(3000, 600),
         ylims=(-0.05, 1), xlims=(-0.05, 1),
         markersize=5, guidefontsize=15, legendfontsize=10,
         titlefontsize=18, tickfontsize=12,
         left_margin=10mm, right_margin=10mm)
    savefig("../Plots/comreg_HP$(maxDim).png")
end


"""
    probe3D(mouseID::Int, angle::Float64)

3D scatter plots of probe recording sites in which each site is represented by a point.
"""
function probe3D(mouseID::Int, angle::Float64)
    coords = load_mouse_data(mouseID, "ccfCoords")
    regIDs = load_mouse_data(mouseID, "brainLoc")
    regLabels = load_mouse_data(mouseID, "areaLabels")
    regColors = cgrad(:gist_earth, 15, rev=true, categorical=true)

    p = plot()
    for i=1:14
        filter = regIDs .== i
        if sum(filter) != 0
            p = scatter3d!(coords[filter, 1], coords[filter, 3], coords[filter, 2], label=regLabels[i],
                           camera=(angle, 55), markersize=3, markercolor=regColors[i+1], markerstrokewidth=0.2)
        end
    end

    plot(p, size=(1450, 1200))
    savefig("../Plots/M$(mouseID)_probes3D.png")
end


function _preprocess(mouseID::Int, mousename::String)
    spkcounts = load_mouse_data(mouseID, "stall")           # spike count matrix
    regIDs = load_mouse_data(mouseID, "brainLoc")           # ID of region each neuron is in
    reglbs = load_mouse_data(mouseID, "areaLabels")         # corresponding region labels
    frameDiff = load_mouse_data(mouseID, "video")           # norm of difference between consecutive frames

    ## extract region IDs
    unique_regIDs = convert(Array{Int64, 1}, sort(unique(regIDs)))
    n_regs = length(unique_regIDs)

    ## initilize arrays
    vholder = Array{Float64, 2}(undef, n_regs+2, size(spkcounts, 2))
    axlbs = Array{String, 1}(undef, n_regs+2)
    accum = zeros(size(spkcounts, 2))

    ## region averages
    for i=1:n_regs
        fltr = regIDs .== unique_regIDs[i]
        vholder[i, :] = mean(spkcounts[fltr, :], dims=1)
        axlbs[i] = reglbs[unique_regIDs[i]]
        accum = accum + vholder[i, :]
    end

    vholder[end-1, :] = accum
    vholder[end, :] = frameDiff
    axlbs[end-1] = "All"
    axlbs[end] = "Video"

    corM = cor(vholder, dims=2)

    return corM, axlbs
end

"""
    corWithBehaviors()

Plot matrices of pairwise correlation between region-average, weighted-population-average spike count series and consecutive video frame differences.
"""
function corWithBehaviors()
    names = load_mouse_data(1, "mstr")
    plot_holder = Array{Plots.Plot{Plots.GRBackend}}(undef, 3)

    for i=1:3
        correlmat, axeslabels = _preprocess(i, names[i])
        plot_holder[i] = heatmap(axeslabels, axeslabels, correlmat, title=names[i], clim=(-0.1, 1), tickfontsize=10)
    end

    plot(plot_holder..., layout=(1, 3), size=(2640, 600), yflip=true, xrotation=90, bottom_margin=21mm)
    savefig("../Plots/corWithBehavior.png")
end

#*********************************************************************************************************************
#--------------------------------------MAKING PLOTS FOR MANUSCIPTS/PRESENTATIONS--------------------------------------
#*********************************************************************************************************************

function data_dist3()
    names = ["Krebs", "Waksman", "Robbins"]
    plot_holder = Array{Plots.Plot{Plots.GRBackend}}(undef, 9)
    nrows = 3
    ncols = length(names)
    reglbs = load_mouse_data(1, "areaLabels")

    for i in 1:ncols
        spkcounts = load_mouse_data(i, "stall", true)
        period = load_mouse_data(i, "tspont")
        regIDs = load_mouse_data(i, "brainLoc", true)
        
        plot_holder[i] = histogram(sum(spkcounts, dims=2)/(period[end]-period[1]), nbin=0:1:60, title=names[i],
                                          xaxis=("Spike Count Rate", (0, 60)), yaxis=("Number of neurons", (0, 600)), bottom_margin=5mm)

        plot_holder[i+nrows] = histogram(std(spkcounts, dims=2), nbin=0:0.1:2.5, #title=names[i],
                                            xaxis=("Standard Deviation", (0, 2.5)), yaxis=("Number of neurons", (0, 500)), bottom_margin=5mm)

        plot_holder[i+nrows*2] = histogram(regIDs, nbin=1:15, #title=names[i],
                                            xticks=(1:14, reglbs), xrotation=90, xlabel="Brain region", yaxis=("Number of neurons", (0, 2000)), bottom_margin=5mm)

    end

    plot(plot_holder..., layout=(nrows, ncols), size=(500*ncols, 450*nrows),
         guidefontsize=10,grid=false, legend=false, left_margin=5mm)
    savefig("C:/Users/trann/Google Drive/Research/NeuropixelSpont/Plots/dist3.png")

end


"""
    neuron_relate(mouseID::Int, sortSchemeName::String, sortScheme::Array{Int, 1}, includeBorder::Bool=false, identifyRegion::Bool=false)

Plot matrices of pairwise correlation, mutual information and physical distance. Plots arranged in 2 rows each corresponding to an ordering of neurons.
In the top row, neurons are grouped by brain region whereas hierarchical clustering is applied to those in the bottom row.
"""
function neuron_relate1(sort_name::String="NoSorting"; include_border::Bool=false, identify_region::Bool=false)
    plot_holder = Array{Plots.Plot{Plots.GRBackend}}(undef, 3, 3)

    for mouseID in 1:3
        # load data
        regIDs = load_mouse_data(mouseID, "brainLoc", true)
        coords = load_mouse_data(mouseID, "ccfCoords", true)

        # compute "relation" matrices
        corM = load_result(mouseID, "cor")
        logmiM = log.(load_result(mouseID, "minfo"))
        pdistM = pairwise(Euclidean(), coords, dims=1)
        regM = _reg_diff(regIDs, identify_region)

        # plot configuration
        border_pos =  include_border ? _locate_borders(regIDs) : nothing
        # clr = identify_region ? cgrad(:gist_earth, 15, rev=true, categorical=true) : cgrad(:heat, 2, categorical=true)
        # clm = identify_region ? (-1, 14) : (0, 1)

        sort_idx = 1:length(regIDs)
        if  sort_name=="GroupByRegion"
            sort_idx = neuron_sort(regIDs)
        elseif sort_name=="HClustAll" || sort_name=="HClustInRegion"
            D = sqrt.(1 .- corM)            
            
            if sort_name=="HClustInRegion"
                sort_idx = neuron_sort(regIDs, D, :average)
            else
                sort_idx = neuron_sort(D, :average)
            end
        end

        plot_holder[mouseID*3-2] = _heatmap_mod(corM[sort_idx, sort_idx], (-0.3, 1), mouseID, 1,  border_pos)
        plot_holder[mouseID*3-1] = _heatmap_mod(logmiM[sort_idx, sort_idx], (-20, 1.1), mouseID, 2, border_pos)
        plot_holder[mouseID*3]   = _heatmap_mod(pdistM[sort_idx, sort_idx], (0, 6050), mouseID, 3, border_pos)
    end


    plot(plot_holder...,
         layout=(3, 3), size=(1800*3, 1200*3), yflip=true, ticks=false,
         top_margin=10mm, bottom_margin=10mm,
         titlefontsize=45, tickfontsize=35)

    annotate!(-250, 550, text("Krebs", :left, 45), subplot=1)
    annotate!(-650, 1000, text("Waksman", :left, 45), subplot=4)
    annotate!(-550, 1000, text("Robbins", :left, 45), subplot=7)

    savefig("C:/Users/trann/Google Drive/Research/NeuropixelSpont/Plots/$sort_name.png")
end


end
