# include("dependencies.jl")

function savePACF(uplim=10)
    names = ["Krebs", "Waksman", "Robbins"]
    spkcounts = load_mouse_data(i, "stall")
    lags = collect(1:uplim)

    pacors = Dict()

    for i=1:3
        pacors[names[i]] = StatsBase.pacf(transpose(spkcounts), lags);
    end

    MAT.matwrite("../Data/save/PACFs.mat", pacors)
end

function clustEval()
    for i=1:3
        spkcounts = load_mouse_data(i, "stall")
        D = Distances.pairwise(Distances.CorrDist(), spkcounts, dims=1)
        regIDs = convert.(Int64, vec(load_mouse_data(i, "brainLoc")))
        regIDs[regIDs .== 0] .= 12
        res = Clustering.silhouettes(regIDs, D)
        display(plot(res))
        println(mean(res))
    end
end


function PACFdist(mouseID, uplim)
    spkcounts = load_mouse_data(mouseID, "stall")
    for i=2:uplim
        display(histogram(vec(StatsBase.pacf(transpose(spkcounts), [i])), nbin=-0.25:0.01:0.5, xlim=(-0.25, 0.5), ylim=(0, 500), legend=false, title="Lag=$i"))
    end

end

function vispop(mouseID)
    root = "C:/Users/trann/Google Drive/Research/NeuropixelSpont"
    spkcounts = convert(Matrix{Int}, load_mouse_data(mouseID, "stall"))
    N, T = size(spkcounts)
    mu = vec(Statistics.mean(spkcounts, dims=1))
    sig = vec(Statistics.std(spkcounts, dims=1; corrected=false))
    p1 = plot(mu)
    p2 = plot(sig)
    plot(p1, p2, layout=(2, 1), size=(round(T/10), 600), legend=false, widen=false)
    savefig(root * "/Plots/pop$mouseID.png")
end

function preproc(mouseID, lim)
    spkcounts = convert(Matrix{Int64}, load_mouse_data(mouseID, "stall"))
    N, T = size(spkcounts)
    tstamp = load_mouse_data(mouseID, "tspont")
    spkcount_rates = sum(spkcounts, dims=2) / (tstamp[end]-tstamp[1])
    return sum(spkcount_rates .< lim) / N
end



function bincol(spkcounts, binsize, slide=true)
    if binsize==1
        return spkcounts
    end

    N, T = size(spkcounts)

    if slide
        nbin = T - (binsize-1)
        binnedspkcounts = hcat([sum(spkcounts[:, i:i+(binsize-1)], dims=2) for i=1:nbin]...)
    else
        rem = T % binsize
        nbin = Int((T-rem) / binsize)
        binnedspkcounts = hcat([sum(spkcounts[:, (i-1)*binsize+1:i*binsize], dims=2) for i=1:nbin]...)
        if binsize > 1
            binnedspkcounts = hcat(binnedspkcounts, sum(spkcounts[:, T-(rem-1):end], dims=2))
        end
    end

    return binnedspkcounts
end

function bintest(binsizes, slide)
    plot_holder = Array{Plots.Plot{Plots.GRBackend}}(undef, 3)
    for mouseID in 1:3
        spkcounts = convert(Matrix{Int}, load_mouse_data(mouseID, "stall"))
        avgcors = [mean(abs.(Statistics.cor(bincol(spkcounts, sz, slide), dims=2))) for sz in binsizes]
        plot_holder[mouseID] = plot(binsizes, avgcors)
    end

    plot(plot_holder..., layout=(3, 1), size=(1200, 600), legend=false)
end

function benchmark(n, loop=true)
    if loop
        a = Array{Int}(undef, n, n)
        for i=1:n
            a[:, i] = ones(n)
        end
    else
        a = hcat([ones(n) for i=1:n]...)
    end

    # return a
end

function raster_plot(mouseID, sortby)
    root = "C:/Users/trann/Google Drive/Research/NeuropixelSpont/"
    spkcounts = convert(Matrix{Int}, load_mouse_data(mouseID, "stall"))
    N, T = size(spkcounts)
    if sortby=="var"
        sortidx = sortperm(vec(var(spkcounts, dims=2; corrected=false)))
    elseif sortby=="pc1"
        infile = matopen(root * "Data/pc1.mat")
        sortidx = sortperm(vec(read(infile, "pc1_$mouseID")))
        close(infile)
    elseif sortby=="emb"
        infile = matopen(root * "Data/emb.mat")
        sortidx = sortperm(vec(read(infile, "emb$mouseID")))
        close(infile)
    end

    heatmap(spkcounts[sortidx, :], size=(round(T/10), N/2), clim=(0, 1), color=cgrad(:grays, rev=true), colorbar=false)
    savefig(root * "Plots/raster$(mouseID)_$sortby.png") 
end

function avgcorT(mouseID)
    names = ["Krebs", "Waksman", "Robbins"]
    root = "C:/Users/trann/Google Drive/Research/NeuropixelSpont/"
    spkcounts = convert(Matrix{Int}, load_mouse_data(mouseID, "stall"))
    N, T = size(spkcounts)
    binsize = 36
    nbins = T - (binsize-1)
    avgcor = Vector{Float64}(undef, nbins)

    for i in 1:nbins
        corM = Statistics.cor(spkcounts[:, i:i+(binsize-1)], dims=2)
        corM[isnan.(corM)] .= 0
        avgcor[i] = mean(abs.(corM))
    end

    plot(avgcor, size=(round(T/10), N), title=names[mouseID], legend=false)
    savefig(root * "Plots/tempcor$mouseID.png")
end

function avgcorVS(mouseID)
    root = "C:/Users/trann/Google Drive/Research/NeuropixelSpont/"
    spkcounts = convert(Matrix{Float64}, load_mouse_data(mouseID, "stall"))
    N, T = size(spkcounts)
    
    cors = Vector{Float64}(undef, N)
    cors[1] = mean(abs.(Statistics.cor(spkcounts[1, :], spkcounts[2:end, :], dims=2)))
    cors[end] = mean(abs.(Statistics.cor(spkcounts[end, :], spkcounts[1:end-1, :], dims=2)))
    cors[2:end-1] = [mean(abs.(Statistics.cor(spkcounts[i, :], vcat(spkcounts[1:i-1, :], spkcounts[i+1:end, :]), dims=2))) for i in 2:N-1]
    dt = StatsBase.fit(ZScoreTransform, spkcounts, dims=2)

    zsc = StatsBase.transform(dt, spkcounts)
    stds = Statistics.std(zsc, dims=2)

    scatter(stds, cors, legend=false)
end



function plot_neurons(mouseID, threshold)
    spkcounts = load_mouse_data(mouseID, "stall")
    avgfluct = mean(abs.(spkcounts[:, 2:end] - spkcounts[:, 1:end-1]), dims=2)
    highids = [i[1] for i in findall(avgfluct .> threshold)]
    lowids = [i for i in setdiff(Set(collect(1:size(spkcounts, 1))), Set(highids))]
    plot_holder = Array{Plots.Plot{Plots.GRBackend}}(undef, 10)

    for i in 1:5
        hid = highids[rand(1:length(highids))]
        plot_holder[i] = plot(spkcounts[hid, :], title="Neuron $hid: $(avgfluct[hid])")
        lid = lowids[rand(1:length(lowids))]
        plot_holder[i+5] = plot(spkcounts[lid, :], title="Neuron $lid: $(avgfluct[lid])")
    end
    plot(plot_holder..., layout=(10, 1), size=(900, 1200), legend=false)
    savefig("C:/Users/trann/Google Drive/Research/NeuropixelSpont/Plots/temp.png")
end

function fluctVSvar()
    plot_holder = Array{Plots.Plot{Plots.GRBackend}}(undef, 3)
    
    for i in 1:3
        spkcounts = load_mouse_data(i, "stall")
        avgfluct = mean(abs.(spkcounts[:, 2:end] - spkcounts[:, 1:end-1]), dims=2)
        stdv = std(spkcounts, dims=2)
        plot_holder[i] = scatter(stdv, avgfluct)
    end
    plot(plot_holder..., guidefontsize=10, layout=(1, 3), size=(900, 400), grid=false, legend=false, left_margin=2mm, bottom_margin=5mm)
end

function lowfrate(mouseID, threshold, below=true)
    spkcounts = load_mouse_data(mouseID, "stall")
    period = vec(load_mouse_data(mouseID, "tspont"))
    vid = load_mouse_data(mouseID, "video")
    frates = vec(sum(spkcounts, dims=2)/(period[end]-period[1]))

    if below
        ids = findall(frates .< threshold)
    else
        ids = findall(frates .> threshold)
    end

    plot_holder = Array{Plots.Plot{Plots.GRBackend}}(undef, 11)

    for i in 1:10
        id = ids[rand(1:length(ids))]
        plot_holder[i] = plot(period .- period[1], spkcounts[id, :], title="Neuron $id: $(frates[id])")
    end

    plot_holder[11] = plot(period .- period[1], vid)
    println(length(ids)/size(spkcounts, 1))

    plot(plot_holder..., layout=(11, 1), size=(1800, 1200), legend=false, widen=false)
    savefig("C:/Users/trann/Google Drive/Research/NeuropixelSpont/Plots/temp.png")
end

function saveClustPerm()
    names = ["Krebs", "Waksman", "Robbins"]
    perms = Dict()

    for i in 1:3
        spkcounts = load_mouse_data(i, "stall")
        corrM = Statistics.cor(spkcounts, dims=2)
        dissM = sqrt.(1 .- corrM)
        perms[names[i]] = Clustering.hclust(dissM, linkage=:average)
    end
    
    MAT.matwrite("C:/Users/trann/Google Drive/Research/NeuropixelSpont/Data/save/clustperms.mat", perms)
end

function clustPACF()
    names = ["Krebs", "Waksman", "Robbins"]
    root = "C:/Users/trann/Google Drive/Research/NeuropixelSpont/Data"

    infile = matopen(root * "/save/PACFs.mat")
    plot_holder = Array{Plots.Plot{Plots.GRBackend}}(undef, 6)

    for i in 1:3
        pacor = read(infile, names[i])
        println(size(pacor))
        # corM = Statistics.cor(pacor, dims=2)
        # disM = sqrt.(1 .- corM)
        disM = Distances.pairwise(Distances.CosineDist(), pacor, dims=2)
        clustres = Clustering.hclust(disM, linkage=:average)
        plot_holder[i] = heatmap(disM, yflip=true)
        plot_holder[i+3] = heatmap(disM[clustres.order, clustres.order], yflip=true)
    end
    
    plot(plot_holder..., layout=(2, 3), size=(1700*3, 1200*2))
    savefig("C:/Users/trann/Google Drive/Research/NeuropixelSpont/Plots/pacfclust.png")

    close(infile)
end

function plot_neurons(mouseID, neuronIDs, reduce)
    spkcounts = load_mouse_data(mouseID, "stall", reduce)
    nneurons = length(neuronIDs)
    plot_holder = Array{Plots.Plot{Plots.GRBackend}}(undef, nneurons)
    

    for i in 1:nneurons
        plot_holder[i] = plot(spkcounts[neuronIDs[i], :])
    end

    plot(plot_holder..., layout=(nneurons, 1), size=(2000, 1500), legend=false)
    savefig("C:/Users/trann/Google Drive/Research/NeuropixelSpont/Plots/neurons.png")
end

function correl_dist()
    names = ["Krebs", "Waksman", "Robbins"]
    plot_holder = Array{Plots.Plot{Plots.GRBackend}}(undef, 3)

    for i in 1:3
        spkcounts = load_mouse_data(i, "stall", true)
        corM = vec(Statistics.cor(spkcounts, dims=2) - I)
        plot_holder[i] = histogram(corM)
    end

    plot(plot_holder..., layout=(1, 3), size=(500*3, 450))
    savefig("C:/Users/trann/Google Drive/Research/NeuropixelSpont/Plots/cordist.png")
end

function percentsc(threshold)
    names = ["Krebs", "Waksman", "Robbins"]
    for i in 1:3
        spkcounts = load_mouse_data(i, "stall", true)
        flatsc = vec(spkcounts)
        println("M$i: $(sum(flatsc .< threshold)/length(flatsc))")
    end
end

function max_dist()
    plot_holder = plot_holder = Array{Plots.Plot{Plots.GRBackend}}(undef, 3)

    for i in 1:3
        spkcounts = load_mouse_data(i, "stall")
        maxes = maximum(spkcounts, dims=2)
        plot_holder[i] = histogram(maxes)
    end

    plot(plot_holder..., layout=(1, 3), size=(500*3, 450))
end

function frateVSstdv()
    plot_holder = plot_holder = Array{Plots.Plot{Plots.GRBackend}}(undef, 3)

    for i in 1:3
        spkcounts = load_mouse_data(i, "stall")
        period = load_mouse_data(i ,"tspont")

        plot_holder[i] = scatter(Statistics.std(spkcounts, dims=2), sum(spkcounts, dims=2)/(period[end]-period[1]),
                              xlabel="stdv", ylabel="spike-count rate")
    end

    plot(plot_holder..., layout=(1, 3), size=(500*3, 450), legend=false)
end

function inact_percent()
    plot_holder = plot_holder = Array{Plots.Plot{Plots.GRBackend}}(undef, 3)

    for i in 1:3
        spkcounts = load_mouse_data(i, "stall")
        N, T = size(spkcounts)
        zeropercent = sum(spkcounts .== 0, dims=2) ./ T 
        println(minimum(zeropercent))
        plot_holder[i] = histogram(zeropercent)
    end

    plot(plot_holder..., layout=(1, 3), size=(500*3, 450), legend=false)
end

function arorder_dist(maxlag=50)
    plot_holder = plot_holder = Array{Plots.Plot{Plots.GRBackend}}(undef, 3)

    for i in 1:3
        spkcounts = load_mouse_data(i, "stall", true)
        N, T = size(spkcounts)
        
        lags = collect(1:maxlag)
        ci = 1.96/sqrt(T)

        orders = zeros(N)

        for j in 1:N
            pacor = StatsBase.pacf(spkcounts[j, :], lags)
            temp = findfirst(abs.(pacor) .< ci)
            orders[j] = temp === nothing ? maxlag : temp-1
        end

        plot_holder[i] = histogram(orders, nbin=0:2:maxlag, xlabel="Lag")
    end

    plot(plot_holder..., layout=(1, 3), size=(500*3, 450), legend=false)

end

function neuron_pacf(mouseID, threshold)
    spkcounts = load_mouse_data(mouseID, "stall")

    pacors = vec(StatsBase.pacf(transpose(spkcounts), [1]))
    output = findall(abs.(pacors) .> threshold)

    return output
end