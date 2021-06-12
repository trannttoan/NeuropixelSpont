# include("dependencies.jl")

function savePACF(uplim=10)
    names = ["Krebs", "Waksman", "Robbins"]
    lags = collect(1:uplim)

    pacors = Dict()

    for i=1:3
        pacors[names[i]] = StatsBase.pacf(transpose(load_mouse_data(i, "stall")), lags);
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
