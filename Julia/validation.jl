module Validation

include("dataIO.jl")
using .DataIO: load_mouse_data


function clustValid(data, id, metric, link, byprobe=false)
    st_all = data[id]["stall"]
    probe_ids = data[id]["iprobe"]
    plot_array = Array{Plots.Plot{Plots.GRBackend}}(undef, 8*2)

    if byprobe
        for i=1:8
            st_iprobe = st_all[vec(probe_ids .== i), :]
            n_neuron_iprobe = size(st_iprobe, 1)
            shuffle_idx = shuffle(1:n_neuron_iprobe)
            # initial
            cor_mat1 = Statistics.cor(st_iprobe, dims=2)
            diss_mat1 = Distances.pairwise(metric, cor_mat1, dims=1)
            result1 = Clustering.hclust(diss_mat1, linkage=link)
            cor_init = cor_mat1[result1.order, result1.order]
            # shuffled
            cor_mat2 = Statistics.cor(st_iprobe[shuffle_idx, :], dims=2)
            diss_mat2 = Distances.pairwise(metric, cor_mat2, dims=1)
            result2 = Clustering.hclust(diss_mat2, link)
            cor_shuffle = cor_mat2[result2.order, result2.order]

            #println("M$id-P$i: $(mean(abs.(cor_init - cor_shuffle)))")

            plot_array[i*2-1] = heatmap(cor_init, title="M$id-P$i(initial)")
            plot_array[i*2] = heatmap(cor_shuffle, title="M$id-P$i(shuffled)")
        end

        display(plot(plot_array[1:8]..., layout=(4, 2), size=(1250, 1800), clim=(-0.3, 1), yflip=true, left_margin=8mm))
        display(plot(plot_array[9:16]..., layout=(4, 2), size=(1250, 1800), clim=(-0.3, 1), yflip=true, left_margin=8mm))
    else
    end
end


function coordValid(data, id)
    probeIDs = data[id]["iprobe"]       # probe IDs of neurons
    hop = data[id]["Wh"]                # neuron heights on probes
    coord = data[id]["ccfCoords"]       # neuron 3D coordinates

    plot_holder = Array{Plots.Plot{Plots.GRBackend}}(undef, 8*2)

    for i=1:8
        extract_idx = vec(probeIDs .== i)
        D_hop = diff(hop[extract_idx])
        D_coord = Distances.pairwise(Distances.Euclidean(), coord[extract_idx, :], dims=1)

        plot_holder[i*2-1] = heatmap(D_hop, title="M$id-P$i-hop")
        plot_holder[i*2] = heatmap(D_coord, title="M$id-P$i-coord")
        #println(mean(abs.(D_hop - D_coord)))
    end

    display(plot(plot_holder[1:8]..., layout=(4, 2), size=(1200, 1800), yflip=true))
    display(plot(plot_holder[9:16]..., layout=(4, 2), size=(1200, 1800), yflip=true))
end



end
