module Analysis

using InformationMeasures: get_mutual_information, get_entropy
using Clustering: hclust


function minfo_compute(sc::Array{Int, 2})
    n = size(sc, 1)
    mi_mat = zeros(n, n)

    # upper
    for i=1:n-1
        for j=i+1:n
            mi_mat[i, j] = get_mutual_information(sc[i, :], sc[j, :])
        end
    end

    # lower
    mi_mat = mi_mat + transpose(mi_mat)

    # diagonal
    for i=1:n
        mi_mat[i, i] = get_entropy(sc[i, :])
    end

    return mi_mat
end

function jentropy_compute(sc)
    n = size(sc, 1)
    je_mat = zeros(n, n)

    # upper
    for i=1:n-1
        for j=i+1:n
            je_mat[i, j] = get_entropy(sc[i, :], sc[j, :])
        end
    end

    # lower
    je_mat = je_mat + transpose(je_mat)

    # diagonal
    for i=1:n
        je_mat[i, i] = get_entropy(sc[i, :], sc[i, :])
    end

    return je_mat
end


function neuron_sort(regIDs::Array{Float64, 1}, shuffleWithinRegion::Bool=false)
    regSort = sortperm(regIDs)

    if shuffleWithinRegion
        regIDs = regIDs[regSort]
        Nreg = 14

        for i=1:Nreg
            filter = regIDs .== i
            if sum(filter != 0)
                regSort[filter] = shuffle(regSort[filter])
            end
        end
    end

    return regSort
end

function neuron_sort(D::Array{Float64, 2}, link::Symbol)
    result = hclust(D; linkage=link)

    return result.order
end

function neuron_sort(regIDs::Array{Float64, 1}, D::Array{Float64, 2}, link::Symbol)
    regSort = sortperm(regIDs)
    D = D[regSort, regSort]
    regIDs = regIDs[regSort]
    Nreg = 14

    for i=1:Nreg
        filter = regIDs .== i
        if sum(filter) != 0
            res = hclust(D[filter, filter], linkage=link)
            regSort[filter] = regSort[filter][res.order]
        end
    end

    return regSort
end


end
