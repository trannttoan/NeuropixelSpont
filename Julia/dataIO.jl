module DataIO

using MAT: matread, matopen, names, read, write, close
using Statistics: cor

include("analysis.jl")
import .Analysis: minfo_compute, jentropy_compute


function load_mouse_data(mouseID::Int, dtname::String="all")
    mice_names = ["Krebs", "Waksman", "Robbins"]
    spkpath = "../Data/source/$(mice_names[mouseID])withFaces_KS2.mat"
    vidpath = "../Data/save/frameDiff_$(mice_names[mouseID]).mat"

    if dtname == "all"
        output = matread(spkpath)
    elseif dtname == "video"
        fin = matopen(vidpath)
        output = vec(read(fin, "frameDiffInterp"))
        close(fin)
    else
        fin = matopen(spkpath)
        if dtname in names(fin)
            output = read(fin, dtname)
            if dtname == "brainLoc" || dtname == "iprobe" || dtname == "Wh" || dtname == "areaLabels"
                output = vec(output)
            end
        else
            println("Variable does not exist.")
            output = nothing
        end
        close(fin)
    end

    return output
end

### Write "relation" matrix to .mat file
function save_result(dtname::String)
    mice_names = ["Krebs", "Waksman", "Robbins"]

    for i=1:3
        spktrain = load_mouse_data(i, "stall")

        if dtname == "cor"
            dt = cor(spktrain, dims=2)
        elseif dtname == "minfo"
            dt = minfo_compute(spktrain)
        elseif dtname == "jentropy"
            dt = jentropy_compute(spktrain)
        end

        outfile = matopen("../Data/save/M$(i)_$(dtname).mat", "w")
        write(outfile, dtname, dt)
        close(outfile)
    end
end


function load_result(mouseID::Int, dtname::String)
    mice_names = ["Krebs", "Waksman", "Robbins"]

    infile = matopen("../Data/save/M$(mouseID)_$(dtname).mat")
    dt = read(infile, dtname)
    close(infile)

    return dt
end


end
