module AccuracyAtTop

using LinearAlgebra, Statistics, Flux

using Zygote: @adjoint, @nograd
using Flux: sigmoid, binarycrossentropy
using Flux.Optimise: Params, runall, @progress, gradient, update!, StopException, batchmemaybe


include("models.jl")
include("utilities.jl")

export loss_baseline, loss_fnr, loss_fpr, DeepTopPush, DeepTopPushK,
       PatMat, PatMatNP, RecAtK, PrecAtRec, hinge, quadratic

const THRESHOLD = Ref{Float64}(Inf)
const THRESHOLD_IND = Ref{Int}(-1)

lastthreshold() = THRESHOLD[]
lastthreshold_ind() = THRESHOLD_IND[]

function set_lastthreshold!(t::Real, ind::Int)
    THRESHOLD[] = t
    THRESHOLD_IND[] = ind
    return
end
update_ind!(inds) = THRESHOLD_IND[] = inds[THRESHOLD_IND[]]
add_inds(inds, ind::Int) = ind > 0 ? vcat(inds, ind) : inds

# train! function
function train_buffer!(loss, ps, loader::Function, data_inds, opt; cb = () -> (), buffer::Bool = true)
    ps = Params(ps)
    cb = runall(cb)

    @progress for inds in data_inds
        if buffer
            inds = add_inds(inds, lastthreshold_ind())
        end
        d = loader(inds)
        try
            gs = gradient(ps) do
                loss(batchmemaybe(d)...)
            end
            update!(opt, ps, gs)
            buffer && update_ind!(inds)
            cb()
        catch ex
            if ex isa StopException
                break
            else
                rethrow(ex)
            end
        end
    end
end

end # module
