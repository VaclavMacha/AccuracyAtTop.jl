module AccuracyAtTop

using LinearAlgebra, Statistics, Flux

using Flux.Optimise: Params, StopException
using Flux.Optimise: @progress
using Flux.Optimise: batchmemaybe, gradient, update!, runall
using Zygote: @adjoint, @nograd

export AllSamples, NegSamples, PosSamples, Buffer
export Maximum, Quantile, Kth, TPRate, TNRate, FPRate, FNRate
export fnr, fpr, hinge, quadratic, threshold, train_with_buffer!

# custom types
abstract type AbstractThreshold end
abstract type SampleIndices end
abstract type AllSamples <: SampleIndices end
abstract type PosSamples <: SampleIndices end
abstract type NegSamples <: SampleIndices end

include("thresholds.jl")
include("utilities.jl")

# buffer
mutable struct Buffer
    t::Float64
    ind::Int64
    active::Bool
end

Buffer(; active = false) = Buffer(Inf, -1, active)

const BUFFER = Ref{Buffer}(Buffer())

function update_buffer!(b::Buffer)
    BUFFER[] = b
    return
end

function update_buffer!(t::Real, ind)
    BUFFER[].t = t
    BUFFER[].ind = ind
    return
end

function update_buffer!(inds)
    BUFFER[].ind = inds[BUFFER[].ind]
    return
end

function update_inds(inds::AbstractArray{<:Int})
    if BUFFER[].ind > 0 && BUFFER[].active
        return cat(inds, BUFFER[].ind; dims = ndims(inds))
    else
        return inds
    end
end

# train! function
function train_with_buffer!(
    loss,
    ps,
    loader::Function,
    data_inds,
    opt;
    cb = () -> (),
    buffer::Buffer = Buffer()
)

    ps = Params(ps)
    cb = runall(cb)
    update_buffer!(buffer)

    @progress for inds in data_inds
        inds2 = update_inds(inds)
        d = loader(inds2)
        try
            gs = gradient(ps) do
                loss(batchmemaybe(d)...)
            end
            update!(opt, ps, gs)
            update_buffer!(inds2)
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
