module AccuracyAtTop

using LinearAlgebra, Statistics, Flux

using Flux.Optimise: Params, StopException
using Flux.Optimise: @progress
using Flux.Optimise: gradient, update!, runall
using Zygote: @adjoint, @nograd

export AllSamples, NegSamples, PosSamples, Buffer
export Maximum, Quantile, Kth, TPRate, TNRate, FPRate, FNRate
export fnr, fpr, hinge, quadratic, threshold

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
end

Buffer() = Buffer(Inf, 0)

const BUFFER = Ref{Buffer}(Buffer())

function reset_buffer!(b::Buffer)
    BUFFER[] = b
    return
end

function update_buffer!(t::Real, ind)
    BUFFER[].t = t
    BUFFER[].ind = ind
    return
end

end # module
