module AccuracyAtTop

using LinearAlgebra, Statistics, Flux

using Flux.Optimise: Params, StopException
using Flux.Optimise: @progress
using Flux.Optimise: gradient, update!, runall
using Zygote: @adjoint, @nograd

export AllSamples, NegSamples, PosSamples, Buffer
export Maximum, Quantile, Kth, SampledQuantile
export PRate, NRate, TPRate, TNRate, FPRate, FNRate
export SampledPRate, SampledNRate, SampledTPRate, SampledTNRate, SampledFPRate, SampledFNRate
export LogUniform
export fnr, fpr, hinge, quadratic, threshold, BatchProvider

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

Buffer() = Buffer(Inf, 1)

const BUFFER = Ref{Buffer}(Buffer())

function reset_buffer!(b::Buffer = Buffer())
    BUFFER[] = b
    return
end

function update_buffer!(t::Real, ind)
    BUFFER[].t = t
    BUFFER[].ind = ind
    return
end

update_buffer!(t, ind) = nothing

struct BatchProvider{I<:Integer}
    loader
    neg::Vector{I}
    pos::Vector{I}
    n_neg::I
    n_pos::I
    buffer::Bool
    batch::Vector{I}

    function BatchProvider(
            loader,
            labels,
            batchsize;
            ratio = 0.5,
            buffer::Bool = false,
            batch = rand(1:length(labels), batchsize),
        )

        n_neg = round(Int, batchsize * ratio)
        n_pos = batchsize - n_neg

        return new{Int}(
            loader,
            findall(labels .== false),
            findall(labels .== true),
            n_pos,
            n_neg,
            buffer,
            batch,
        )
    end
end

function Base.show(io::IO, b::BatchProvider)
    n_neg = length(b.neg)
    n_pos = length(b.pos)
    n = n_neg + n_pos

    println(io, "BatchProvider:")
    println(io, " - dataset (n_neg/n_pos/n): $(n_neg)/$(n_pos)/$(n_neg + n_pos)")
    println(io, " - batch (k_neg/k_pos/k): $(b.n_neg)/$(b.n_pos)/$(b.n_neg + b.n_pos)")
    print(io, " - buffer: $(b.buffer)")
    return
end

function (b::BatchProvider)(buffer = BUFFER[])
    inds = vcat(rand(b.neg, b.n_neg), rand(b.pos, b.n_pos))
    if b.buffer
        inds[rand(1:(b.n_neg + b.n_pos))] = b.batch[buffer.ind]
        b.batch .= inds
    end
    return b.loader(inds)
end

end # module
