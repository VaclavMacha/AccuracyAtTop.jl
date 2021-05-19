module AccuracyAtTop

using ChainRulesCore
using LinearAlgebra
using Statistics

using Distributions: Sampleable, Univariate, Continuous, Uniform
using Random: AbstractRNG

export All, Neg, Pos, LogUniform
export Maximum, Minimum, Quantile, Kth, SampledQuantile
export objective, predict, FNRate, FPRate, FNFPRate
export AccAtTop, DeepTopPush, DeepTopPushK, PatMat, PatMatNP
export hinge, quadratic, threshold
export buffer, reset_buffer!, update_buffer!

# custom types
abstract type Objective end
abstract type Threshold end
abstract type Indices end
struct All <: Indices end
struct Pos <: Indices end
struct Neg <: Indices end

Base.show(io::IO, ::Type{All}) = print(io, "all")
Base.show(io::IO, ::Type{Neg}) = print(io, "negative")
Base.show(io::IO, ::Type{Pos}) = print(io, "positive")

include("thresholds.jl")
include("objectives.jl")

# buffer
const LAST_THRESHOLD = Ref{Vector{Float32}}([Inf32])
const LAST_THRESHOLD_IND = Ref{Vector{Int}}([1])

buffer() = LAST_THRESHOLD[], LAST_THRESHOLD_IND[]

function reset_buffer!()
    LAST_THRESHOLD[] = [Inf32]
    LAST_THRESHOLD_IND[] = [1]
    return
end

function update_buffer!(t, ind)
    LAST_THRESHOLD[] = [t...,]
    LAST_THRESHOLD_IND[] = [ind...,]
    return
end
end # module
