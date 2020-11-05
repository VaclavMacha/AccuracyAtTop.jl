# auxiliary functions
using Distributions: Sampleable, Univariate, Continuous, Uniform
using Random: AbstractRNG

find_negatives(targets) = findall(vec(targets) .== 0)
find_positives(targets) = findall(vec(targets) .== 1)

@nograd find_negatives, find_positives

function find_kth(x, k::Int; rev::Bool = false)
    ind = partialsortperm(x, k, rev = rev)
    return x[ind], ind
end

function find_kth(x, ks; rev::Bool = false)
    inds = partialsortperm(x, 1:maximum(ks), rev = rev)
    return x[inds[ks]], inds[ks]
end

function find_quantile(x, τ; rev::Bool = false)
    all(0 .<= τ .<= 1) || throw(ArgumentError("input probability out of [0,1] range"))

    n = length(x)
    i = rev ? 1 .- τ : τ
    ks = min.(max.(1, round.(Int64, n.*i)), n)

    if maximum(ks) <= n/2
        return find_kth(x, ks; rev = false)
    else
        return find_kth(x, n .- ks .+ 1; rev = true)
    end
end

# find score
function find_score(::Type{AllSamples}, find::Function, targets, scores, args...; kwargs...)
    return find(view(scores, :), args...; kwargs...)
end

function find_score(::Type{NegSamples}, find::Function, targets, scores, args...; kwargs...)
    inds = find_negatives(targets)
    val, ind = find(view(scores, inds), args...; kwargs...)
    return val, inds[ind]
end

function find_score(::Type{PosSamples}, find::Function, targets, scores, args...; kwargs...)
    inds = find_positives(targets)
    val, ind = find(view(scores, inds), args...; kwargs...)
    return val, inds[ind]
end

# surrogate functions
hinge(x, ϑ::Real = 1) = max(zero(x), 1 + ϑ * x)
quadratic(x, ϑ::Real = 1) = max(zero(x), 1 + ϑ * x)^2

# objectives
function fnr(targets, scores, t, surrogate = quadratic)
    return mean(surrogate.(t .- scores[find_positives(targets)]))
end

function fpr(targets, scores, t, surrogate = quadratic)
    return mean(surrogate.(scores[find_negatives(targets)] .- t))
end

# samplers
struct LogUniform <: Sampleable{Univariate,Continuous}
    d
    LogUniform(a, b) = new(Uniform(log10(a), log10(b)))
end

Base.rand(rng::AbstractRNG, s::LogUniform) = 10 ^ rand(rng, s.d)
