# auxiliary functions
find_negatives(targets) = findall(vec(targets) .== 0)
find_positives(targets) = findall(vec(targets) .== 1)

@nograd find_negatives, find_positives

function find_kth(x, k::Int; rev::Bool = false)
    ind = partialsortperm(x, k, rev = rev)
    return x[ind], ind
end

function find_quantile(x, τ::Real; rev::Bool = false)
    0 <= τ <= 1 || throw(ArgumentError("input probability out of [0,1] range"))

    n = length(x)
    i = rev ? 1 - τ : τ
    k = min(max(1, round(Int64, n*i)), n)

    if k <= n/2
        return find_kth(x, k; rev = false)
    else
        return find_kth(x, n - k + 1; rev = true)
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
function fnr(target, scores, t, surrogate = quadratic)
    return mean(surrogate.(t .- scores[find_positives(target)]))
end

function fpr(target, scores, t, surrogate = quadratic)
    return mean(surrogate.(scores[find_negatives(target)] .- t))
end
