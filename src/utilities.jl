getdim(A::AbstractArray, d::Integer, i) = getindex(A, Base._setindex(i, d, axes(A)...)...)

clip(x, xmin, xmax) = min(max(xmin, x), xmax)

isneg(target) = target == 0
ispos(target) = target == 1

find_negatives(target) = findall(isneg.(vec(target)))
find_positives(target) = findall(ispos.(vec(target)))

function weights(target)
    n_pos = sum(ispos.(target))
    n_neg = length(target) - n_pos
    return target ./ n_pos .+ (1 .- target) ./ n_neg
end

@nograd find_negatives, find_positives, weights


function scores_max(scores, inds = LinearIndices(scores))
    val, ind = findmax(view(scores, inds))

    return val, inds[ind]
end


function scores_kth(scores::AbstractArray{T, 2}, k::Int, inds = LinearIndices(scores); kwargs...) where T
    size(scores, 1) == 1 || throw(ArgumentError("scores must be row or column vector"))
    return scores_kth(vec(scores), k, vec(inds); kwargs...)
end


function scores_kth(scores::AbstractVector, k::Int, inds = LinearIndices(scores); rev::Bool = false)
    vals = view(scores, inds)
    n    = length(vals)
    1 <= k <= n || throw(ArgumentError("input index out of {1,$n} set"))

    ind = partialsortperm(vals, k, rev = rev)

    return vals[ind], inds[ind]
end


function scores_quantile(scores, p::Real, inds = LinearIndices(scores))
    0 <= p <= 1 || throw(ArgumentError("input probability out of [0,1] range"))

    n = min(length(scores), length(inds))
    k = clip(floor(Int64, n*p), 1, n)

    if k <= n/2
        return scores_kth(scores, k, inds; rev = false)
    else
        return scores_kth(scores, n - k + 1, inds; rev = true)
    end
end


# -------------------------------------------------------------------------------
# Surrogate functions
# -------------------------------------------------------------------------------
hinge(x) = max(zero(x), 1 + x)
quadratic(x) = max(zero(x), 1 + x)^2