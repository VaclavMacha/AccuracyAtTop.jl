# Surrogate functions
hinge(x, ϑ::Real = 1) = max(zero(x), 1 + ϑ * x)
quadratic(x, ϑ::Real = 1) = max(zero(x), 1 + ϑ * x)^2

# Objective functions
struct FNRate <: Objective end
Base.show(io::IO, ::FNRate) = print(io, "false-negative rate")
function objective(::FNRate, y, s, t::Real, surrogate)
    inds = findall(y .== 1)
    if isempty(inds)
        @warn "no positive samples"
        return zero(t)
    else
        return mean(surrogate.(t .-  s[inds]))
    end
end

struct FPRate <: Objective end
Base.show(io::IO, ::FPRate) = print(io, "false-positive rate")
function objective(::FPRate, y, s, t::Real, surrogate)
    inds = findall(y .== 0)
    if isempty(inds)
        @warn "no negative samples"
        return zero(t)
    else
        return mean(surrogate.(s[inds] .- t))
    end
end

struct FNFPRate <: Objective
    α::Real
end

function Base.show(io::IO, o::FNFPRate)
    return print(io, "$(o.α)⋅false-negative + $(1-o.α)⋅false-positive rate")
end

function objective(o::FNFPRate, y, s, t::Real, surrogate)
    return o.α*objective(FNRate(), y, s, t, surrogate) +
           (1 - o.α)*objective(FPRate(), y, s, t, surrogate)
end

# Accuracy at Top formulation
struct AccAtTop
    threshold_type::Threshold
    objective_type::Objective
end

function Base.show(io::IO, m::AccAtTop)
    println(io, "Accuracy at the top:")
    println(io, " - threshold: $(m.threshold_type)")
    print(io, " - objective function: $(m.objective_type)")
    return
end

function objective(
    m::AccAtTop,
    y::AbstractMatrix,
    s::AbstractMatrix;
    surrogate = hinge,
    weights = 1,
    update_buffer = true,
)

    if size(y) != size(s)
        throw(DimensionMismatch("dimensions must match: y has dims $(size(y)), s has dims $(size(s))"))
    end

    k = size(s, 1)
    if length(weights) == 1
        ws = fill(weights, k)
    elseif length(weights) == k
        ws = weights
    else
        throw(DimensionMismatch("length of weights $(length(weights)) does not match first dim of scores $(k)"))
    end

    l = zero(eltype(s))
    ts = threshold(m.threshold_type, y, s; update_buffer)
    @inbounds for i in 1:k
        l += ws[i] * objective(m.objective_type, y[i, :], s[i, :], ts[i], surrogate)
    end
    return l
end

function predict(
    m::AccAtTop,
    y::AbstractMatrix,
    s::AbstractMatrix;
    ts = threshold(m.threshold_type, y, s; update_buffer = false),
)

    if size(y) != size(s)
        throw(DimensionMismatch("dimensions must match: y has dims $(size(y)), s has dims $(size(s))"))
    end

    k = size(s, 1)
    if length(ts) != 1
        throw(DimensionMismatch("length of thresholds $(length(ts)) does not match first dim of scores $(k)"))
    end

    y_predict = similar(y)
    @inbounds for i in 1:k
        y_predict[i, :] = s[i, :] .>= ts[i]
    end
    return y_predict
end

# Basic models
DeepTopPush() = AccAtTop(Maximum(Neg), FNRate())
DeepTopPushK(K) = AccAtTop(Kth(K, Neg), FNRate())
PatMat(τ::Real) = AccAtTop(Quantile(τ, All; rev = true), FNRate())
PatMat(sampler) = AccAtTop(SampledQuantile(sampler, All; rev = true), FNRate())
PatMatNP(τ::Real) = AccAtTop(Quantile(τ, Neg; rev = true), FNRate())
PatMatNP(sampler) = AccAtTop(SampledQuantile(sampler, Neg; rev = true), FNRate())
