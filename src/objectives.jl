# Surrogate functions
hinge(x, ϑ::Real = 1) = max(zero(x), 1 + ϑ * x)
quadratic(x, ϑ::Real = 1) = max(zero(x), 1 + ϑ * x)^2

# Objective functions
struct FNRate <: Objective end

Base.show(io::IO, ::FNRate) = print(io, "false-negative rate")

function objective(::FNRate, y, s::AbstractArray{T}, t, surrogate) where {T<:Real}
    inds = zero(T) .+ (y .== 1)
    return sum(surrogate.(t .-  s) .* inds; dims = 2) ./ sum(inds; dims = 2)
end

struct FPRate <: Objective end

Base.show(io::IO, ::FPRate) = print(io, "false-positive rate")

function objective(::FPRate, y, s::AbstractArray{T}, t, surrogate) where {T<:Real}
    inds = zero(T) .+ (y .== 1)
    return sum(surrogate.(s .-  t) .* inds; dims = 2) ./ sum(inds; dims = 2)
end

struct FNFPRate <: Objective
    α::Real
end

function Base.show(io::IO, o::FNFPRate)
    return print(io, "$(o.α)⋅false-negative + $(1-o.α)⋅false-positive rate")
end

function objective(o::FNFPRate, y, s, t, surrogate)
    return o.α .* objective(FNRate(), y, s, t, surrogate) .+
           (1 - o.α) .* objective(FPRate(), y, s, t, surrogate)
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
    agg = mean,
    update_buffer = true,
)

    if size(y) != size(s)
        throw(DimensionMismatch("dimensions must match: y has dims $(size(y)), s has dims $(size(s))"))
    end

    ts = threshold(m.threshold_type, y, s; update_buffer)
    return agg(objective(m.objective_type, y, s, ts, surrogate))
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

    if length(ts) != size(s, 1)
        throw(DimensionMismatch("length of thresholds $(length(ts)) does not match first dim of scores $(size(s, 1))"))
    end
    return s .>= ts
end

# Basic models
DeepTopPush() = AccAtTop(Maximum(Neg), FNRate())
DeepTopPushK(K) = AccAtTop(Kth(K, Neg), FNRate())
PatMat(τ::Real) = AccAtTop(Quantile(τ, All; rev = true), FNRate())
PatMat(sampler) = AccAtTop(SampledQuantile(sampler, All; rev = true), FNRate())
PatMatNP(τ::Real) = AccAtTop(Quantile(τ, Neg; rev = true), FNRate())
PatMatNP(sampler) = AccAtTop(SampledQuantile(sampler, Neg; rev = true), FNRate())
