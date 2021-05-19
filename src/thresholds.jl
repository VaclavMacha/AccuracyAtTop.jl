function select(::Type{Neg}, y, s)
    inds = findall(y .== 0)
    return view(s, inds), inds
end

function select(::Type{Pos}, y, s)
    inds = findall(y .== 1)
    return view(s, inds), inds
end

# Thresholds
function threshold(
    tp::Threshold,
    y::AbstractMatrix,
    s::AbstractMatrix;
    update_buffer = true
)
    # TODO buffer
    return find_threshold(tp, y, s)[1]
end

function ChainRulesCore.rrule(
    ::typeof(threshold),
    tp::Threshold,
    y,
    s;
    update_buffer = true,
)

    ts, inds = find_threshold(tp, y, s)
    if update_buffer

    end
    # TODO buffer

    function threshold_pullback(Δ)
        Δt = zero(s)
        Δt[CartesianIndex.(1:length(inds), inds)] .= 1
        return NO_FIELDS, DoesNotExist(), DoesNotExist(), Δ .* Δt
    end
    return ts, threshold_pullback
end

function find_threshold(tp::Threshold, y::AbstractMatrix, s::AbstractMatrix)
    k = size(s, 1)
    ts, inds = similar(s, k), zeros(Int, k)

    @inbounds for i in 1:k
        ts[i], inds[i] = find_threshold(tp, Vector(y[i, :]), Vector(s[i, :]))
    end
    return ts, inds
end

function find_threshold(
    I::Type{<:Indices},
    find_func,
    y_in::AbstractVector,
    s_in::AbstractVector,
    args...,
)
    s, inds = select(I, y_in, s_in)
    t, ind = find_func(s, args...)
    return t, inds[ind]
end

function find_threshold(
    ::Type{All},
    find_func,
    ::AbstractVector,
    s_in::AbstractVector,
    args...,
)

    return find_func(s_in, args...)
end

function find_kth(x, k::Int, rev::Bool)
    ind = partialsortperm(x, k; rev)
    return x[ind], ind
end

function find_kth(x, ks, rev::Bool)
    inds = partialsortperm(x, 1:maximum(ks); rev)
    return x[inds[ks]], inds[ks]
end

function find_quantile(x, τ, rev::Bool)
    if !all(0 .<= τ .<= 1)
        throw(ArgumentError("input probability out of [0,1] range"))
    end

    n = length(x)
    i = rev ? 1 .- τ : τ
    ks = min.(max.(1, round.(Int64, n .* i)), n)

    if maximum(ks) <= n/2
        return find_kth(x, ks, false)
    else
        return find_kth(x, n .- ks .+ 1, true)
    end
end

# Maximum and minimum scores
struct Maximum{I<:Indices} <: Threshold
    Maximum(indices = Neg) = new{indices}()
end

function Base.show(io::IO, ::Maximum{I}) where {I}
    from = I <: All ? "" : "$(I) "
    return print(io, "largest $(from)score")
end


function find_threshold(::Maximum{I}, y::AbstractVector, s::AbstractVector) where {I}
    return find_threshold(I, findmax, y, s)
end

struct Minimum{I<:Indices} <: Threshold
    Minimum(indices = Pos) = new{indices}()
end

function Base.show(io::IO, ::Minimum{I}) where {I}
    from = I <: All ? "" : "$(I) "
    return print(io, "smallest $(from)score")
end

function find_threshold(::Minimum{I}, y::AbstractVector, s::AbstractVector) where {I}
    return find_threshold(I, findmin, y, s)
end

# K-th laregst/smallest scores
struct Kth{I<:Indices} <: Threshold
    K::Int64
    rev::Bool

    Kth(K, indices = Neg; rev = true) = new{indices}(K, rev)
end

function ordinal(i::Int)
    j = i % 10
    k = i % 100
    if j == 1 && k != 11
        return "$(i)st"
    elseif j == 2 && k != 12
        return  "$(i)nd"
    elseif j == 3 && k != 13
        return  "$(i)rd"
    else
        return "$(i)th"
    end
end

function Base.show(io::IO, tp::Kth{I}) where {I}
    ord = tp.rev ? "largest" : "smallest"
    from = I <: All ? "" : "$(I) "
    return print(io, "$(ordinal(tp.K)) $(ord) $(from)score")
end

function find_threshold(tp::Kth{I}, y::AbstractVector, s::AbstractVector) where {I}
    return find_threshold(I, find_kth, y, s, tp.K, tp.rev)
end

# Quantiles
struct Quantile{I<:Indices} <: Threshold
    τ::Float64
    rev::Bool

    Quantile(τ, indices = Neg; rev = true) = new{indices}(τ, rev)
end

function Base.show(io::IO, tp::Quantile{I}) where {I}
    ord = tp.rev ? "top" : "bottom"
    print(io, "$(ord) $(tp.τ)-quantile from $(I) scores")
end

function find_threshold(tp::Quantile{I}, y::AbstractVector, s::AbstractVector) where {I}
    return find_threshold(I, find_quantile, y, s, tp.τ, tp.rev)
end

struct SampledQuantile{I<:Indices} <: Threshold
    sampler
    rev::Bool

    SampledQuantile(sampler, indices = Neg; rev = true) = new{indices}(sampler, rev)
end

function Base.show(io::IO, tp::SampledQuantile{I}) where {I}
    ord = tp.rev ? "top" : "bottom"
    print(io, "$(ord) sampled quantile from $(I) scores")
end

function find_threshold(tp::SampledQuantile{I}, y::AbstractVector, s::AbstractVector) where {I}
    return find_threshold(I, find_quantile, y, s, tp.sampler(), tp.rev)
end

# samplers
struct LogUniform <: Sampleable{Univariate,Continuous}
    d

    LogUniform(a, b) = new(Uniform(log10(a), log10(b)))
end

Base.rand(rng::AbstractRNG, s::LogUniform) = 10 ^ rand(rng, s.d)
