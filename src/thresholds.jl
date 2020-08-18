function threshold(thres::AbstractThreshold, targets, scores)
    return find_threshold(thres, targets, scores)[1]
end

@adjoint function threshold(thres::AbstractThreshold, targets, scores)
    t, ind = find_threshold(thres, targets, scores)
    Δt_s = zero(scores)
    Δt_s[ind] = 1

    update_buffer!(t, ind)
    return t, Δ -> (nothing, nothing, Δ .* Δt_s)
end

# thresholds
struct Maximum{I<:SampleIndices} <: AbstractThreshold
    Maximum(; samples = NegSamples) = new{samples}()
end

function find_threshold(::Maximum{T}, targets, scores) where T
    return find_score(T, findmax, targets, scores)
end

struct Minimum{I<:SampleIndices} <: AbstractThreshold
    Minimum(; samples = PosSamples) = new{samples}()
end

function find_threshold(::Minimum{T}, targets, scores) where T
    return find_score(T, findmin, targets, scores)
end


struct Quantile{I<:SampleIndices, T<:Real} <: AbstractThreshold
    τ::T
    rev::Bool

    function Quantile(τ; samples = NegSamples, rev = true)
        return new{samples, typeof(τ)}(τ, rev)
    end
end

function find_threshold(t::Quantile{I, T}, targets, scores) where {I, T}
    return find_score(I, find_quantile, targets, scores, t.τ; rev = t.rev)
end

struct Kth{I<:SampleIndices, T<:Integer} <: AbstractThreshold
    k::T
    rev::Bool

    function Kth(k; samples = NegSamples, rev = true)
        return new{samples, typeof(k)}(k, rev)
    end
end

function find_threshold(t::Kth{T, I}, targets, scores) where {I, T}
    return find_score(I, find_kth, targets, scores, t.k; rev = t.rev)
end

# basic options
TPRate(τ) = Quantile(τ; samples = PosSamples, rev = true)
TNRate(τ) = Quantile(τ; samples = NegSamples, rev = false)
FPRate(τ) = Quantile(τ; samples = NegSamples, rev = true)
FNRate(τ) = Quantile(τ; samples = PosSamples, rev = false)
