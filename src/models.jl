abstract type Model; end
abstract type Buffer; end
abstract type Surrogate; end

(model::Model)(data::AbstractArray) = model.classifier(data)


# -------------------------------------------------------------------------------
# BaseLine model
# -------------------------------------------------------------------------------
abstract type BaseLineModel <: Model end

@with_kw_noshow mutable struct BaseLine <: BaseLineModel
    classifier
    objective::Function = binarycrossentropy
    T::Type = Float32

    BaseLine(cls, obj, T) = new(deepcopy(cls), obj, T)
end


BaseLine(classifier; kwargs...) =
    BaseLine(classifier = classifier; kwargs...)


show(io::IO, model::BaseLine) =
    print(io, "BaseLine($(model.objective))")


loss(model::BaseLine, data, target) =
    mean(model.objective.(sigmoid.(model(data)), target))


@with_kw_noshow mutable struct BalancedBaseLine <: BaseLineModel
    classifier
    objective::Function = binarycrossentropy
    T::Type = Float32

    BalancedBaseLine(cls, obj, T) = new(deepcopy(cls), obj, T)
end


BalancedBaseLine(classifier; kwargs...) =
    BalancedBaseLine(classifier = classifier; kwargs...)


show(io::IO, model::BalancedBaseLine) =
    print(io, "BalancedBaseLine($(model.objective))")


function loss(model::BalancedBaseLine, data, target, w = weights(target))
    sum(model.objective.(sigmoid.(model(data)), target) .* w)
end


function weights(target)
    y = cpu(target)
    w = y ./ sum(y .== 1) .+ (1 .- y) ./ sum(y .== 0) 
    return gpu(w)
end


# -------------------------------------------------------------------------------
# Threshold models
# -------------------------------------------------------------------------------
abstract type ThresholdModel{Buffer} <: Model end


threshold(model::ThresholdModel, target, scores) =
    find_threshold(model, target, scores)[1]


function loss(model::ThresholdModel, data, target)
    scores = model(data)
    t      = threshold(model, target, scores)
    return loss(model, target, scores, t)
end


##############
# FNR models #
##############
abstract type FNRModel{Buffer} <: ThresholdModel{Buffer} end


loss(model::FNRModel, target, scores, t) =
    mean(model.surrogate.value.(t .- scores[ispos.(target)]))


function loss_gradient(model::FNRModel, target, scores, t)
    ∇L_s   = @. - model.surrogate.gradient(t - scores) * ispos.(target)
    n_pos = sum(target) 
    return ∇L_s./n_pos, -sum(∇L_s)/n_pos
end


# TopPush
@with_kw_noshow mutable struct TopPush{B<:Buffer} <: FNRModel{B}
    classifier
    surrogate::Surrogate = Hinge()
    buffer::B = NoBuffer()

    T::Type   = Float32
    threshold = zero(T)

    TopPush{B}(cls, surr, buff::B, T, t) where {B} =
        new(deepcopy(cls), surr, buff, T, t)
end

TopPush(classifier; kwargs...) =
    TopPush(classifier = classifier; kwargs...)


show(io::IO, model::TopPush) =
    print(io, "TopPush($(model.surrogate), $(model.buffer))")


find_threshold(model::TopPush, target, scores) =
    scores_max(scores, find_negatives(target))


# TopPushK
@with_kw_noshow mutable struct TopPushK{B<:Buffer} <: FNRModel{B}
    K::Int
    classifier
    surrogate::Surrogate = Hinge()
    buffer::B            = NoBuffer()

    T::Type   = Float32
    threshold = zero(T)

    TopPushK{B}(K, cls, surr, buff::B, T, t) where {B} =
        new(K, deepcopy(cls), surr, buff, T, t)
end


TopPushK(K, classifier; kwargs...) =
    TopPushK(K = K, classifier = classifier; kwargs...)


show(io::IO, model::TopPushK) =
    print(io, "TopPushK($(model.K), $(model.surrogate), $(model.buffer))")


find_threshold(model::TopPushK, target, scores) =
    scores_kth(scores, model.K, find_negatives(target); rev = true)


# PatMat
@with_kw_noshow mutable struct PatMat{B<:Buffer} <: FNRModel{B}
    τ::Real
    classifier
    surrogate::Surrogate = Hinge()
    buffer::B            = NoBuffer()

    T::Type   = Float32
    threshold = zero(T)

    PatMat{B}(τ, cls, surr, buff::B, T, t) where {B} =
        new(τ, deepcopy(cls), surr, buff, T, t)
end


PatMat(τ, classifier; kwargs...) =
    PatMat(τ = τ, classifier = classifier; kwargs...)


show(io::IO, model::PatMat) = 
    print(io, "PatMat($(model.τ), $(model.surrogate), $(model.buffer))")


find_threshold(model::PatMat, target, scores) = 
    scores_quantile(scores, 1 - model.τ)


# PatMatNP
@with_kw_noshow mutable struct PatMatNP{B<:Buffer} <: FNRModel{B}
    τ
    classifier
    surrogate::Surrogate = Hinge()
    buffer::B            = NoBuffer()

    T::Type   = Float32
    threshold = zero(T)

    PatMatNP{B}(τ, cls, surr, buff::B, T, t) where {B} =
        new(τ, deepcopy(cls), surr, buff, T, t)
end


PatMatNP(τ, classifier; kwargs...) =
    PatMatNP(τ = τ, classifier = classifier; kwargs...)


show(io::IO, model::PatMatNP) =
    print(io, "PatMatNP($(model.τ), $(model.surrogate), $(model.buffer))")


find_threshold(model::PatMatNP, target, scores) =
    scores_quantile(scores, 1 - model.τ, find_negatives(target))


# RecAtK
@with_kw_noshow mutable struct RecAtK{B<:Buffer} <: FNRModel{B}
    K::Int
    classifier
    surrogate::Surrogate = Hinge()
    buffer::B            = NoBuffer()

    T::Type   = Float32
    threshold = zero(T)

    RecAtK{B}(K, cls, surr, buff::B, T, t) where {B} =
        new(K, deepcopy(cls), surr, buff, T, t)
end


RecAtK(K, classifier; kwargs...) =
    RecAtK(K = K, classifier = classifier; kwargs...)


show(io::IO, model::RecAtK) =
    print(io, "RecAtK($(model.K), $(model.surrogate), $(model.buffer))")


find_threshold(model::RecAtK, target, scores) =
    scores_kth(scores, model.K; rev = true)


##############
# FPR models #
##############
abstract type FPRModel{Buffer} <: ThresholdModel{Buffer} end


loss(model::FPRModel, target, scores, t) =
    return mean(model.surrogate.value.(scores[isneg.(target)] .- t))


function loss_gradient(model::FPRModel, target, scores, t)
    ∇L_s  = @. model.surrogate.gradient(scores - t) * isneg.(target)
    n_neg = length(target) - sum(target) 
    return ∇L_s./n_neg, -sum(∇L_s)/n_neg
end


# PrecAtRec
@with_kw_noshow mutable struct PrecAtRec{B<:Buffer} <: FPRModel{B}
    rec::Real
    classifier
    surrogate::Surrogate = Hinge()
    buffer::B            = NoBuffer()

    T::Type   = Float32
    threshold = zero(T)

    PrecAtRec{B}(rec, cls, surr, buff::B, T, t) where {B} =
        new(rec, deepcopy(cls), surr, buff, T, t)
end


PrecAtRec(rec, classifier; kwargs...) =
    PrecAtRec(rec = rec, classifier = classifier; kwargs...)


show(io::IO, model::PrecAtRec) =
    print(io, "PrecAtRec($(model.rec), $(model.surrogate), $(model.buffer))")


find_threshold(model::PrecAtRec, target, scores) =
    scores_quantile(scores, 1 - model.rec, find_positives(target))