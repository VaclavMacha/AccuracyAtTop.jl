abstract type Model; end
abstract type Buffer; end
abstract type Surrogate; end

(model::Model)(data::AbstractArray) = model.classifier(data)


# -------------------------------------------------------------------------------
# BaseLine model
# -------------------------------------------------------------------------------
@with_kw_noshow mutable struct BaseLine <: Model
    classifier
    T::Type = Float32
end

BaseLine(classifier; kwargs...) =
    BaseLine(classifier = deepcopy(classifier); kwargs...)


show(io::IO, model::BaseLine) =
    print(io, "BaseLine")


loss(model::BaseLine, data, target) =
    mean(binarycrossentropy.(sigmoid.(model(data)), target))


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
    buffer::B            = NoBuffer()

    T::Type   = Float32
    threshold = zero(T)
end


TopPush(classifier; kwargs...) =
    TopPush(classifier = deepcopy(classifier); kwargs...)


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
end


TopPushK(K, classifier; kwargs...) =
    TopPushK(K = K, classifier = deepcopy(classifier); kwargs...)


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
end


PatMat(τ, classifier; kwargs...) =
    PatMat(τ = τ, classifier = deepcopy(classifier); kwargs...)


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
end


PatMatNP(τ, classifier; kwargs...) =
    PatMatNP(τ = τ, classifier = deepcopy(classifier); kwargs...)


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
end


RecAtK(K, classifier; kwargs...) =
    RecAtK(K = K, classifier = deepcopy(classifier); kwargs...)


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
end


PrecAtRec(rec, classifier; kwargs...) =
    PrecAtRec(rec = rec, classifier = deepcopy(classifier); kwargs...)


show(io::IO, model::PrecAtRec) =
    print(io, "PrecAtRec($(model.rec), $(model.surrogate), $(model.buffer))")


find_threshold(model::PrecAtRec, target, scores) =
    scores_quantile(scores, 1 - model.rec, find_positives(target))