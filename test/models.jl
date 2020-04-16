import AccuracyAtTop: Model, Buffer, ThresholdModel, FNRModel, FPRModel
import AccuracyAtTop: Statistics, Flux, loss, loss_gradient
import AccuracyAtTop: find_threshold

import Statistics: mean


T          = Float32
data       = rand(T, 10, 50) 
target     = rand(1, 50) .>= 0.5
classifier = Chain(Dense(10, 1))
scores     = classifier(data)

# -------------------------------------------------------------------------------
# BaseLine model
# -------------------------------------------------------------------------------
@testset "BaseLine model" begin
    model = BaseLine(classifier)

    @test typeof(model) <: BaseLine
    @test BaseLine <: Model
    @test fieldnames(BaseLine) == (:classifier, :T)
    @test model(data) == classifier(data)
end


# -------------------------------------------------------------------------------
# Threshold models
# -------------------------------------------------------------------------------
@testset "Threshold models" begin
    @test ThresholdModel <: Model
    @test FNRModel <: ThresholdModel
    @test FPRModel <: ThresholdModel
end


##############
# FNR models #
##############
# TopPush
@testset "TopPush model" begin
    model     = TopPush(classifier)
    surrogate = model.surrogate

    @test typeof(surrogate) <: Hinge
    @test typeof(model) <: TopPush
    @test TopPush <: FNRModel
    @test fieldnames(TopPush) == (:classifier, :surrogate, :buffer, :T, :threshold)

    ind_neg = findall(vec(target) .== 0)
    ind_pos = findall(vec(target) .== 1)
    t, ind  = findmax(scores[ind_neg])
    t_ind   = ind_neg[ind]

    @test model(data) == scores
    @test threshold(model, target, scores) == t
    @test find_threshold(model, target, scores) == (t, t_ind)
    @test loss(model, data, target) == mean(surrogate.value.(t .- scores[ind_pos]))

    tmp  = @. - surrogate.gradient(t - scores) * (target .== 1)
    ∇L_s = tmp./length(ind_pos)
    ∇L_t = - sum(tmp)./length(ind_pos)
    @test loss_gradient(model, target, scores, threshold(model, target, scores)) == (∇L_s, ∇L_t)
end


# TopPushK
@testset "TopPushK model: K = $K" for K in [2, 4, 5]
    model     = TopPushK(K, classifier)
    surrogate = model.surrogate

    @test typeof(surrogate) <: Hinge
    @test typeof(model) <: TopPushK
    @test TopPushK <: FNRModel
    @test fieldnames(TopPushK) == (:K, :classifier, :surrogate, :buffer, :T, :threshold)

    ind_neg = findall(vec(target) .== 0)
    ind_pos = findall(vec(target) .== 1)
    ind     = partialsortperm(scores[ind_neg], K; rev = true)
    t       = scores[ind_neg][ind]
    t_ind   = ind_neg[ind]

    @test model(data) == scores
    @test threshold(model, target, scores) == t
    @test find_threshold(model, target, scores) == (t, t_ind)
    @test loss(model, data, target) == mean(surrogate.value.(t .- scores[ind_pos]))

    tmp  = @. - surrogate.gradient(t - scores) * (target .== 1)
    ∇L_s = tmp./length(ind_pos)
    ∇L_t = - sum(tmp)./length(ind_pos)
    @test loss_gradient(model, target, scores, threshold(model, target, scores)) == (∇L_s, ∇L_t)
end


# PatMat
@testset "PatMat model: τ = $τ" for τ in [0.1, 0.2, 0.3, 0.5, 0.6]
    model     = PatMat(τ, classifier)
    surrogate = model.surrogate

    @test typeof(surrogate) <: Hinge
    @test typeof(model) <: PatMat
    @test PatMat <: FNRModel
    @test fieldnames(PatMat) == (:τ, :classifier, :surrogate, :buffer, :T, :threshold)

    ind_pos = findall(vec(target) .== 1)
    n       = length(scores)
    k       = min(max(floor(Int64, n*(1-τ)), 1), n)
    t_ind   = partialsortperm(vec(scores), k)
    t       = scores[t_ind]

    @test model(data) == scores
    @test threshold(model, target, scores) == t
    @test find_threshold(model, target, scores) == (t, t_ind)
    @test loss(model, data, target) == mean(surrogate.value.(t .- scores[ind_pos]))

    tmp  = @. - surrogate.gradient(t - scores) * (target .== 1)
    ∇L_s = tmp./length(ind_pos)
    ∇L_t = - sum(tmp)./length(ind_pos)
    @test loss_gradient(model, target, scores, threshold(model, target, scores)) == (∇L_s, ∇L_t)
end


# PatMatNP
@testset "PatMatNP model: τ = $τ" for τ in [0.1, 0.2, 0.3, 0.5, 0.6]
    model     = PatMatNP(τ, classifier)
    surrogate = model.surrogate

    @test typeof(surrogate) <: Hinge
    @test typeof(model) <: PatMatNP
    @test PatMatNP <: FNRModel
    @test fieldnames(PatMatNP) == (:τ, :classifier, :surrogate, :buffer, :T, :threshold)

    ind_neg = findall(vec(target) .== 0)
    ind_pos = findall(vec(target) .== 1)
    n       = length(scores[ind_neg])
    k       = min(max(floor(Int64, n*(1-τ)), 1), n)
    ind     = partialsortperm(vec(scores[ind_neg]), k)
    t       = scores[ind_neg][ind]
    t_ind   = ind_neg[ind]

    @test model(data) == scores
    @test threshold(model, target, scores) == t
    @test find_threshold(model, target, scores) == (t, t_ind)
    @test loss(model, data, target) == mean(surrogate.value.(t .- scores[ind_pos]))

    tmp  = @. - surrogate.gradient(t - scores) * (target .== 1)
    ∇L_s = tmp./length(ind_pos)
    ∇L_t = - sum(tmp)./length(ind_pos)
    @test loss_gradient(model, target, scores, threshold(model, target, scores)) == (∇L_s, ∇L_t)
end


# RecAtK
@testset "RecAtK model: K = $K" for K in [2, 4, 5]
    model     = RecAtK(K, classifier)
    surrogate = model.surrogate

    @test typeof(surrogate) <: Hinge
    @test typeof(model) <: RecAtK
    @test RecAtK <: FNRModel
    @test fieldnames(RecAtK) == (:K, :classifier, :surrogate, :buffer, :T, :threshold)

    ind_pos = findall(vec(target) .== 1)
    t_ind   = partialsortperm(vec(scores), K; rev = true)
    t       = scores[t_ind]

    @test model(data) == scores
    @test threshold(model, target, scores) == t
    @test find_threshold(model, target, scores) == (t, t_ind)
    @test loss(model, data, target) == mean(surrogate.value.(t .- scores[ind_pos]))

    tmp  = @. - surrogate.gradient(t - scores) * (target .== 1)
    ∇L_s = tmp./length(ind_pos)
    ∇L_t = - sum(tmp)./length(ind_pos)
    @test loss_gradient(model, target, scores, threshold(model, target, scores)) == (∇L_s, ∇L_t)
end


##############
# FPR models #
##############
# PrecAtRec
@testset "PrecAtRec model: rec = $rec" for rec in [0.1, 0.2, 0.3, 0.5, 0.6]
    model     = PrecAtRec(rec, classifier)
    surrogate = model.surrogate

    @test typeof(surrogate) <: Hinge
    @test typeof(model) <: PrecAtRec
    @test PrecAtRec <: FPRModel
    @test fieldnames(PrecAtRec) == (:rec, :classifier, :surrogate, :buffer, :T, :threshold)

    ind_neg = findall(vec(target) .== 0)
    ind_pos = findall(vec(target) .== 1)
    n       = length(scores[ind_pos])
    k       = min(max(floor(Int64, n*(1 - rec)), 1), n)
    ind     = partialsortperm(vec(scores[ind_pos]), k)
    t       = scores[ind_pos][ind]
    t_ind   = ind_pos[ind]

    @test model(data) == scores
    @test threshold(model, target, scores) == t
    @test find_threshold(model, target, scores) == (t, t_ind)
    @test loss(model, data, target) == mean(surrogate.value.(scores[ind_neg] .- t))

    tmp  = @. surrogate.gradient(scores - t) * (target .== 0)
    ∇L_s = tmp./length(ind_neg)
    ∇L_t = - sum(tmp)./length(ind_neg)
    @test loss_gradient(model, target, scores, threshold(model, target, scores)) == (∇L_s, ∇L_t)
end

