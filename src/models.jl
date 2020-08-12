abstract type AbstractThreshold end

threshold(tp::AbstractThreshold, target, scores) =
    find_threshold(tp, target, scores)[1]


@adjoint function threshold(tp::AbstractThreshold, target, scores)
    t, t_ind = find_threshold(tp, target, scores)
    Δt_s = zero(scores)
    Δt_s[t_ind] = 1

    set_lastthreshold!(t, t_ind)

    return t, Δ -> (nothing, nothing, Δ .* Δt_s)
end


# -------------------------------------------------------------------------------
# BaseLine models
# -------------------------------------------------------------------------------
function loss_baseline(target, scores; objective = binarycrossentropy)
    w = weights(target)
    return sum(objective.(sigmoid.(scores), target) .* w)
end

# -------------------------------------------------------------------------------
# FNR models
# -------------------------------------------------------------------------------
function loss_fnr(tp::AbstractThreshold, target, scores; surrogate = quadratic)
    ind = find_positives(target)
    return mean(surrogate.(threshold(tp, target, scores) .- scores[ind]))
end


# DeepTopPush
struct DeepTopPush <: AbstractThreshold end


find_threshold(tp::DeepTopPush, target, scores) =
    scores_max(scores, find_negatives(target))


# DeepTopPushK
struct DeepTopPushK <: AbstractThreshold
    K
end


find_threshold(tp::DeepTopPushK, target, scores) =
    scores_kth(scores, tp.K, find_negatives(target); rev = true)


# PatMat
struct PatMat <: AbstractThreshold
    τ
end

find_threshold(tp::PatMat, target, scores) =
    scores_quantile(scores, 1 - tp.τ)


# PatMatNP
struct PatMatNP <: AbstractThreshold
    τ
end

find_threshold(tp::PatMatNP, target, scores) =
    scores_quantile(scores, 1 - tp.τ, find_negatives(target))


# RecAtK
struct RecAtK <: AbstractThreshold
    K
end

find_threshold(tp::RecAtK, target, scores) =
    scores_kth(scores, tp.K; rev = true)


# -------------------------------------------------------------------------------
# FPR models
# -------------------------------------------------------------------------------
function loss_fpr(tp::AbstractThreshold, target, scores; surrogate = quadratic)
    ind = find_negatives(target)
    return mean(surrogate.(scores[ind] .- threshold(tp, target, scores)))
end


# DeepTopPush
struct PrecAtRec <: AbstractThreshold
    K
end

find_threshold(tp::PrecAtRec, target, scores) =
    scores_quantile(scores, 1 - tp.rec, find_positives(target))
