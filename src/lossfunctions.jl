# -------------------------------------------------------------------------------
# Surrogate functions
# -------------------------------------------------------------------------------
@with_kw_noshow struct Hinge
    value    = (x) -> max(zero(x), 1 + x)
    gradient = (x) -> 1 + x >= 0 ? one(x) : zero(x)
end


function show(io::IO, surrogate::Hinge)
    print(io, "Hinge")
end


@with_kw_noshow struct Quadratic
    value    = (x) -> max(zero(x), 1 + x)^2
    gradient = (x) -> (val = 1 + x; val >= 0 ? 2*val : zero(x))
end


function show(io::IO, surrogate::Quadratic)
    print(io, "Quadratic")
end


# -------------------------------------------------------------------------------
# Loss functions
# -------------------------------------------------------------------------------
abstract type Loss; end


# -------------------------------------------------------------------------------
# Simple model
# -------------------------------------------------------------------------------
abstract type SimpleLoss <: Loss; end


function loss(model::SimpleModel, x, y)
    return value(model.loss, scores(model, x), y)
end


########################
# Binary cross eptropy #
########################
struct BinCrossEntropy <: SimpleLoss; end


function show(io::IO, surrogate::BinCrossEntropy)
    print(io, "BinCrossEntropy")
end


function value(L::BinCrossEntropy, s, y)
    return mean(binarycrossentropy.(sigmoid.(s), y))
end


# -------------------------------------------------------------------------------
# Threshold models
# -------------------------------------------------------------------------------
abstract type ThresholdLoss <: Loss; end


function loss(model::ThresholdModel, x, y)
    s = scores(model, x)
    t = threshold(model, s, y)
    return loss(model, s, t, y)
end


function loss(model::ThresholdModel, s, t, y)
    return value(model.loss, s, t, y)
end


function gradient_loss(model::ThresholdModel, s, t, y)
    return gradient(model.loss, s, t, y)
end


#######################
# False negative rate #
#######################
@with_kw_noshow struct FNR <: ThresholdLoss
    surrogate = Hinge()
end


function show(io::IO, loss::FNR)
    print(io, "FNR($(loss.surrogate))")
end


function value(L::FNR, s, t, y)
    return mean(L.surrogate.value.(t .- s[y .== 1]))
end


function gradient(L::FNR, s, t, y)
    ∇L_s   = @. - L.surrogate.gradient(t - s) * (y == 1)
    n_pos = sum(y) 
    return ∇L_s./n_pos, -sum(∇L_s)/n_pos
end


#######################
# False positive rate #
#######################
@with_kw_noshow struct FPR <: ThresholdLoss
    surrogate = Hinge()
end


function show(io::IO, loss::FPR)
    print(io, "FPR($(loss.surrogate))")
end


function value(L::FPR, s, t, y)
    return mean(L.surrogate.value.(s[y .== 0] .- t))
end


function gradient(L::FPR, s, t, y)
    ∇L_s   = @. - L.surrogate.gradient(s - t) * (y == 0)
    n_neg = sum(y) 
    return ∇L_s./n_neg, -sum(∇L_s)/n_neg
end