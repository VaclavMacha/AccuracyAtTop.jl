# -------------------------------------------------------------------------------
# Scores
# -------------------------------------------------------------------------------
function scores(model::Model, x)
    return model.classifier(x)
end


# -------------------------------------------------------------------------------
# Surrogate functions
# -------------------------------------------------------------------------------
@with_kw_noshow struct Hinge
    ϑ        = 1
    value    = (x) -> max(zero(x), 1 + ϑ*x)
    gradient = (x) -> 1 + ϑ*x >= 0 ? ϑ*one(x) : zero(x)
end


show(io::IO, surrogate::Hinge) = print(io, "Hinge($(surrogate.ϑ))")


@with_kw_noshow struct Quadratic
    ϑ        = 1
    value    = (x) -> max(zero(x), 1 + ϑ*x)^2
    gradient = (x) -> (val = 1 + ϑ*x; val >= 0 ? 2*ϑ*val : zero(x))
end


show(io::IO, surrogate::Quadratic) = print(io, "Quadratic($(surrogate.ϑ))")


# -------------------------------------------------------------------------------
# Simple model
# -------------------------------------------------------------------------------
function loss(model::BaseLineModel, x, y)
    s = scores(model, x)
    return mean(binarycrossentropy.(sigmoid.(s), y))
end


# -------------------------------------------------------------------------------
# Threshold models
# -------------------------------------------------------------------------------
function loss(model::ThresholdModel, x, y)
    s = scores(model, x)
    t = threshold(model, s, y)
    return loss(model, s, t, y)
end


# False negative rate
function loss(model::FNRModel, s, t, y)
    return mean(model.surrogate.value.(t .- s[y .== 1]))
end


function loss_gradient(model::FNRModel, s, t, y)
    ∇L_s   = @. - model.surrogate.gradient(t - s) * (y == 1)
    n_pos = sum(y) 
    return ∇L_s./n_pos, -sum(∇L_s)/n_pos
end


# False positive rate
function loss(model::FPRModel, s, t, y)
    return mean(model.surrogate.value.(s[y .== 0] .- t))
end


function loss_gradient(model::FPRModel, s, t, y)
    ∇L_s   = @. model.surrogate.gradient(s - t) * (y == 0)
    n_neg = sum(y) 
    return ∇L_s./n_neg, -sum(∇L_s)/n_neg
end