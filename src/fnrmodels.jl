# -------------------------------------------------------------------------------
# Models minimizing false negative rate
# -------------------------------------------------------------------------------
# TopPush
mutable struct TopPush{B} <: FNRModel{B}
    surrogate 
    classifier
    threshold
    buffer::B

    function TopPush(classifier; T::Type = Float32, surrogate = Hinge(), buffer = NoBuffer())
        new{typeof(buffer)}(surrogate, deepcopy(classifier), zero(T), deepcopy(buffer))
    end
end


function show(io::IO, model::TopPush)
    print(io, "TopPush($(model.surrogate), $(model.buffer))")
end


function threshold(model::TopPush, s, y)
    maximum(s[vec(y) .== 0])
end


function threshold_gradient(model::TopPush, s, y)
    ind_neg = findall(vec(y) .== 0)
    t, ind  = findmax(s[ind_neg])
    t_ind   = ind_neg[ind]

    model.threshold = t

    return t, t_ind
end


# TopPushK
mutable struct TopPushK{B} <: FNRModel{B}
    K
    surrogate 
    classifier
    threshold
    buffer::B

    function TopPushK(K, classifier; T::Type = Float32, surrogate = Hinge(), buffer = NoBuffer())
        new{typeof(buffer)}(K, surrogate, deepcopy(classifier), zero(T), deepcopy(buffer))
    end
end


function show(io::IO, model::TopPushK)
    print(io, "TopPushK($(model.K), $(model.surrogate), $(model.buffer))")
end


function threshold(model::TopPushK, s, y)
    findkth(s[vec(y) .== 0], model.K; rev = true)[1]
end


function threshold_gradient(model::TopPushK, s, y)
    ind_neg = findall(vec(y) .== 0)
    t, ind  = findkth(s[ind_neg], model.K; rev = true)
    t_ind   = ind_neg[ind]

    model.threshold = t

    return t, t_ind
end


# PatMat
mutable struct PatMat{B} <: FNRModel{B}
    τ
    surrogate 
    classifier
    threshold
    buffer::B

    function PatMat(τ, classifier; T::Type = Float32, surrogate = Hinge(), buffer = NoBuffer())
        new{typeof(buffer)}(τ, surrogate, deepcopy(classifier), zero(T), deepcopy(buffer))
    end
end


function show(io::IO, model::PatMat)
    print(io, "PatMat($(model.τ), $(model.surrogate), $(model.buffer))")
end


function threshold(model::PatMat, s, y)
    findquantile(vec(s), 1 - model.τ)[1]
end


function threshold_gradient(model::PatMat, s, y)
    t, t_ind = findquantile(vec(s), 1 - model.τ)

    model.threshold = t

    return t, t_ind
end


# PatMatNP
mutable struct PatMatNP{B} <: FNRModel{B}
    τ
    surrogate 
    classifier
    threshold
    buffer::B

    function PatMatNP(τ, classifier; T::Type = Float32, surrogate = Hinge(), buffer = NoBuffer())
        new{typeof(buffer)}(τ, surrogate, deepcopy(classifier), zero(T), deepcopy(buffer))
    end
end


function show(io::IO, model::PatMatNP)
    print(io, "PatMatNP($(model.τ), $(model.surrogate), $(model.buffer))")
end


function threshold(model::PatMatNP, s, y)
    findquantile(s[vec(y) .== 0], 1 - model.τ)[1]
end


function threshold_gradient(model::PatMatNP, s, y)
    ind_neg = findall(vec(y) .== 0)
    t, ind  = findquantile(s[ind_neg], 1 - model.τ)
    t_ind   = ind_neg[ind]

    model.threshold = t

    return t, t_ind
end