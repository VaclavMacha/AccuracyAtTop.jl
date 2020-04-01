# -------------------------------------------------------------------------------
# Models minimizing false negative rate
# -------------------------------------------------------------------------------
# PrecAtRec
mutable struct PrecAtRec{B} <: FPRModel{B}
    rec
    surrogate 
    classifier
    threshold
    buffer::B

    function PrecAtRec(rec, classifier; T::Type = Float32, surrogate = Hinge(), buffer = NoBuffer())
        new{typeof(buffer)}(rec, surrogate, deepcopy(classifier), zero(T), deepcopy(buffer))
    end
end


function show(io::IO, model::PrecAtRec)
    print(io, "PrecAtRec($(model.rec), $(model.surrogate), $(model.buffer))")
end


function threshold(model::PrecAtRec, s, y)
    findquantile(s[vec(y) .== 1], 1 - model.rec)[1]
end


function threshold_gradient(model::PrecAtRec, s, y)
    ind_pos = findall(vec(y) .== 1)
    t, ind  = findquantile(s[ind_pos], 1 - model.rec)
    t_ind   = ind_pos[ind]

    model.threshold = t

    return t, t_ind
end
