# -------------------------------------------------------------------------------
# Cuda
# -------------------------------------------------------------------------------
function allow_cuda(flag::Bool)
    AccuracyAtTop.Flux.use_cuda[] = flag
    status_cuda()
    return
end


function status_cuda()
    if AccuracyAtTop.Flux.use_cuda[]
        @info "Cuda is allowed"
    else
        @info "Cuda is not allowed"
    end
    return
end


function cpu(model_in::Model)
    model = deepcopy(model_in)
    model.classifier = cpu(model.classifier)
    return model
end


function gpu(model_in::Model)
    model = deepcopy(model_in)
    model.classifier = gpu(model.classifier)
    return model
end


# -------------------------------------------------------------------------------
# auxiliary model functions
# -------------------------------------------------------------------------------
params(model::Model)    = params(model.classifier)
save_model(name, model) = BSON.bson(name, model = cpu(model))
load_model(name)        = BSON.load(name)[:model]


# -------------------------------------------------------------------------------
# train model
# -------------------------------------------------------------------------------
function train!(model::Model, batches, optimiser; cb = () -> ())
    pars    = params(model)
    cb      = runall(cb)
    indexes = randperm(length(batches))

    for ind in indexes
        try
            gs = gradient(model, pars, batches, ind)
            @timeit "update parameters" update!(optimiser, pars, gs)
            cb()
        catch ex
            if ex isa StopException
                break
            else
                rethrow(ex)
            end
        end
    end
end


macro runepochs(n, ex)
    quote 
        @showprogress for i = 1:$(esc(n))
            @timeit "epoch" $(esc(ex))
        end
    end
end


macro tracktime(expr)
    quote
        reset_timer!()
        $(esc(expr))
        print_timer()
        println()
    end
end


# -------------------------------------------------------------------------------
# auxiliary threshold functions
# -------------------------------------------------------------------------------
getdim(A::AbstractArray, d::Integer, i) =
    getindex(A, Base._setindex(i, d, axes(A)...)...)


clip(x, xmin, xmax) =
    min(max(xmin, x), xmax)


isneg(target) = target == 0
ispos(target) = target == 1


find_negatives(target) = findall(isneg.(vec(target)))
find_positives(target) = findall(ispos.(vec(target)))


function scores_max(scores, inds = LinearIndices(scores))
    val, ind = findmax(view(scores, inds))

    return val, inds[ind]
end

function scores_kth(scores::AbstractArray{T, 2}, k::Int, inds = LinearIndices(scores); kwargs...) where T
    size(scores, 1) == 1 || throw(ArgumentError("scores must be row or column vector"))
    return scores_kth(vec(scores), k, vec(inds); kwargs...)
end


function scores_kth(scores::AbstractVector, k::Int, inds = LinearIndices(scores); rev::Bool = false)
    vals = view(scores, inds)
    n    = length(vals)
    1 <= k <= n || throw(ArgumentError("input index out of {1,$n} set"))

    ind = partialsortperm(vals, k, rev = rev)

    return vals[ind], inds[ind]
end


function scores_quantile(scores, p::Real, inds = LinearIndices(scores))
    0 <= p <= 1 || throw(ArgumentError("input probability out of [0,1] range"))

    n = min(length(scores), length(inds))
    k = clip(floor(Int64, n*p), 1, n)

    if k <= n/2
        return scores_kth(scores, k, inds; rev = false)
    else
        return scores_kth(scores, n - k + 1, inds; rev = true)
    end
end


# -------------------------------------------------------------------------------
# Surrogate functions
# -------------------------------------------------------------------------------
@with_kw_noshow struct Hinge <: Surrogate
    ϑ        = 1
    value    = (x) -> max(zero(x), 1 + ϑ*x)
    gradient = (x) -> 1 + ϑ*x >= 0 ? ϑ*one(x) : zero(x)
end

Hinge(ϑ::Real) = Hinge(ϑ = ϑ)


show(io::IO, surrogate::Hinge) =
    print(io, "Hinge($(surrogate.ϑ))")


@with_kw_noshow struct Quadratic <: Surrogate
    ϑ        = 1
    value    = (x) -> max(zero(x), 1 + ϑ*x)^2
    gradient = (x) -> (val = 1 + ϑ*x; val >= 0 ? 2*ϑ*val : zero(x))
end

Quadratic(ϑ::Real) = Quadratic(ϑ = ϑ)


show(io::IO, surrogate::Quadratic) =
    print(io, "Quadratic($(surrogate.ϑ))")
