# -------------------------------------------------------------------------------
# Simple model
# -------------------------------------------------------------------------------
function gradient(model::SimpleModel, pars, batch_ind::Int, batches)
    @timeit "extract batch" x, y = batches[batch_ind]
    @timeit "gradient" gs = gradient(pars) do
        loss(model, x, y) 
    end
    return gs
end


mutable struct Simple <: SimpleModel
    loss
    classifier

    Simple(loss, classifier) = new(loss, deepcopy(classifier))
end


function show(io::IO, model::Simple)
    print(io, "Simple($(model.loss))")
end


# -------------------------------------------------------------------------------
# Threshold models
# -------------------------------------------------------------------------------
function extract_batch(model::ThresholdModel, current_batch, batches)
    @timeit "extract x" x = batches[current_batch][1]
    @timeit "extract y" y = batches[current_batch][2]
    @timeit "compute scores" s = scores(model, x)

    @timeit "copy to cpu" begin 
        y = cpu(y)
        s = cpu(s)
    end

    @timeit "update buffer" update!(model.buffer, s, y, current_batch)
    return x, y, s
end


function update_batch(model::ThresholdModel, x, y, s, t, current_batch, t_ind, batches)
    batch_ind  = model.buffer.batch_inds[t_ind]
    sample_ind = model.buffer.sample_inds[t_ind]

    # select the sample that represents the threshold
    x_t = selectdim(batches[batch_ind][1], ndims(x), sample_ind)
    y_t = model.buffer.labels[t_ind]
    s_t = model.buffer.scores[t_ind]

    # add samples 
    if current_batch != batch_ind
        x = cat(x, x_t, dims = ndims(x))
        y = cat(y, y_t; dims = ndims(y))
        s = cat(s, s_t; dims = ndims(s))
        sample_ind = length(s)
    end

    # compute gradient of threshold with respect to scores
    ∇t_s             = zero(s)
    ∇t_s[sample_ind] = 1

    return x, y, s, ∇t_s
end


function gradient(model::ThresholdModel, pars, current_batch::Int, batches)
    @timeit "extract batch" begin 
        x, y, s = extract_batch(model, current_batch, batches)
    end
    @timeit "gradient threshold" begin 
        t, t_ind = gradient_threshold(model, model.buffer.scores, model.buffer.labels)
    end
    @timeit "update batch" begin
        x, y, s, ∇t_s = update_batch(model, x, y, s, t, current_batch, t_ind, batches)
    end

    @timeit "gradient loss" begin
        ∇l_s, ∇l_t = gradient_loss(model, s, t, y)
    end

    @timeit "copy to gpu" begin
        T    = eltype(x)
        ∇t_s = gpu(∇t_s)
        ∇l_s = gpu(T.(∇l_s))
        ∇l_t = T.(∇l_t)
    end

    @timeit "gradient classifier" gs = gradient(pars) do
        sum(hook(∇ ->(∇l_s .+ ∇l_t .* ∇t_s) .* ∇, model.classifier(x)))
    end
    return gs
end


###########
# TopPush #
###########
@with_kw_noshow mutable struct TopPush <: ThresholdModel
    loss = FNR(Hinge())
    classifier
    buffer
end


function TopPush(classifier, batch_size, buffer_size; T::Type = Float32)
    buffer = ScoreBuffer(batch_size, buffer_size; T = T)
    return TopPush(classifier = deepcopy(classifier), buffer = buffer)
end


function show(io::IO, model::TopPush)
    print(io, "TopPush($(model.loss),$(model.buffer))")
end



function threshold(model::TopPush, s, y)
    maximum(s[y .== 0])
end

function gradient_threshold(model::TopPush, s, y)
    ind_neg = findall(y .== 0)
    t, ind  = findmax(s[ind_neg])
    t_ind   = ind_neg[ind]

    return t, t_ind
end


############
# TopPushK #
############
@with_kw_noshow mutable struct TopPush <: ThresholdModel
    loss = FNR(Hinge())
    classifier
    buffer
end
mutable struct TopPushK <: ThresholdModel
    K
    loss
    classifier
    buffer
    
    function TopPushK(K, loss, classifier, batch_size, buffer_size; T::Type = Float32)
        buffer = ScoreBuffer(batch_size, buffer_size; T = T)
        return new(K, loss, deepcopy(classifier), buffer)
    end
end


function show(io::IO, model::TopPushK)
    print(io, "TopPushK($(model.K),$(model.loss),$(model.buffer))")
end


function threshold(model::TopPushK, s, y)
    return partialsort(s[y .== 0], model.K, rev = true)
end


function gradient_threshold(model::TopPushK, s, y)
    ind_neg = findall(y .== 0)
    ind     = partialsortperm(s[ind_neg], model.K, rev = true)
    t_ind   = ind_neg[ind]
    t       = s[t_ind]

    return t, t_ind
end


##########
# PatMat #
##########
mutable struct PatMat <: ThresholdModel
    τ
    loss
    classifier
    buffer
    
    function PatMat(τ, loss, classifier, batch_size, buffer_size; T::Type = Float32)
        buffer = ScoreBuffer(batch_size, buffer_size; T = T)
        return new(τ, loss, deepcopy(classifier), buffer)
    end
end


function show(io::IO, model::PatMat)
    print(io, "PatMat($(model.τ),$(model.loss),$(model.buffer))")
end


function findquantile(x, p)
    0 <= p <= 1 || throw(ArgumentError("input probability out of [0,1] range"))

    if p <= 0.5
        k   = floor(Int64, length(x)*p)
        ind = partialsortperm(x, k, rev = false)
    else
        k   = floor(Int64, length(x)*(1 - p)) + 1
        ind = partialsortperm(x, k, rev = true)
    end
    return x[ind], ind
end


function threshold(model::PatMat, s, y)
    return findquantile(s, 1 - model.τ)[1]
end

function gradient_threshold(model::PatMat, s, y)
    t, t_ind = findquantile(s, 1 - model.τ)

    return t, t_ind
end



############
# PatMatNP #
############
mutable struct PatMatNP <: ThresholdModel
    τ
    loss
    classifier
    buffer
    
    function PatMatNP(τ, loss, classifier, batch_size, buffer_size; T::Type = Float32)
        buffer = ScoreBuffer(batch_size, buffer_size; T = T)
        return new(τ, loss, deepcopy(classifier), buffer)
    end
end


function show(io::IO, model::PatMatNP)
    print(io, "PatMatNP($(model.τ),$(model.loss),$(model.buffer))")
end


function threshold(model::PatMatNP, s, y)
    return findquantile(s[y .== 0], 1 - model.τ)[1]
end

function gradient_threshold(model::PatMatNP, s, y)
    ind_neg = findall(y .== 0)
    t, ind  = findquantile(s[ind_neg], 1 - model.τ)
    t_ind   = ind_neg[ind]

    return t, t_ind
end