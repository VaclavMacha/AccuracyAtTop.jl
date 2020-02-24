# -------------------------------------------------------------------------------
# Surroagtes
# -------------------------------------------------------------------------------
hinge(x)      = max(zero(x), one(x) + x)
quadratic(x)  = max(zero(x), one(x) + x)^2


# -------------------------------------------------------------------------------
# Model
# -------------------------------------------------------------------------------
abstract type Model; end


function scores(model::Model, x)
    return model.classifier(x)
end



################
# Simple model #
################
mutable struct Simple <: Model
    classifier
    params

    function Simple(classifier)
        cls = deepcopy(classifier)
        return new(cls, params(cls))
    end
end


function loss(model::Simple, x, y)
    s = scores(model, x)
    return sum(binarycrossentropy.(sigmoid.(s), y))/length(y)
end


# -------------------------------------------------------------------------------
# Threshold models
# -------------------------------------------------------------------------------
abstract type ThresholdModel <: Model; end

function loss(model::ThresholdModel, x, y)
    s = scores(model, x)
    t = threshold(model, s, y)
    return loss(model, s, t, y)
end


function loss(model::ThresholdModel, s, t, y)
    s_pos = s[y .== 1]
    n_pos = length(s_pos)

    return sum(model.surrogate.(t .- s_pos))/n_pos
end


function gradient(model::ThresholdModel, buff::ScoresBuffer, batch_ind, batches)
    x, y, s, t, ∇t_s = gradient_threshold(model, buff, batch_ind, batches)

    ∇l_s, ∇l_t = gradient(params(s, t)) do 
        loss(model, s, t, y) 
    end |> ∇ -> (∇[s], ∇[t])

    T    = eltype(x)
    ∇l_s = T.(∇l_s)
    ∇l_t = T.(∇l_t)

    return gradient(model.params) do
        sum(hook(∇ ->(∇l_s .+ ∇l_t .* ∇t_s) .* ∇, model.classifier(x)))
    end
end


function gradient2(model::ThresholdModel, batch, thres)
    x, y, s, t, ∇t_s = gradient_threshold2(model, batch, thres)

    ∇l_s, ∇l_t = gradient(params(s, t)) do 
        loss(model, s, t, y) 
    end |> ∇ -> (∇[s], ∇[t])

    T    = eltype(x)
    ∇l_s = T.(∇l_s)
    ∇l_t = T.(∇l_t)

    return gradient(model.params) do
        sum(hook(∇ ->(∇l_s .+ ∇l_t .* ∇t_s) .* ∇, model.classifier(x)))
    end
end


function extract_batch(model::ThresholdModel, buff::ScoresBuffer, batch_ind, batches)
    x, y = batches[batch_ind]
    s    = scores(model, x)
    update!(buff, batch_ind, s, y)
    return x, y, s
end


function update_batch(buff::ScoresBuffer, x, y, s, batch_ind, inds, batches)
    batch_inds = @view buff.batch_ind[inds]
    score_inds = @view buff.score_ind[inds]
    dim        = ndims(x)

    ind_map = IdDict()
    for key in unique(batch_inds)
        i            = findall(batch_inds .== key)
        ind_map[key] = score_inds[i]

        key == batch_ind && continue
        x = cat(x, selectdim(batches[key][1], dim, i), dims = dim)
        s = hcat(s, reshape(buff.s[i], 1, :))
        y = hcat(y, reshape(buff.y[i], 1, :))
    end
    return x, y, s, ind_map
end


###########
# TopPush #
###########
mutable struct TopPush <: ThresholdModel
    surrogate
    classifier
    params
    
    function TopPush(surrogate, classifier)
        cls = deepcopy(classifier)
        return new(surrogate, cls, params(cls))
    end
end


threshold(model::TopPush, s, y) = maximum(s[y .== 0])


function gradient_threshold(model::TopPush, buff::ScoresBuffer, batch_ind, batches)
    x, y, s = extract_batch(model, buff, batch_ind, batches)

    ind_neg   = findall(vec(buff.y .== 0))
    t, t_ind  = findmax(buff.s[ind_neg])
    inds      = [ind_neg[t_ind]...]

    ∇t_s = zero(s)
    x, y, s, ind_map = update_batch(buff, x, y, s, batch_ind, inds, batches)

    if haskey(ind_map, batch_ind)
        ∇t_s[ind_map[batch_ind]] .= 1
    else
        T    = eltype(∇t_s) 
        ∇t_s = hcat(∇t_s, T.(1))
    end
    return x, y, s, [t], ∇t_s
end


function gradient_threshold2(model::TopPush, batch, thres)
    x_d = ndims(batch[1])
    y_d = ndims(batch[2])
    x = cat(batch[1], thres.x; dims = x_d)
    y = cat(batch[2], thres.y; dims = y_d)
    s = scores(model, x)

    ind_neg   = findall(vec(y .== 0))
    t, t_ind  = findmax(s[ind_neg])
    ind       = ind_neg[t_ind]

    ∇t_s      = zero(s)
    ∇t_s[ind] = 1

    thres.x = selectdim(x, x_d, ind)
    thres.y = selectdim(y, y_d, ind) 

    return x, y, s, [t], ∇t_s
end


############
# TopPushK #
############
mutable struct TopPushK <: ThresholdModel
    K
    surrogate
    classifier
    params

    function TopPushK(K, surrogate, classifier)
        cls = deepcopy(classifier)
        return new(surrogate, cls, params(cls))
    end
end


threshold(model::TopPushK, s, y)   = partialsort(s[y .== 0], model.K, rev = true)


function gradient_threshold(model::TopPushK, buff::ScoresBuffer, batch_ind, batches)
    x, y, s = extract_batch(model, buff, batch_ind, batches)

    ind_neg = findall(vec(buff.y .== 0))
    t_ind   = partialsortperm(buff.s[ind_neg], model.K; rev = true)
    t       = buff.s[ind_neg[t_ind]]
    inds    = [ind_neg[t_ind]...]

    ∇t_s = zero(s)
    x, y, s, ind_map = update_batch(buff, x, y, s, batch_ind, inds, batches)

    if haskey(ind_map, batch_ind)
        ∇t_s[ind_map[batch_ind]] .= 1
    else
        T    = eltype(∇t_s) 
        ∇t_s = hcat(∇t_s, T.(1))
    end
    return x, y, s, [t], ∇t_s
end