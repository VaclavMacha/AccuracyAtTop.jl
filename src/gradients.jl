# -------------------------------------------------------------------------------
# BaseLine model
# -------------------------------------------------------------------------------
function gradient(model::BaseLineModel, pars, batches, ind)
    @timeit "extract batch" begin 
        data, target = batches[ind]
    end

    @timeit "gradient" begin
        gs = gradient(pars) do
            loss(model, data, target) 
        end
    end
    return gs
end


function gradient(model::BalancedBaseLine, pars, batches, ind)
    @timeit "extract batch" begin 
        data, target = batches[ind]
    end

    @timeit "gradient" begin
        w = weights(target)
        gs = gradient(pars) do
            loss(model, data, target, w) 
        end
    end
    return gs
end


# -------------------------------------------------------------------------------
# Threshold models
# -------------------------------------------------------------------------------
function extract_batch(model::ThresholdModel, batches, ind)
    @timeit "extract batch" begin
        data, target = batches[ind]
    end

    @timeit "scores computation" begin
        scores = model(data)
    end

    @timeit "copy to cpu" begin
        target = cpu(target)
        scores = cpu(scores)
    end

    return data, target, scores
end


function gradient(model::ThresholdModel, pars, batches, ind)
    @timeit "threshold gradient" begin
        data, target, scores, t, ∇t_s = threshold_gradient(model, batches, ind)
        model.threshold = t
    end


    @timeit "loss gradient" begin
        ∇l_s, ∇l_t = loss_gradient(model, target, scores, t)
    end

    @timeit "copy to gpu" begin
        ∇t_s = gpu(∇t_s)
        ∇l_s = gpu(∇l_s)
    end

    @timeit "classifier gradient" begin
        gs = gradient(pars) do
            sum(hook(∇ ->(∇l_s .+ ∇l_t .* ∇t_s) .* ∇, model(data)))
        end
    end
    return gs
end


# -------------------------------------------------------------------------------
# NoBuffer
# -------------------------------------------------------------------------------
struct NoBuffer <: Buffer; end


show(io::IO, buffer::NoBuffer) =
    print(io, "NoBuffer")


function threshold_gradient(model::ThresholdModel{B}, batches, ind) where {B <: NoBuffer}
    data, target, scores = extract_batch(model, batches, ind)

    t, t_ind    = find_threshold(model, target, scores)
    ∇t_s        = zero(scores)
    ∇t_s[t_ind] = 1

    return data, target, scores, t, ∇t_s
end


# -------------------------------------------------------------------------------
# Scores Delay
# -------------------------------------------------------------------------------
@with_kw_noshow struct ScoresDelay <: Buffer
    T::Type
    buffer_size::Int

    scores      = CircularBuffer{T}(buffer_size)
    target      = CircularBuffer{Int}(buffer_size)
    batch_inds  = CircularBuffer{Int}(buffer_size)
    sample_inds = CircularBuffer{Int}(buffer_size)
end


ScoresDelay(T, buffer_size; kwargs...) =
    ScoresDelay(T = T, buffer_size = buffer_size; kwargs...)


show(io::IO, buffer::ScoresDelay) =
    print(io, "ScoresDelay($(buffer.T), $(buffer.buffer_size))")


function update!(buffer::ScoresDelay, target, scores, batch_ind)
    n = length(scores)

    append!(buffer.scores, vec(scores))
    append!(buffer.target, vec(target))
    append!(buffer.batch_inds, fill(batch_ind, n))
    append!(buffer.sample_inds, 1:n)
end


function threshold_gradient(model::ThresholdModel{B}, batches, ind) where {B <: ScoresDelay}
    data, target, scores = extract_batch(model, batches, ind)

    # update buffer
    buffer = model.buffer
    update!(buffer, target, scores, ind)

    # find threshold
    t, t_ind   = find_threshold(model, buffer.target, buffer.scores)
    batch_ind  = buffer.batch_inds[t_ind]
    sample_ind = buffer.sample_inds[t_ind]

    # update batch
    if ind != batch_ind
        sample     = getdim(batches[batch_ind][1], ndims(data), sample_ind)
        data       = cat(data, sample, dims = ndims(data))
        target     = hcat(target, buffer.target[t_ind])
        scores     = hcat(scores, buffer.scores[t_ind])
        sample_ind = length(scores)
    end

    # compute gradient of threshold with respect to scores
    ∇t_s             = zero(scores)
    ∇t_s[sample_ind] = 1

    return data, target, scores, t, ∇t_s
end


# -------------------------------------------------------------------------------
# Last thresahold
# -------------------------------------------------------------------------------
mutable struct LastThreshold <: Buffer
    batch_ind::Int
    sample_ind::Int
end


show(io::IO, buffer::LastThreshold) =
    print(io, "LastThreshold")


function update!(buffer::LastThreshold, batch_ind, sample_ind)
    buffer.batch_ind  = batch_ind
    buffer.sample_ind = sample_ind
end


function extract_buffer(model::ThresholdModel{B}, batches) where {B <: LastThreshold}
    buffer       = model.buffer
    data, target = batches[buffer.batch_ind]

    data_i   = getdim(data, ndims(data), [buffer.sample_ind]);
    target_i = getdim(target, ndims(target), [buffer.sample_ind])

    return copy(data_i), cpu(target_i), cpu(model(data_i))
end


function threshold_gradient(model::ThresholdModel{B}, batches, ind) where {B <: LastThreshold}
    data, target, scores = extract_batch(model, batches, ind)

    # extract buffer
    buffer = model.buffer

    data_t, target_t, scores_t = extract_buffer(model, batches)

    data   = cat(data, data_t, dims = ndims(data))
    target = hcat(target, target_t)
    scores = hcat(scores, scores_t)

    # compute gradient of threshold with respect to scores
    t, t_ind    = find_threshold(model, target, scores)
    ∇t_s        = zero(scores)
    ∇t_s[t_ind] = 1

    # update buffer
    if t_ind != length(scores)
        update!(buffer, ind, t_ind)
    end

    return data, target, scores, t, ∇t_s
end