# -------------------------------------------------------------------------------
# BaseLine model
# -------------------------------------------------------------------------------
function gradient(model::BaseLineModel, pars, batch_ind::Int, batches)
    @timeit "extract batch" begin 
        x, y = batches[batch_ind]
    end

    @timeit "gradient" begin
        gs = gradient(pars) do
            loss(model, x, y) 
        end
    end
    return gs
end


# -------------------------------------------------------------------------------
# Threshold models
# -------------------------------------------------------------------------------
function gradient(model::ThresholdModel, pars, current_batch, batches)
    @timeit "gradient threshold" begin 
        x, y, s, t, ∇t_s = gradient_threshold(model, current_batch, batches)
    end

    @timeit "gradient loss" begin
        ∇l_s, ∇l_t = loss_gradient(model, s, t, y)
    end

    @timeit "copy to gpu" begin
        # T    = eltype(x)
        T    = Float32
        ∇t_s = gpu(∇t_s)
        ∇l_s = gpu(T.(∇l_s))
        ∇l_t = T.(∇l_t)
    end

    @timeit "gradient classifier" begin
        gs = gradient(pars) do
            sum(hook(∇ ->(∇l_s .+ ∇l_t .* ∇t_s) .* ∇, scores(model, x)))
        end
    end
    return gs
end


function extract_batch(model::ThresholdModel, current_batch, batches)
    x, y = batches[current_batch]

    @timeit "compute scores" begin
        s = scores(model, x)
    end

    @timeit "copy to cpu" begin
        y = cpu(y)
        s = cpu(s)
    end
    return x, y, s
end


# -------------------------------------------------------------------------------
# NoBuffer
# -------------------------------------------------------------------------------
struct NoBuffer <: Buffer; end

show(io::IO, buffer::NoBuffer) = print(io, "NoBuffer")


function gradient_threshold(model::ThresholdModel{<:NoBuffer}, current_batch, batches)
    x, y, s = extract_batch(model, current_batch, batches)

    @timeit "threshold index" begin
        t, t_ind = threshold_gradient(model, s, y)
    end

    # compute gradient of threshold with respect to scores
    ∇t_s        = zero(s)
    ∇t_s[t_ind] = 1

    return x, y, s, t, ∇t_s
end


# -------------------------------------------------------------------------------
# Scores Delay
# -------------------------------------------------------------------------------
@with_kw_noshow struct ScoresDelay <: Buffer
    batch_size
    buffer_size

    T           = Float32
    ind         = [buffer_size]
    scores      = fill(typemin(T), buffer_size)
    labels      = fill(-1, buffer_size)
    batch_inds  = Vector{Int32}(undef, buffer_size)
    sample_inds = Vector{Int32}(undef, buffer_size)
end


function ScoresDelay(batch_size, buffer_size; kwargs...)
    ScoresDelay(batch_size = batch_size, buffer_size = buffer_size; kwargs...)
end


function show(io::IO, buffer::ScoresDelay)
    print(io, "ScoresDelay($(buffer.batch_size), $(buffer.buffer_size))")
end


function update!(buff::ScoresDelay, scores, labels, batch_ind)
    n     = length(scores)
    n_max = min(buff.buffer_size - buff.ind[1], n)
    inds  = vcat(buff.ind[1] .+ (1:n_max), 1:(n-n_max))

    buff.scores[inds]      = vec(scores)
    buff.labels[inds]      = vec(labels)
    buff.batch_inds[inds] .= batch_ind
    buff.sample_inds[inds] = 1:n
    buff.ind[1]            = inds[end]
end


function gradient_threshold(model::ThresholdModel{<:ScoresDelay}, current_batch, batches)
    x, y, s = extract_batch(model, current_batch, batches)

    @timeit "update buffer" begin
        update!(model.buffer, s, y, current_batch)
    end

    @timeit "threshold index" begin
        t, t_ind     = threshold_gradient(model, model.buffer.scores, model.buffer.labels)
        t_batch_ind  = model.buffer.batch_inds[t_ind]
        t_sample_ind = model.buffer.sample_inds[t_ind]

        # select the sample that represents the threshold
        x_t = selectdim(batches[t_batch_ind][1], ndims(x), t_sample_ind)
        y_t = model.buffer.labels[t_ind]
        s_t = model.buffer.scores[t_ind]
    end

    @timeit "batch update" begin
        if current_batch != t_batch_ind
            x = cat(x, x_t, dims = ndims(x))
            y = cat(y, y_t; dims = ndims(y))
            s = cat(s, s_t; dims = ndims(s))
            t_sample_ind = length(s)
        end
    end

    # compute gradient of threshold with respect to scores
    ∇t_s             = zero(s)
    ∇t_s[t_sample_ind] = 1

    return x, y, s, t, ∇t_s
end


# -------------------------------------------------------------------------------
# Last thresahold
# -------------------------------------------------------------------------------
mutable struct LastThreshold <: Buffer
    batch_ind::Int
    sample_ind::Int
end

show(io::IO, buffer::LastThreshold) = print(io, "LastThreshold")


function update!(buff::LastThreshold, batch_ind, sample_ind)
    buff.batch_ind  = batch_ind
    buff.sample_ind = sample_ind
end


function extract_buffer(buffer::LastThreshold, batches)
    x, y = batches[buffer.batch_ind]
    x_t = selectdim(x, ndims(x), [buffer.sample_ind]) |> copy
    y_t = selectdim(y, ndims(y), [buffer.sample_ind]) |> cpu
    return x_t, y_t
end



function gradient_threshold(model::ThresholdModel{<:LastThreshold}, current_batch, batches)
    x, y, s = extract_batch(model, current_batch, batches)

    @timeit "batch update" begin
        x_t, y_t = extract_buffer(model.buffer, batches)
        s_t      = scores(model, copy(x_t)) |> cpu

        x = cat(x, x_t, dims = ndims(x))
        y = cat(y, y_t; dims = ndims(y))
        s = cat(s, s_t; dims = ndims(s))
    end

    @timeit "threshold index" begin
        t, t_ind = threshold_gradient(model, s, y)
    end

    @timeit "update buffer" begin
        if t_ind .!= length(s)
            update!(model.buffer, current_batch, t_ind)
        end
    end

    # compute gradient of threshold with respect to scores
    ∇t_s        = zero(s)
    ∇t_s[t_ind] = 1

    return x, y, s, t, ∇t_s
end