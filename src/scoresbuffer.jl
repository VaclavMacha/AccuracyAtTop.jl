# -------------------------------------------------------------------------------
# Sore buffer
# -------------------------------------------------------------------------------
mutable struct ScoreBuffer{T, S, L, I}
    batch_size::T
    buffer_size::T
    ind::T

    scores::S
    labels::L
    batch_inds::I
    sample_inds::I
end


function show(io::IO, buffer::ScoreBuffer)
    print(io, "ScoreBuffer($(buffer.batch_size), $(buffer.buffer_size))")
end


function ScoreBuffer(batch_size::Int, buffer_size::Int; T::Type = Float32)
    ind         = buffer_size
    scores      = fill(typemin(T), buffer_size)
    labels      = fill(0, buffer_size)
    batch_inds  = Vector{Int32}(undef, buffer_size)
    sample_inds = Vector{Int32}(undef, buffer_size)

    return ScoreBuffer(batch_size, buffer_size, ind, scores,
                       labels, batch_inds, sample_inds)
end


function update!(buff::ScoreBuffer, scores, labels, batch_ind)
    n     = length(scores)
    n_max = min(buff.buffer_size - buff.ind, n)
    inds  = vcat(buff.ind .+ (1:n_max), 1:(n-n_max))

    buff.scores[inds]      = vec(scores)
    buff.labels[inds]      = vec(labels)
    buff.batch_inds[inds] .= batch_ind
    buff.sample_inds[inds] = 1:n
    buff.ind               = inds[end]
end