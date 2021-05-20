# samplers
struct LogUniform <: Sampleable{Univariate,Continuous}
    d

    LogUniform(a, b) = new(Uniform(log10(a), log10(b)))
end

Base.rand(rng::AbstractRNG, s::LogUniform) = 10 ^ rand(rng, s.d)

# Batch partitioning with buffer
using Random: AbstractRNG, shuffle!, GLOBAL_RNG, randperm

struct BatchPartition{R<:AbstractRNG}
    indices::Vector{Int}
    n::Int
    batchsize::Int
    epochs::Int
    epochlength::Int
    batches::Int
    lastbatch::Vector{Int}
    shuffle::Bool
    include_buffer::Bool
    buffer
    rng::R
end

function BatchPartition(
    n;
    epochs = 1,
    batchsize = 1,
    lastbatch = rand(1:n, batchsize),
    shuffle = false,
    include_buffer = false,
    buffer = () -> Int[],
    rng = GLOBAL_RNG
)

    batchsize > 0 || throw(ArgumentError("batchsize must be positive integer"))
    batchsize <= n || throw(ArgumentError("number of observations less than batchsize"))

    return BatchPartition(
        collect(1:n),
        n,
        batchsize,
        epochs,
        n ÷ batchsize,
        epochs * (n ÷ batchsize),
        lastbatch,
        shuffle,
        include_buffer,
        buffer,
        rng
    )
end

function Base.show(io::IO, d::BatchPartition)
    println(io, "BatchPartition:")
    println(io, "  ⋅ batch size: ", d.batchsize)
    println(io, "  ⋅ epoch length: ", d.epochlength)
    println(io, "  ⋅ number of epochs: ", d.epochs, " (", d.batches, " batches)")
    println(io, "  ⋅ shuffle: ", d.shuffle)
    print(io,   "  ⋅ include buffer: ", d.include_buffer)
end

Base.length(d::BatchPartition) = d.batches
Base.eltype(::BatchPartition) = Vector{Int}

function Base.iterate(d::BatchPartition, batch_ind = 1)
    if batch_ind > d.batches
        return nothing
    end
    if batch_ind % d.epochlength == 1 && d.shuffle
        shuffle!(d.rng, d.indices)
    end

    inds = collect(1:d.batchsize) .+ rem(batch_ind - 1, d.epochlength)*d.batchsize
    d.include_buffer && update_indices!(d, inds, d.buffer())
    batch = d.indices[inds]
    d.lastbatch .= batch
    return (batch, batch_ind + 1)
end

function update_indices!(::BatchPartition, inds, b_inds)
    @warn("indices provided by buffer muset be of type Int or Vector{Int}")
    return
end

function update_indices!(d::BatchPartition, inds, b_inds::Union{Int, Vector{Int}})
    if !all(0 .< b_inds .<= d.batchsize)
        @warn("indices in buffer are corrupted")
        return
    end
    k = d.lastbatch[b_inds]
    if !all(0 .< k .<= d.n)
        @warn("last batch is corrupted")
        return
    end

    pos = randperm(d.batchsize)[1:length(k)]
    inds[pos] .= k
    return
end
