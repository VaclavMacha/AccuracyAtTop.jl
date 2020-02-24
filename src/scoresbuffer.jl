struct ScoresBuffer
    batch_ind
    score_ind
    s
    y
end


function ScoresBuffer(batch; buffer_size::Int = 100)
    x, y = batch

    return ScoresBuffer(CircularBuffer{Int64}(buffer_size),
                        CircularBuffer{Int64}(buffer_size),
                        CircularBuffer{eltype(x)}(buffer_size),
                        CircularBuffer{eltype(y)}(buffer_size))
end


function update!(sb::ScoresBuffer, batch_ind, s, y)
    n = length(y)
    append!(sb.batch_ind, fill(batch_ind, n))
    append!(sb.score_ind, 1:n)
    append!(sb.s, vec(s))
    append!(sb.y, vec(y))
end