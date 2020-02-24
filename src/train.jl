function train!(model::M, batches, opt, n_epochs::Int, test_set = batches[1]; kwargs...) where {M<:Model}
    L0 = loss(model, test_set...)
    p  = Progress(n_epochs; desc = "$(M.name): ")

    for epoch in 1:n_epochs
        train!(model, batches, opt; kwargs...)
        next!(p; showvalues = [(:L0, L0), (:L, loss(model, test_set...))])
    end;
end


function train!(model::Simple, batches, opt; cb = () -> ())
    cb = runall(cb)

    for batch in batches
        try
            gs = gradient(model.params) do
                loss(model, batch...) 
            end
            update!(opt, model.params, gs)
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


function train!(model::ThresholdModel, batches, opt;
                cb = () -> (),
                buffer_size::Integer = 10*length(batches[1][2]))

    cb   = runall(cb)
    buff = ScoresBuffer(batches[1]; buffer_size = buffer_size)

    for batch_ind in eachindex(batches)
        try
            gs = gradient(model, buff, batch_ind, batches)
            update!(opt, model.params, gs)
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



function train2!(model::M, batches, opt, n_epochs::Int, test_set = batches[1]; kwargs...) where {M<:Model}
    L0 = loss(model, test_set...)
    p  = Progress(n_epochs; desc = "$(M.name) (type 2): ")

    for epoch in 1:n_epochs
        train2!(model, batches, opt; kwargs...)
        next!(p; showvalues = [(:L0, L0), (:L, loss(model, test_set...))])
    end;
end



mutable struct Threshold
    x
    y
end


function train2!(model::ThresholdModel, batches, opt; cb = () -> ())

    cb    = runall(cb)
    
    x, y = batches[rand(1:length(batches))]
    ind  = rand(1:length(y))
    x    = selectdim(x, ndims(x), ind)
    y    = selectdim(y, ndims(y), ind)
    thres = Threshold(x, y)

    for batch in batches
        try
            gs = gradient2(model, batch, thres)
            update!(opt, model.params, gs)
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
