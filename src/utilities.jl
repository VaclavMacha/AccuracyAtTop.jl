# -------------------------------------------------------------------------------
# Mini batches
# -------------------------------------------------------------------------------
function binarize(y, positives)
    negatives = setdiff(unique(y), positives)
    return Bool.(recode(y, positives => 1, negatives => 0))
end


function reshape_data(x, y, positives)
    d     = ndims(x)
    y_new = binarize(reshape(y, 1, :), positives)

    if d == 2 || d == 4
        x_new = x
    elseif d == 3
        s     = size(x)
        x_new = reshape(x, s[1:2]..., 1, s[3])
    else
        @error "x: unsupported size"
    end
    return x_new, y_new
end


function get_partition(y::BitArray, batch_size::Int)
    inds_neg = findall(vec(y) .== 0)
    inds_pos = findall(vec(y) .== 1)

    n       = length(y)
    n_neg   = length(inds_neg)
    n_pos   = length(inds_pos)
    n_batch = ceil(Int, n/batch_size)
    
    ratio_neg = n_neg/n
    k_neg     = round(Int, n_batch*ratio_neg*batch_size)
    k_pos     = n_batch*batch_size - k_neg

    neg_pert = shuffle(vcat(1:n_neg, sample(1:n_neg, k_neg - n_neg))) 
    pos_pert = shuffle(vcat(1:n_pos, sample(1:n_pos, k_pos - n_pos)))

    li_neg, ui_neg = 0, 0
    li_pos, ui_pos = 0, 0
    map(1:n_batch) do i
        li_neg = ui_neg + 1
        ui_neg = round(Int64, i/n_batch * k_neg)

        li_pos = ui_pos + 1
        ui_pos = li_pos + batch_size - (ui_neg - li_neg) - 2
        
        i_neg = inds_neg[neg_pert[li_neg:ui_neg]]
        i_pos = inds_pos[pos_pert[li_pos:ui_pos]]
        return vcat(i_neg, i_pos)
    end
end



function make_minibatch(x, y::BitArray, batch_size::Integer)
    map(get_partition(y, batch_size)) do inds
        x_i = selectdim(x, ndims(x), inds)
        y_i = selectdim(y, ndims(y), inds)
        return (Array(x_i), Array(y_i))
    end
end


# -------------------------------------------------------------------------------
# Gradient test
# -------------------------------------------------------------------------------
function test_gradient(model_in::Model, batch; verbose::Bool = false)
    # random direction
    pars_in = params(model_in)
    dirs    = rand_direction(pars_in)

    # compute loss function and gradient
    gs = gradient(model_in, pars_in, 1, [batch])
    ∇  = map(zip(pars_in, dirs)) do (par, dir)
        sum(gs[par] .* dir)
    end |> sum

    # compute numerical gradient
    err = map(10.0 .^ (-12:0.2:-1)) do s
        # numerical gradient
        f1   = loss(new_model(model_in, dirs, s), batch...)
        f2   = loss(new_model(model_in, dirs, - s), batch...)
        ∇num = (f1 - f2)/(2*s)

        # show
        verbose && @show (∇, ∇num) 

        return norm(∇ - ∇num)/max(norm(∇), norm(∇num))
    end |> minimum

    @info "minimal error: $err"
    return 
end


function rand_direction(pars::Params)
    dirs = deepcopy(pars)
    for (dir, par) in zip(dirs, pars)
        dir .= gpu(rand(eltype(par), size(par)...))
    end
    return dirs
end


function new_model(model_in::Model, dirs, s)
    #copy model
    model = deepcopy(model_in)

    # update parameters of the copy
    for (par, dir) in zip(params(model), dirs)
        par .+= s .* dir
    end
    return model
end


# -------------------------------------------------------------------------------
# train model
# -------------------------------------------------------------------------------
function train!(model::Model, batches, optimiser; cb = () -> ())
    pars    = params(model)
    cb      = runall(cb)
    indexes = randperm(length(batches))

    for ind in indexes
        try
            gs = gradient(model, pars, ind, batches)
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
    end
end


# -------------------------------------------------------------------------------
# auxiliary functions
# -------------------------------------------------------------------------------
function scores(model::Model, x)
    return model.classifier(x)
end


function cpu(model_in::Model)
    model = deepcopy(model_in)
    model.classifier = cpu(model.classifier)
    return model
end


function gpu(model_in::Model)
    model = deepcopy(model_in)
    model.classifier = gpu(model.classifier)
    return mdl
end


function params(model::Model)
    return params(model.classifier)
end


# -------------------------------------------------------------------------------
# save and load model
# -------------------------------------------------------------------------------
save_model(name, model) = BSON.bson(name, model = cpu(model))
load_model(name)        = BSON.load(name)