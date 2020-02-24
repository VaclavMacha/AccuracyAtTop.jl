using CategoricalArrays: recode
using Base.Iterators: partition

function binarize(y, positives)
    if isempty(positives)
        return y
    else
        return Bool.(recode(y, positives => 1, setdiff(unique(y), positives) => 0))
    end
end


function reshape_data(x, y; positives = [])
    d     = ndims(x)
    y_new = binarize(reshape(y, 1, :), positives)

    if d == 2 || d == 4
        x_new = x
    elseif d == 3
        s     = size(x)
        x_new = reshape(x, s[1:2]..., 1, s[3])
    else
        @error "unsupported x size"
    end
    return x_new, y_new
end


function get_partition(y, batch_size)
    ind_neg = findall(.~vec(y))
    ind_pos = findall(vec(y))

    n_neg = length(ind_neg)
    n_pos = length(ind_pos)
    r_pos = n_pos/(n_pos + n_neg)

    size_pos = round(Int, r_pos*batch_size)
    size_neg = batch_size - size_pos

    p_neg = partition(1:n_neg, size_neg)
    p_pos = partition(1:n_pos, size_pos)

    map(zip(p_neg, p_pos)) do (i_neg, i_pos) 
        vcat(ind_neg[i_neg], ind_pos[i_pos])
    end
end


function make_minibatch(x, y, batch_size::Integer)
    map(get_partition(y, batch_size)) do inds
        (selectdim(x, ndims(x), inds), y[:, inds])
    end
end

# test model
using EvalMetrics, EvalPlots, DataFrames
pyplot()

function test_model(models::AbstractArray{<:Model}, x, y, rec)
    plts = []
    dfs  = []
    for m in models
        df, plt = test_model(m, x, y, rec)
        push!(dfs, df)
        push!(plts, plt)
    end
    k   = length(plts)
    plt = plot(plts..., layout = (1,k), size = (k*400, 400), dpi = 200)
    display(plt)

    table = vcat(dfs...)
    display(table)

    return table, plt
end


function test_model(model::M, x, y, rec) where {M<:Model}
    s = vec(scores(model, x))

    df = DataFrame(model = M.name)
    for r in rec
        t    = threshold_at_tpr(y, s, r)
        df[:, Symbol("PatR_", r)] = [precision(y, s, t)]
    end

    thres = [(value = quantile(s, 0.95), label = "0.05-quantile")] 
    plt   = scoresdensity(y, s; title = "$(M.name)", thres = thres)
    return df, plt
end