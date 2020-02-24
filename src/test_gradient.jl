function test_gradient(model::Model, batch; verbose::Bool = false)
    x, y = batch
    buff = ScoresBuffer(batch; buffer_size = length(y))
    pars = deepcopy(model.params)
    dirs = rand_direction(pars)

    f() = loss(model, x, y)
    gs  = gradient(model, buff, 1, [batch])

    fun  = f()
    ∇ = map(zip(model.params, dirs)) do (par, dir)
        sum(gs[par] .* dir)
    end |> sum

    err = map(10.0 .^ (-12:0.2:-1)) do s
        # compute derivation
        update_params!(model, pars, dirs, s)
        f1   = f()
        update_params!(model, pars, dirs, - s)
        f2   = f()
        ∇num = (f1 - f2)/(2*s)

        # compute derivation
        verbose && @show (∇, ∇num) 

        return norm(∇ - ∇num)/max(norm(∇), norm(∇num))
    end |> minimum

    @info "minimal error: $err"
    reset_params!(model, pars)
    return 
end


function rand_direction(pars::Params)
    dirs = deepcopy(pars)
    for (dir, par) in zip(dirs, pars)
        dir .= rand(eltype(par), size(par)...)
    end
    return dirs
end


function update_params!(model::Model, pars::Params, dirs, s)
    for (par, par_old, dir) in zip(model.params, pars, dirs)
        par .= par_old .+ s .* dir
    end
end


function reset_params!(model::Model, pars::Params)
    for (par, par_old) in zip(model.params, pars)
        par .= par_old
    end
end