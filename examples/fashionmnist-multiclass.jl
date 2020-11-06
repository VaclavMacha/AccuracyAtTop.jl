using AccuracyAtTop, EvalMetrics, Flux, MLDatasets, Plots, ProgressMeter

using Flux: cpu, gpu, onehotbatch, onecold
using Flux.Data: DataLoader
using Random: randperm
using IterTools: ncycle

# ------------------------------------------------------------------------------------------
# Auxiliary functions
# ------------------------------------------------------------------------------------------
function reshape_data(X::AbstractArray{T, 3}, y::AbstractVector; runon = cpu) where T
    s = size(X)
    return runon(reshape(X, s[1], s[2], 1, s[3])), runon(onehotbatch(y, 0:9))
end

function eval_scores(model, batches)
    targets = []
    scores = []

    @showprogress for (x,y) in batches
        push!(targets, cpu(y))
        push!(scores, cpu(model(x)))
    end
    return hcat(targets...), hcat(scores...)
end

# ------------------------------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------------------------------
T = Float32
pos_class = 0
batch_size = 128
runon = gpu

X_train, y_train = reshape_data(MLDatasets.FashionMNIST.traindata(T)...; runon = runon);
X_test, y_test = reshape_data(MLDatasets.FashionMNIST.testdata(T)...; runon = runon);

batches_train = DataLoader((X_train, y_train); batchsize = batch_size, shuffle = true)
batches_test = DataLoader((X_test, y_test); batchsize = batch_size, shuffle = true)

# ------------------------------------------------------------------------------------------
# Model preparation
# ------------------------------------------------------------------------------------------
model = Chain(
    # First convolution
    Conv((5, 5), 1=>20, stride=(1,1), relu),
    MaxPool((2,2)),

    # Second convolution
    Conv((5, 5), 20=>50, stride=(1,1), relu),
    MaxPool((2,2)),

    flatten,
    Dense(4*4*50, 500),
    Dense(500, 10)
) |> gpu

# objective
surrogate = quadratic;
pars = params(model);
thres = FPRate(0.01)
γ = T(1e-4)
w = ones(T, 10) ./ 10
w[7] = 1

sqsum(x) = sum(abs2, x)

function loss(x, y)
    s = model(x)
    t = [threshold(thres, y[i, :], s[i, :]) for i in 1:10]
    return fnr(y, s, t, surrogate; weights = w) + γ * sum(sqsum, pars)
end

x, y = collect(batches_train)[1]
@time loss(x, y)

# ------------------------------------------------------------------------------------------
# Model training
# ------------------------------------------------------------------------------------------
opt = ADAM(0.001);
n_epochs = 10

Flux.train!(loss, pars, ncycle(batches_train, n_epochs), opt)

# ------------------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------------------
tar_train, s_train = eval_scores(model, batches_train)
tar_test, s_test = eval_scores(model, batches_test)


for ind in 1:10
    plt = plot(
        prplot(tar_train[ind, :], s_train[ind, :], title = "Train class = $(ind)"),
        prplot(tar_test[ind, :], s_test[ind, :]; title = "Test"),
        size = (800, 400)
    )
    display(plt)
end

for ind in 1:10
    plt = plot(
        rocplot(tar_train[ind, :], s_train[ind, :], title = "Train = $(ind)", xscale = :log10, xlims = (1e-4, 1)),
        rocplot(tar_test[ind, :], s_test[ind, :]; title = "Test", xscale = :log10, xlims = (1e-4, 1)),
        size = (800, 400)
    )
    display(plt)
end
