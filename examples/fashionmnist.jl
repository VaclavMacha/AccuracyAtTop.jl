using AccuracyAtTop, EvalMetrics, Flux, MLDatasets, Plots, ProgressMeter

using Flux: gpu
using Random: randperm
using Base.Iterators: partition

# ------------------------------------------------------------------------------------------
# Auxiliary functions
# ------------------------------------------------------------------------------------------
function reshape_data(X::AbstractArray{T, 3}, y::AbstractVector, pos_class) where T
    s = size(X)
    return reshape(X, s[1], s[2], 1, s[3]), reshape(y, 1, :) .== pos_class
end

function eval_scores(model, batches)
    targets = Bool[]
    scores = Float32[]

    @showprogress for (x,y) in batches
        append!(targets, vec(y))
        append!(scores, vec(model(x)))
    end
    return targets, scores
end

# ------------------------------------------------------------------------------------------
# Data preparation
# ------------------------------------------------------------------------------------------
T = Float32
pos_class = 0
batch_size = 128

X_train, y_train = reshape_data(MLDatasets.FashionMNIST.traindata(T)..., pos_class);
X_test, y_test = reshape_data(MLDatasets.FashionMNIST.testdata(T)..., pos_class);

batches_train = map(partition(randperm(size(y_train, 2)), batch_size)) do inds
    return (gpu(X_train[:, :, :, inds]), gpu(y_train[:, inds]))
end

batches_test = map(partition(randperm(size(y_test, 2)), batch_size)) do inds
    return (gpu(X_test[:, :, :, inds]), gpu(y_test[:, inds]))
end

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
    Dense(500, 1)
) |> gpu

# objective
surrogate = quadratic;
pars = params(model);
thres = Maximum(; samples = NegSamples);
γ = T(1e-4)

sqsum(x) = sum(abs2, x)

function loss(x, y)
    s = model(x)
    t = threshold(thres, y, s)
    return fnr(y, s, t, surrogate) + γ * sum(sqsum, pars)
end

# test
@time loss(batches_train[1]...)

# ------------------------------------------------------------------------------------------
# Model training
# ------------------------------------------------------------------------------------------
opt = ADAM(0.001);
n_epochs = 10

@showprogress for i in 1:n_epochs
    Flux.train!(loss, pars, batches_train, opt)
end

# ------------------------------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------------------------------
tar_train, s_train = eval_scores(model, batches_train)
tar_test, s_test = eval_scores(model, batches_test)

plot(
    prplot(tar_train, s_train, title = "Train"),
    prplot(tar_test, s_test; title = "Test"),
    size = (800, 400)
)
