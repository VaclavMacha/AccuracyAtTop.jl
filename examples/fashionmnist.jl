using AccuracyAtTop
using EvalMetrics
using Flux
using MLDatasets
using Plots
using ProgressMeter

using Flux: gpu

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
using Flux.Data: DataLoader

T = Float32
pos_class = 0
batchsize = 128
device = gpu

X_train, y_train = reshape_data(MLDatasets.FashionMNIST.traindata(T)..., pos_class);
X_test, y_test = reshape_data(MLDatasets.FashionMNIST.testdata(T)..., pos_class);

batches_train = (device(batch) for batch in DataLoader((X_train, y_train); batchsize))
batches_test = (device(batch) for batch in DataLoader((X_test, y_test); batchsize))

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
) |> device

# objective
surrogate = quadratic;
pars = params(model);
aatp = DeepTopPush()

loss(x, y) = objective(aatp, y, model(x); surrogate)

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

kwargs = (xscale = :log10, xlims = (1e-5, 1))
plot(
    rocplot(tar_train, s_train; title = "Train", kwargs...),
    rocplot(tar_test, s_test; title = "Test", kwargs...),
    size = (800, 400)
)
