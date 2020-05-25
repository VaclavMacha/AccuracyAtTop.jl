
# Title

This repository is a complementary material to our paper *End-to-end network for Accuracy at the top*. This paper was submitted to the [NeurIPS | 2020 Thirty-fourth Conference on Neural Information Processing Systems](https://nips.cc/Conferences/2020).

 ## Instalation

To install this package use [Pkg REPL]([https://docs.julialang.org/en/v1/stdlib/Pkg/index.html](https://docs.julialang.org/en/v1/stdlib/Pkg/index.html)) and following command

```julia
 add https://github.com/VaclavMacha/AccuracyAtTop.jl
```

 ## Usage
 
 This package provides the following methods
* Models:
    1. `BaseLine(classifier; [objective, ...])`
    2. `BalancedBaseLine(classifier; [objective, ...])`
    3. `TopPush(classifier; [objective, ...])`
    4. `TopPushK(K, classifier; [surrogate, buffer, ...])`
    5. `PatMat(tau, classifier; [surrogate, buffer, ...])`
    6. `PatMatNP(tau, classifier; [surrogate, buffer, ...])`
    7. `RecAtK(K, classifier; [surrogate, buffer, ...])`
    8. `PrecAtRec(rec, classifier; [surrogate, buffer, ...])`
* Buffers:
    1. `NoBuffer()`
    2. `ScoresDelay(T, buffer_size)`
    3. `LastThreshold(batch_ind, sample_ind)`
* Surrogates:
    1. `Hinge(theta)`
    2. `Quadratic(theta)`
    3. `Sigmoid(theta)`

The following example shows the basic use of the package. The complete set of all experiments included in the article is in a separate repository [AccuracyAtTop_experiments.jl](https://github.com/VaclavMacha/AccuracyAtTop_experiments.jl)

```julia
using AccuracyAtTop, MLDatasets, MLDataPattern
import AccuracyAtTop: @runepochs

function make_minibatches(x, y, batchsize, n)
    map(RandomBatches((x,y), size = batchsize, count = n)) do (x, y)
        Array(x), Array(y)
    end
end

T = Float32

# load dataset and create minibatches
x, y = MLDatasets.FashionMNIST.traindata(T);
x = Array(reshape(x, 28, 28, 1, :));
y = Array(reshape(y .== 0, 1, :));
batches = make_minibatches(x, y, 64, 1000) |> gpu;


# create model
cls = Chain(
    Conv((5, 5), 1=>20, stride=(1,1), relu),
    MaxPool((2,2)),

    Conv((5, 5), 20=>50, stride=(1,1), relu),
    MaxPool((2,2)),

    x -> reshape(x, :, size(x, 4)),
    Dense(4*4*50, 500),
    Dense(500, 1)
);

model = TopPush(cls; T = T, surrogate = Quadratic(), buffer = LastThreshold(1,1)) |> gpu


# train model
opt = Descent(0.0001)
@runepochs 10 train!(model, batches, opt)


# test
using EvalMetrics

scores = Float32[]
target = Bool[]

map(batches) do (x, y)
    append!(target, vec(cpu(y)))
    append!(scores, vec(cpu(model(x))))
end

t = threshold(cpu(model), target, scores)
c = counts(target, scores, t; classes = (false, true))

precision(c)
accuracy(c)
```