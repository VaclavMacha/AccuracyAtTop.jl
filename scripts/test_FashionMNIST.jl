using Revise
using AccuracyAtTop

using Statistics, LinearAlgebra, MLDatasets
import Flux: @epochs, throttle, sigmoid, binarycrossentropy
import BSON: @load

include("./scripts/makebatch.jl")


# -------------------------------------------------------------------------------
# Data
# -------------------------------------------------------------------------------
T = Float32
train_set = reshape_data(FashionMNIST.traindata(T)...; positives = 0)
test_set  = reshape_data(FashionMNIST.testdata(T)...; positives = 0)

batch_size = 64
batches    = make_minibatch(train_set..., batch_size);


# -------------------------------------------------------------------------------
# Neural network
# -------------------------------------------------------------------------------
# classifier = Chain(
#     # First convolution, operating upon a 28x28 image
#     Conv((3, 3), 1=>16, pad=(1,1), relu),
#     MaxPool((2,2)),

#     # Second convolution, operating upon a 14x14 image
#     Conv((3, 3), 16=>32, pad=(1,1), relu),
#     MaxPool((2,2)),

#     # Third convolution, operating upon a 7x7 image
#     Conv((3, 3), 32=>32, pad=(1,1), relu),
#     MaxPool((2,2)),

#     x -> reshape(x, :, size(x, 4)),
#     Dense(288, 1)
# )

classifier = Chain(
    x -> reshape(x, :, size(x, 4)),
    Dense(784, 1)
)


# -------------------------------------------------------------------------------
# Models
# -------------------------------------------------------------------------------
model_simple = Simple(classifier);
opt_simple   = Descent(0.001)

model1 = TopPush(hinge, classifier);
opt1   = Descent(0.001)
model2 = TopPush(hinge, classifier);
opt2   = Descent(0.001)
model3 = TopPush(hinge, classifier);
opt3   = Descent(0.001)


# test
# loss(model, batches[1]...)
# scores(model, batches[1][1])
# test_gradient(model, batches[1])


# -------------------------------------------------------------------------------
# Train models
# -------------------------------------------------------------------------------
n_epochs = 200

train!(model_simple, batches, opt_simple, n_epochs, train_set)
train!(model1, batches, opt1, n_epochs, train_set; buffer_size = batch_size)
train!(model2, batches, opt2, n_epochs, train_set; buffer_size = 10*batch_size)
train2!(model3, batches, opt3, n_epochs, train_set)

# -------------------------------------------------------------------------------
# Test models
# -------------------------------------------------------------------------------
x, y = test_set;
# x, y = train_set;

rec = [0.05; collect(0.1:0.1:0.9)]

df, plt = test_model([model_simple, model1, model2, model3], x, Bool.(vec(y)), rec);
