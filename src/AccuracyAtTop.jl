module AccuracyAtTop

using Reexport, LinearAlgebra, DataStructures, Zygote, ProgressMeter

@reexport using Flux

import Flux: @epochs, @progress, Params, gradient,
             binarycrossentropy, sigmoid
import Flux.Optimise: train!, update!, runall, StopException
import Zygote: hook, Grads

export loss, scores, train!, train2!, hinge, quadratic, test_gradient
export Model, Simple,
       ThresholdModel, TopPush, TopPushK

include("scoresbuffer.jl")
include("models.jl")
include("train.jl")
include("test_gradient.jl")

end # module
