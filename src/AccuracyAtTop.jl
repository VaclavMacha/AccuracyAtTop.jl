module AccuracyAtTop

using Reexport
using LinearAlgebra, Statistics, Parameters, BSON, ProgressMeter, TimerOutputs

@reexport using Flux

import CuArrays 
import Flux: cpu, gpu, Params, params, gradient, sigmoid, binarycrossentropy
import Flux.Optimise: train!, update!, runall, StopException
import Zygote: hook, Grads
import Random: randperm, shuffle
import CategoricalArrays: recode
import MLBase: sample
import Base: show

export loss, scores, threshold, train!, @runepochs, @tracktime,
       make_minibatch, reshape_data,
       Hinge, Quadratic, BinCrossEntropy, FNR,
       allow_cuda, status_cuda,
       ScoreBuffer,
       Model, Simple,
       ThresholdModel, TopPush, TopPushK, PatMat, PatMatNP,
       test_gradient


########
# Cuda #
########
function allow_cuda(flag::Bool)
    AccuracyAtTop.Flux.use_cuda[] = flag
    status_cuda()
    return
end


function status_cuda()
    if AccuracyAtTop.Flux.use_cuda[]
        @info "Cuda is allowed"
    else
        @info "Cuda is not allowed"
    end
    return
end


##################
# Abstract types #
##################
abstract type Model; end
abstract type SimpleModel <: Model; end
abstract type ThresholdModel <: Model; end

include("utilities.jl")
include("scoresbuffer.jl")
include("models.jl")
include("lossfunctions.jl")

end # module
