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
       Buffer, NoBuffer, ScoresDelay, LastThreshold,
       Model,
       BaseLineModel, BaseLine,
       ThresholdModel, TopPush, TopPushK, PatMat, PatMatNP, PrecAtRec,
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
abstract type Buffer; end
abstract type BaseLineModel <: Model end
abstract type ThresholdModel{Buffer} <: Model end
abstract type FNRModel{Buffer} <: ThresholdModel{Buffer} end
abstract type FPRModel{Buffer} <: ThresholdModel{Buffer} end

include("utilities.jl")
include("gradients.jl")
include("baselinemodels.jl")
include("fnrmodels.jl")
include("fprmodels.jl")
include("lossfunctions.jl")

end # module
