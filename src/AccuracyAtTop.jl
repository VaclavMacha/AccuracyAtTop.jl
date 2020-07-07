module AccuracyAtTop

using LinearAlgebra, Statistics

import Zygote: @adjoint, @nograd
import Flux: sigmoid, binarycrossentropy


include("models.jl")
include("utilities.jl")


export loss_baseline, loss_fnr, loss_fpr, DeepTopPush, DeepTopPushK,
       PatMat, PatMatNP, RecAtK, PrecAtRec, hinge, quadratic

end # module
