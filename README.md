# AccuracyAtTop.jl


 ## Instalation

To install this package use [Pkg REPL]([https://docs.julialang.org/en/v1/stdlib/Pkg/index.html](https://docs.julialang.org/en/v1/stdlib/Pkg/index.html)) and following command

```julia
 add https://github.com/VaclavMacha/AccuracyAtTop.jl
```

 ## Usage

This package provides a simple but powerful interface for solving many optimization problems with decision threshold
constraints. The package provides two functions that can be used as an objectives for the optimization

* `fnr(targets, scores, t, [surrogate = quadratic])`: computes the approximation of false-negative rate
* `fpr(targets, scores, t, [surrogate = quadratic])`: computes the approximation of false-positive rate

where `targets` is a vector of targets (true labels); `scores` is a vector of classification scores given by the used model; `t` is a decision threshold and `surrogate` is a function that is used as an approximation of the indicator function (indicator function returns `1` if its argument is true and `0` otherwise). The package provides two basic surrogate functions namely hinge loss and quadratic hinge loss

* `hinge(x, [ϑ = 1])`: hinge loss defined as ` max(0, 1 + ϑ * x)`
* `quadratic(x, [ϑ = 1])`: quadratic hinge loss defined as `max(0, 1 + ϑ * x)^2`

However, it is possible to define and use any other surrogate function. To define the decision threshold `t`, the package provides function `threshold(t_type, targets, scores)` where the first argument specifies the type of the threshold. There are four basic type of the decision thresholds

* `Maximum(; [samples = NegSamples])`: the threshold represents the maximum of `scores`
* `Minimum(; [samples = PosSamples])`: the threshold represents the maximum of `scores`
* `Kth(k; [samples = NegSamples, rev = true])`: the threshold represents `k`-th largest element of `scores` if `rev = true` and `k`-th smallest element otherwise
* `Quantile(τ; [samples = NegSamples, rev = true])`: the threshold represents `τ`-quantile of `scores` if `rev = false` and `(1 - τ)`-quantile otherwise

The keyword argument `samples` determines from which samples the threshold should be calculated. There are three options

* `AllSamples`: the threshold is computed from all classification scores
* `NegSamples`: the threshold is computed only from the scores corresponding to the negative samples
* `PosSamples`: the threshold is computed only from the scores corresponding to the positive samples

The package also provides four commonly used types of thresholds (in fact these are only outer constructors for `Quantile` threshold type)

* `TPRate(τ::Real)`: the threshold represents true-positive rate
* `TNRate(τ::Real)`: the threshold represents true-negative rate
* `FPRate(τ::Real)`: the threshold represents false-positive rate
* `FNRate(τ::Real)`: the threshold represents false-negative rate

The following example shows, how to minimize false-negative rate with a given constraint that false-positive rate is smaller or equal to `5%`

```julia
model = Chain(...)
t_type = FPRate(0.05)
surrogate = hinge

function loss(data, targets)
    scores = model(data)
    t = threshold(t_type, targets, scores)
    return fnr(target, scores, t, surrogate)
end
```

Further examples are given in the [examples](https://github.com/VaclavMacha/AccuracyAtTop.jl/tree/develop/examples) folder.
