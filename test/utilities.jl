using AccuracyAtTop: find_negatives, find_positives
using AccuracyAtTop: find_kth, find_quantile, find_score
using AccuracyAtTop: hinge, quadratic

scores = collect(1:10)
targets = scores .>= 6

@testset "auxiliary threshold functions" begin
    @test find_negatives(targets) == [1, 2, 3, 4, 5]
    @test find_positives(targets) == [6, 7, 8, 9, 10]

    @testset "maximum" begin
        @test find_score(AllSamples, findmax, targets, scores) == (10, 10)
        @test find_score(NegSamples, findmax, targets, scores) == (5, 5)
        @test find_score(PosSamples, findmax, targets, scores) == (10, 10)
    end

    @testset "minimum" begin
        @test find_score(AllSamples, findmin, targets, scores) == (1, 1)
        @test find_score(NegSamples, findmin, targets, scores) == (1, 1)
        @test find_score(PosSamples, findmin, targets, scores) == (6, 6)
    end

    @testset "$(k) all samples" for k in 1:10
        @test find_kth(scores, k) == (k, k)
        @test find_score(AllSamples, find_kth, targets, scores, k) == (k, k)
        @test find_kth(scores, k; rev = true) == (11 - k, 11 - k)
        @test find_score(AllSamples, find_kth, targets, scores, k; rev = true) == (11 - k, 11 - k)
    end

    @testset "$(k) pos/neg samples" for k in 1:5
        @test find_score(NegSamples, find_kth, targets, scores, k) == (k, k)
        @test find_score(NegSamples, find_kth, targets, scores, k; rev = true) == (6 - k, 6 - k)
        @test find_score(PosSamples, find_kth, targets, scores, k) == (5 + k, 5 + k)
        @test find_score(PosSamples, find_kth, targets, scores, k; rev = true) == (11 - k, 11 - k)
    end

    @testset "$(τ)-quantile all samples" for (k, τ) in zip(vcat(1, 1:10), 0:0.1:1)
        @test find_quantile(scores, τ) == (k, k)
        @test find_score(AllSamples, find_quantile, targets, scores, τ) == (k, k)
    end

    @testset "$(τ)-quantile all samples" for (k, τ) in zip(vcat(10:-1:1, 1), 0:0.1:1)
        @test find_quantile(scores, τ; rev = true) == (k, k)
        @test find_score(AllSamples, find_quantile, targets, scores, τ; rev = true) == (k, k)
    end

    @testset "$(τ)-quantile pos/neg samples" for (k, τ) in zip(vcat(1, 1:5), 0:0.2:1)
        @test find_score(NegSamples, find_quantile, targets, scores, τ) == (k, k)
        @test find_score(PosSamples, find_quantile, targets, scores, τ) == (k + 5, k + 5)
    end

    @testset "find $(τ)-quantile all samples" for (k, τ) in zip(vcat(5:-1:1, 1), 0:0.2:1)
        @test find_score(NegSamples, find_quantile, targets, scores, τ; rev = true) == (k, k)
        @test find_score(PosSamples, find_quantile, targets, scores, τ; rev = true) == (k + 5, k + 5)
    end
end

# Surrogate functions
ϑ = rand()

@testset "surrrogate functions" begin
    @test hinge.([-1/ϑ - 1, 0, -1/ϑ + 1], ϑ) ≈ [0, 1, ϑ]
    @test quadratic.([-1/ϑ - 1, 0, -1/ϑ + 1], ϑ) ≈ [0, 1, ϑ^2]
end
