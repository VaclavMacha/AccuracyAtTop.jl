import AccuracyAtTop: clip, ispos, isneg, find_negatives, find_positives
import AccuracyAtTop: scores_max, scores_kth, scores_quantile
import AccuracyAtTop: Surrogate

# -------------------------------------------------------------------------------
# Cuda
# -------------------------------------------------------------------------------
@testset "Cuda" begin
    tmp = AccuracyAtTop.Flux.use_cuda[]
    allow_cuda(true)
    @test AccuracyAtTop.Flux.use_cuda[] == true
    allow_cuda(false)
    @test AccuracyAtTop.Flux.use_cuda[] == false
    allow_cuda(tmp)
end


# -------------------------------------------------------------------------------
# auxiliary threshold functions
# -------------------------------------------------------------------------------
scores = reshape(collect(1:10), 1, :)
target = scores .>= 6

@testset "auxiliary threshold functions" begin
    @test clip(2,1,3) == 2
    @test clip(0,1,3) == 1
    @test clip(4,2,3) == 3

    @test isneg.(target) == .~target
    @test ispos.(target) == target

    @test find_negatives(target) == [1,2,3,4,5]
    @test find_positives(target) == [6,7,8,9,10]

    @test scores_max(scores)                         == (10, 10)
    @test scores_max(scores, find_negatives(target)) == (5, 5)
    @test scores_max(scores, find_positives(target)) == (10, 10)

    @test scores_kth(scores, 2)                                     == (2, 2)
    @test scores_kth(scores, 2, find_negatives(target))             == (2, 2)
    @test scores_kth(scores, 2, find_positives(target))             == (7, 7)
    @test scores_kth(scores, 2; rev = true)                         == (9, 9)
    @test scores_kth(scores, 2, find_negatives(target); rev = true) == (4, 4)
    @test scores_kth(scores, 2, find_positives(target); rev = true) == (9, 9)

    @test scores_quantile(scores, 0.4)                         == (4, 4)
    @test scores_quantile(scores, 0.4, find_negatives(target)) == (2, 2)
    @test scores_quantile(scores, 0.4, find_positives(target)) == (7, 7)
end


# -------------------------------------------------------------------------------
# Surrogate functions
# -------------------------------------------------------------------------------
ϑ = rand()

@testset "auxiliary threshold functions" begin
    l1 = Hinge(ϑ)
    @testset "hinge loss" begin
        @test typeof(l1) <: Surrogate
        @test l1.ϑ == ϑ
        @test l1.value.([-1/ϑ - 1, 0, -1/ϑ + 1])    ≈ [0, 1, ϑ]
        @test l1.gradient.([-1/ϑ - 1, 0, -1/ϑ + 1]) ≈ [0, ϑ, ϑ]
    end


    l2 = Quadratic(ϑ)
    @testset "quadratic loss" begin
        @test typeof(l2) <: Surrogate
        @test l2.ϑ == ϑ
        @test l2.value.([-1/ϑ - 1, 0, -1/ϑ + 1])    ≈ [0, 1, ϑ^2]
        @test l2.gradient.([-1/ϑ - 1, 0, -1/ϑ + 1]) ≈ [0, 2*ϑ, 2*ϑ^2]
    end
end