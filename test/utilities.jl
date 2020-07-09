import AccuracyAtTop: clip, ispos, isneg, find_negatives, find_positives
import AccuracyAtTop: scores_max, scores_kth, scores_quantile
import AccuracyAtTop: hinge, quadratic


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

@testset "surrrogate functions" begin
    @test hinge.([-1/ϑ - 1, 0, -1/ϑ + 1], ϑ)    ≈ [0, 1, ϑ]
    @test quadratic.([-1/ϑ - 1, 0, -1/ϑ + 1], ϑ)    ≈ [0, 1, ϑ^2]
end