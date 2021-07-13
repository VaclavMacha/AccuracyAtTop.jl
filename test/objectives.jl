using AccuracyAtTop: hinge, quadratic

# Surrogate functions
ϑ = rand()

@testset "surrrogate functions" begin
    @test hinge.([-1/ϑ - 1, 0, -1/ϑ + 1], ϑ) ≈ [0, 1, ϑ]
    @test quadratic.([-1/ϑ - 1, 0, -1/ϑ + 1], ϑ) ≈ [0, 1, ϑ^2]
end
