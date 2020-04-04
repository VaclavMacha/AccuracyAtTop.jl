import AccuracyAtTop: threshold_gradient, update!, Buffer


# -------------------------------------------------------------------------------
# NoBuffer
# -------------------------------------------------------------------------------
buffer = NoBuffer()

@testset "NoBuffer" begin
    @test typeof(buffer) <: NoBuffer
    @test NoBuffer <: Buffer
    @test fieldnames(NoBuffer) == ()
end


# -------------------------------------------------------------------------------
# Scores Delay
# -------------------------------------------------------------------------------
T      = Float32
buffer = ScoresDelay(T, 10)

target = rand(1, 6) .>= 0.5
scores = rand(T, 1, 6)

@testset "ScoresDelay" begin
    @test typeof(buffer) <: ScoresDelay
    @test ScoresDelay <: Buffer
    @test fieldnames(ScoresDelay) == (:T, :buffer_size, :scores, :target, :batch_inds, :sample_inds)

    update!(buffer, target, scores, 1)
    @test buffer.scores      == vec(scores)
    @test buffer.target      == vec(target)
    @test buffer.batch_inds  == fill(1, 6)
    @test buffer.sample_inds == collect(1:6)

    update!(buffer, target, scores, 2)
    @test buffer.scores      == vcat(vec(scores)[3:end], vec(scores))
    @test buffer.target      == vcat(vec(target)[3:end], vec(target))
    @test buffer.batch_inds  == vcat(fill(1, 4), fill(2, 6))
    @test buffer.sample_inds == vcat(3:6, 1:6)
end


# -------------------------------------------------------------------------------
# Last thresahold
# -------------------------------------------------------------------------------
buffer = LastThreshold(1, 11)


@testset "LastThreshold" begin
    @test typeof(buffer) <: LastThreshold
    @test LastThreshold <: Buffer
    @test fieldnames(LastThreshold) == (:batch_ind, :sample_ind)

    @test buffer.batch_ind  == 1
    @test buffer.sample_ind == 11

    update!(buffer, 2, 13)
    @test buffer.batch_ind  == 2
    @test buffer.sample_ind == 13
end