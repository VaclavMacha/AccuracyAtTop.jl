using AccuracyAtTop: find_threshold, find_kth, find_quantile

s = collect(1:10)
y = s .>= 6

@testset "auxiliary threshold functions" begin
    @testset "maximum" begin
        @test find_threshold(All, findmax, y, s) == (10, 10)
        @test find_threshold(Neg, findmax, y, s) == (5, 5)
        @test find_threshold(Pos, findmax, y, s) == (10, 10)

        @test find_threshold(Maximum(All), y, s) == (10, 10)
        @test find_threshold(Maximum(Neg), y, s) == (5, 5)
        @test find_threshold(Maximum(Pos), y, s) == (10, 10)
    end

    @testset "minimum" begin
        @test find_threshold(All, findmin, y, s) == (1, 1)
        @test find_threshold(Neg, findmin, y, s) == (1, 1)
        @test find_threshold(Pos, findmin, y, s) == (6, 6)

        @test find_threshold(Minimum(All), y, s) == (1, 1)
        @test find_threshold(Minimum(Neg), y, s) == (1, 1)
        @test find_threshold(Minimum(Pos), y, s) == (6, 6)
    end

    @testset "$(k) all samples" for k in 1:10
        @test find_kth(s, k, false) == (k, k)
        @test find_kth(s, k, true) == (11 - k, 11 - k)

        @test find_threshold(All, find_kth, y, s, k, false) == (k, k)
        @test find_threshold(All, find_kth, y, s, k, true) == (11 - k, 11 - k)

        @test find_threshold(Kth(k, All; rev = false), y, s) == (k, k)
        @test find_threshold(Kth(k, All; rev = true), y, s) == (11 - k, 11 - k)
    end

    @testset "$(k) pos/neg samples" for k in 1:5
        @test find_threshold(Neg, find_kth, y, s, k, false) == (k, k)
        @test find_threshold(Neg, find_kth, y, s, k, true) == (6 - k, 6 - k)
        @test find_threshold(Kth(k, Neg; rev = false), y, s) == (k, k)
        @test find_threshold(Kth(k, Neg; rev = true), y, s) == (6 - k, 6 - k)

        @test find_threshold(Pos, find_kth, y, s, k, false) == (5 + k, 5 + k)
        @test find_threshold(Pos, find_kth, y, s, k, true) == (11 - k, 11 - k)
        @test find_threshold(Kth(k, Pos; rev = false), y, s) == (5 + k, 5 + k)
        @test find_threshold(Kth(k, Pos; rev = true), y, s) == (11 - k, 11 - k)
    end

    @testset "$(τ)-quantile all samples" for (k, τ) in zip(vcat(1, 1:10), 0:0.1:1)
        @test find_quantile(s, τ, false) == (k, k)
        @test find_threshold(All, find_quantile, y, s, τ, false) == (k, k)
        @test find_threshold(Quantile(τ, All; rev = false), y, s) == (k, k)
    end

    @testset "$(τ)-quantile all samples reversed" for (k, τ) in zip(vcat(10:-1:1, 1), 0:0.1:1)
        @test find_quantile(s, τ, true) == (k, k)
        @test find_threshold(All, find_quantile, y, s, τ, true) == (k, k)
        @test find_threshold(Quantile(τ, All; rev = true), y, s) == (k, k)
    end

    @testset "$(τ)-quantile pos/neg samples" for (k, τ) in zip(vcat(1, 1:5), 0:0.2:1)
        @test find_threshold(Neg, find_quantile, y, s, τ, false) == (k, k)
        @test find_threshold(Quantile(τ, Neg; rev = false), y, s) == (k, k)

        @test find_threshold(Pos, find_quantile, y, s, τ, false) == (k + 5, k + 5)
        @test find_threshold(Quantile(τ, Pos; rev = false), y, s) == (k + 5, k + 5)
    end

    @testset "$(τ)-quantile pos/neg samples reversed" for (k, τ) in zip(vcat(5:-1:1, 1), 0:0.2:1)
        @test find_threshold(Neg, find_quantile, y, s, τ, true) == (k, k)
        @test find_threshold(Quantile(τ, Neg; rev = true), y, s) == (k, k)

        @test find_threshold(Pos, find_quantile, y, s, τ, true) == (k + 5, k + 5)
        @test find_threshold(Quantile(τ, Pos; rev = true), y, s) == (k + 5, k + 5)
    end
end

s = rand(4, 100)
y = rand(Bool, 4, 100)

function test_indices(tp, y, s)
    t, ind = find_threshold(tp, y, s)

    @testset "$tp for $i-th row" for i in 1:size(s, 1)
        @test t[i] == s[i, ind[i]]
    end
end

@testset "thresholds rrule" begin
    @testset "$tp" for tp in Maximum.([All, Neg, Pos])
        test_rrule(threshold, tp ⊢ DoesNotExist(), y ⊢ DoesNotExist(), s)
        test_indices(tp, y, s)
    end

    @testset "$tp" for tp in Minimum.([All, Neg, Pos])
        test_rrule(threshold, tp ⊢ DoesNotExist(), y ⊢ DoesNotExist(), s)
        test_indices(tp, y, s)
    end

    @testset "$K" for K in [1, 2, 5, 10, 15]
        @testset "$rev" for rev in [true, false]
            @testset "$tp" for tp in Kth.(K, [All, Neg, Pos]; rev)
                test_rrule(threshold, tp ⊢ DoesNotExist(), y ⊢ DoesNotExist(), s)
                test_indices(tp, y, s)
            end
        end
    end

    @testset "$τ" for τ in [0.05, 0.1, 0.15, 0.2, 0.5]
        @testset "$rev" for rev in [true, false]
            @testset "$tp" for tp in Quantile.(τ, [All, Neg, Pos]; rev)
                test_rrule(threshold, tp ⊢ DoesNotExist(), y ⊢ DoesNotExist(), s)
                test_indices(tp, y, s)
            end
        end
    end

    @testset "$τ" for τ in [0.05, 0.1, 0.15, 0.2, 0.5]
        sampler() = τ
        @testset "$rev" for rev in [true, false]
            @testset "$tp" for tp in SampledQuantile.(sampler, [All, Neg, Pos]; rev)
                test_rrule(threshold, tp ⊢ DoesNotExist(), y ⊢ DoesNotExist(), s)
                test_indices(tp, y, s)
            end
        end
    end
end
