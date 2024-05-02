@testset "formal_solvers.jl" begin
    @testset "Weights" begin
        @test all(Muspel._w2(60.) .== (1, 1))
        @test all(Muspel._w2(1.) .≈ (1-exp(-1), 1 - 2*exp(-1)))
        @test all(Muspel._w2(1f-6) .≈ (9.999995f-7, 4.9999967f-13))
    end
    @testset "Piecewise" begin
        # Constant source function
        z = collect(LinRange(1, 1e6, 20))
        alpha = ones(20) * 1e-20
        S = ones(20)
        @test piecewise_1D_linear(z, alpha, S) ≈ S
        @test piecewise_1D_nn(z, alpha, S) ≈ S
        @test begin
            result = similar(S)
            piecewise_1D_linear!(z, alpha, S, result)
            result ≈ S
        end
        alpha = ones(20)
        @test piecewise_1D_linear(z, alpha, S) ≈ S
        @test piecewise_1D_linear(z, alpha, S;
                                  initial_condition=:zero)[[1, end]] ≈ [S[1], S[1]*0]
        @test begin
            result = similar(S)
            piecewise_1D_linear!(z, alpha, S, result)
            result ≈ S
        end
        @test begin
            result = similar(S)
            piecewise_1D_linear!(z, alpha, S, result; initial_condition=:zero)
            result[[1, end]] ≈ [S[1], S[1]*0]
        end
        @test piecewise_1D_nn(z, alpha, S) ≈ S
        @test piecewise_1D_nn(z, alpha, S;
                              initial_condition=:zero)[[1, end]] ≈ [S[1], S[1]*0]
        # Linear extinction and source function, test reversibility
        alpha = collect(LinRange(1e-3, 1e-5, 20))
        S = collect(LinRange(1, 100, 20))
        @test (piecewise_1D_linear(z, reverse(alpha), reverse(S); to_end=true) ≈
               reverse(piecewise_1D_linear(z, alpha, S)))
        @test (piecewise_1D_nn(z, reverse(alpha), reverse(S); to_end=true) ≈
               reverse(piecewise_1D_nn(z, alpha, S)))
        @test begin
            result_i = similar(S)
            piecewise_1D_linear!(z, reverse(alpha), reverse(S), result_i; to_end=true)
            result = similar(S)
            piecewise_1D_linear!(z, alpha, S, result)
            result_i ≈ reverse(result)
        end
        # Exceptions
        result = similar(S)
        @test_throws ErrorException piecewise_1D_linear!(z, alpha, S, result; initial_condition=:aaa)
        @test_throws ErrorException piecewise_1D_linear(z, alpha, S; initial_condition=:aaa)
        @test_throws ErrorException piecewise_1D_nn(z, alpha, S; initial_condition=:aaa)
    end
    @testset "Feautrier" begin
        z = collect(LinRange(2e6, -1e5, 20))
        alpha = 1e-5 * ones(20)
        S = zeros(20)
        # Simple tests
        @test feautrier(z, alpha, S) ≈ S
        S = 100 * ones(20)
        # Against implementation
        @test feautrier(z, alpha, S)[1] * 2 ≈ 106.65292755967045
        # Test reversibility
        @test feautrier(z, alpha, S)[1] ≈ feautrier(z, reverse(alpha), reverse(S))[end]
    end
end
