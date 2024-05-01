


@testset "zeeman.jl" begin
    @testset "g factors" begin
        @test g_LS(0, 0, 0) == 1
        @test g_LS(1, 1, 0) == 1
        @test g_LS(1, 0, 1) == 2
        @test g_LS(1, 1, 1) == 1.5
        @test g_jj(0, 1, 1, 1, 1) == 1
        @test g_JK(0, 1, 1, 1, 1, 1) == 1
        @test typeof(g_LS(0, 0, 1)) == Float32
        @test typeof(g_LS(0., 0., 1.)) == Float64
        @test typeof(g_jj(0, 0, 1, 1, 1)) == Float32
        @test typeof(g_jj(0., 0., 1., 1., 1.)) == Float64
        @test typeof(g_JK(0, 0, 1, 1, 1, 0)) == Float32
        @test typeof(g_JK(0., 0., 1., 1., 1., 0.)) == Float64
        @test g_eff(0, 0, 1, 1) == 0
        @test g_eff(1, 1, 1, 1) == 1
        @test g_eff(2.5, 1, 2, 1) == g_eff(1, 2.5, 1, 2)
    end
    @testset "zeeman_strength" begin
        @test zeeman_strength(1, 1, -1, 0) ==  zeeman_strength(1, 1, 0, 1)
        @test zeeman_strength(1, 1, 1, 1) == 0.5
        @test zeeman_strength(2, 2, 1, 1) == 0.1
        @test zeeman_strength(1, 0, -1, 0) == zeeman_strength(0, 1, 0, -1)
        # normal zeeman: only one component of each, strength is one
        @test zeeman_strength(1, 0, 1, 0) == 1
        @test zeeman_strength(1, 0, 0, 0) == 1
        @test zeeman_strength(1, 0, -1, 0) == 1
        # two components: sum must add to one
        @test zeeman_strength(1, 1, 1, 0) + zeeman_strength(1, 1, 0, -1) == 1
        @test zeeman_strength(1, 1, 0, 1) + zeeman_strength(1, 1, -1, 0) == 1
        @test zeeman_strength(1, 1, 1, 1) + zeeman_strength(1, 1, -1, -1) == 1
        # normalisation of many components
        tmp_b = tmp_pi = tmp_r = 0
        J = 10
        for Ml in -J:J,  Mu in -J:J
            ΔM = Mu - Ml
            if ΔM == -1
                tmp_r += zeeman_strength(J, J, Ml, Mu)
            elseif ΔM == 0
                tmp_pi += zeeman_strength(J, J, Ml, Mu)
            elseif ΔM == 1
                tmp_b += zeeman_strength(J, J, Ml, Mu)
            end
        end
        @test tmp_b ≈ tmp_pi ≈ tmp_r ≈ 1
        # Test invalid transitions
        @test_throws ErrorException zeeman_strength(0, 0, 1, 0)
        @test_throws ErrorException zeeman_strength(3, 0, 1, 0)
        @test_throws ErrorException zeeman_strength(1, 1, 2, 0)
        @test_throws ErrorException zeeman_strength(1, 3, 0, 1)
        @test_throws ErrorException zeeman_strength(2, 0, 0, 0)
    end
    @testset "parse_label" begin
        @test Muspel.parse_label_LS("CA II 3P6 4S 2SE") == (0.5, 0)
        @test Muspel.parse_label_LS("CA II 3P6 3D 2DE 3") == (0.5, 2)
        @test Muspel.parse_label_LS("CA II 3P6 3D 2DE 5") == (0.5, 2)
        @test Muspel.parse_label_LS("CA II 3P6 4P 2PO 1") == (0.5, 1)
        @test Muspel.parse_label_LS("CA II 3P6 4P 2PO 3") == (0.5, 1)
        @test Muspel.parse_label_LS("CA III 3P6 1SE") == (0, 0)
        @test Muspel.parse_label_LS("FeI 3d6 4s 5s e5DE 2") == (2, 2)
        @test Muspel.parse_label_LS("FEI 3D6 4S 4P z7DE 3") == (3, 2)
        @test Muspel.parse_label_LS("O II 2P2(3PE) 3P 4PO") == (1.5, 1)
        @test Muspel.parse_label_LS("HE I 1S 5F 3FO 4") == (1, 3)
        @test Muspel.parse_label_LS("9Z") == (4, 20)
        @test_throws ErrorException Muspel.parse_label_LS("HE I")
        @test_throws ErrorException Muspel.parse_label_LS("E asdPFS")
    end
    @testset "zeeman_components" begin
        # normal Zeeman
        @test get_zeeman_components(0, 1, 1, 0, 1, 0) == ([1], [-1], [1], [0], [1], [1])
        @test get_zeeman_components(0, 1, 0, 0, 1, 1) == ([1], [-1], [1], [0], [1], [1])
        # anomalous Zeeman
        @test get_zeeman_components(1, 1, 1, 1, 1, 1) == (
            [0.5, 0.5], [-1.5, -1.5], [0.5, 0.5], [0, 0], [0.5, 0.5], [1.5, 1.5]
        )
        # consistency checks
        conf1 = [3.5, 3, 4, 2.5, 4, 4]
        conf2 = [9.5, 10, 16, 10.5, 10, 16]
        for conf in [conf1, conf2]
            σr_S, σr_Δ, π_S, π_Δ, σb_S, σb_Δ = get_zeeman_components(conf...)
            @test length(σr_S) == length(σb_S)
            @test maximum(π_Δ) ≈ -minimum(π_Δ)
            @test sum(σr_S) ≈ 1
            @test sum(π_S) ≈ 1
            @test sum(σb_S) ≈ 1
        end
    end

end
