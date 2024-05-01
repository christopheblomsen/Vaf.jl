using AtomicData

@testset "intensity.jl" begin
    FALC_RH_file = joinpath(@__DIR__, "..", "data", "atmospheres", "FALC.hdf5")
    H_ATOM_file = "test_atoms/H_test.yaml"
    atm = read_atmos_rh(FALC_RH_file)[1, 1]
    @testset "calc_line_1D!" begin
        h_atom = read_atom(H_ATOM_file)
        my_line = h_atom.lines[5]  # Halpha for example
        # Populations from an RH run with the FALC atmosphere
        n_l = [1.16279f5, 1.35481f5, 1.61520f5, 2.12806f5, 3.03263f5, 3.80475f5,
               5.04064f5, 7.26205f5, 1.18963f6, 1.64719f6, 2.40792f6, 3.87856f6,
               7.30773f6, 1.61016f7, 4.53617f7, 1.80606f8, 1.06021f9, 3.95242f9,
               8.26403f9, 1.4043f10, 1.8359f10, 2.0588f10, 2.2119f10, 2.2901f10,
               2.3004f10, 2.2431f10, 2.1327f10, 2.0180f10, 1.8993f10, 1.8701f10,
               1.9338f10, 2.0760f10, 2.1978f10, 2.2854f10, 2.4134f10, 2.6021f10,
               2.9463f10, 3.5499f10, 4.5300f10, 5.8483f10, 8.1597f10, 1.1706f11,
               1.5797f11, 2.0822f11, 2.6417f11, 2.9828f11, 2.9801f11, 2.7552f11,
               2.4179f11, 2.0283f11, 1.6012f11, 1.0986f11, 6.8648f10, 4.9831f10,
               4.3881f10, 5.1496f10, 8.0354f10, 1.8207f11, 4.3557f11, 1.0726f12,
               2.6948f12, 6.8632f12, 1.1562f13, 2.0921f13, 4.2045f13, 8.8683f13,
               2.0155f14, 5.0060f14, 1.0288f15, 2.0863f15, 3.5157f15, 6.0941f15,
               1.0787f16, 2.1219f16, 4.3268f16, 8.4872f16, 1.5797f17, 2.8450f17,
               4.8905f17, 8.0779f17, 1.2209f18, 1.7534f18]
        n_u = [5.90943f4, 6.76884f4, 7.90113f4, 1.00357f5, 1.35520f5, 1.63550f5,
               2.05444f5, 2.73754f5, 3.96664f5, 5.00951f5, 6.50285f5, 8.87271f5,
               1.30742f6, 2.05471f6, 3.73978f6, 8.22859f6, 2.24261f7, 5.01449f7,
               8.01383f7, 1.13485f8, 1.35617f8, 1.46254f8, 1.53010f8, 1.55528f8,
               1.53999f8, 1.48802f8, 1.40722f8, 1.32848f8, 1.24931f8, 1.24270f8,
               1.30920f8, 1.45870f8, 1.61694f8, 1.75948f8, 1.94269f8, 2.17835f8,
               2.59706f8, 3.35659f8, 4.67752f8, 6.60090f8, 1.03705f9, 1.70055f9,
               2.60173f9, 3.90411f9, 5.56120f9, 6.93626f9, 7.36287f9, 7.09179f9,
               6.39042f9, 5.46844f9, 4.37954f9, 3.03579f9, 1.90705f9, 1.38767f9,
               1.22380f9, 1.43808f9, 2.24854f9, 5.12172f9, 1.2405f10, 3.1392f10,
               8.3078f10, 2.2850f11, 4.0413f11, 7.7443f11, 1.6682f12, 3.8039f12,
               9.4887f12, 2.6459f13, 6.0266f13, 1.3630f14, 2.5012f14, 4.7587f14,
               9.3023f14, 2.0650f15, 4.7931f15, 1.0635f16, 2.2174f16, 4.4491f16,
               8.4517f16, 1.5316f17, 2.4973f17, 3.8322f17]
        σ_itp = get_σ_itp(atm, my_line.λ0, empty([""]))  # no background atoms
        a = LinRange(1f-4, 1f1, 20000)
        v = LinRange(0f2, 5f2, 2500)
        voigt_itp = create_voigt_itp(a, v)
        buf = RTBuffer(atm.nz, my_line.nλ, Float32)

        calc_line_1D!(my_line, buf, atm, n_u, n_l, σ_itp, voigt_itp)

        # Consistency tests
        @test buf.intensity[end] == buf.int_tmp[1]
        S_Planck = blackbody_λ.(my_line.λ0, atm.temperature)
        @test isapprox(S_Planck, buf.source_function, rtol=1e-2)  # S = B for continuum
        @test isapprox(buf.j_c ./ buf.α_c, buf.source_function, rtol=1e-2)  # S = S_c
        @test all(buf.source_function .> 0)
        @test all(buf.α_c .> 0)
        @test all(buf.j_c .> 0)
        @test all(isfinite.(buf.source_function))
        @test all(isfinite.(buf.intensity))
        @test all(isfinite.(buf.int_tmp))
        @test all(isfinite.(buf.ΔλD))
        @test all(isfinite.(buf.γ))
        @test all(buf.ΔλD .< 0.1)
        @test all(buf.γ .>= my_line.γ.natural)
        # Tests for this specific line profile
        @test argmin(buf.intensity) == my_line.nλ ÷ 2  # no velocity, min at line centre
        @test buf.intensity[1] > buf.intensity[my_line.nλ ÷ 2 + 1]  # absorption line
        @test isapprox(buf.intensity[my_line.nλ ÷ 2 + 1], 4.673165, rtol=1e-3)  # line core
        @test isapprox(buf.intensity[1], 27.279392, rtol=1e-3)   # continuum intensity
        @test isapprox(buf.intensity[19], 15.642044, rtol=1e-3)  # line wing
        @test isapprox(  # symmetric line profile
            buf.intensity[1:my_line.nλ ÷ 2],
            reverse(buf.intensity[my_line.nλ ÷ 2+1:end]),
            rtol=1e-3
        )
        @test all(diff(buf.intensity[1:my_line.nλ ÷ 2]) .< 0)        # smooth decrease
        @test all(diff(buf.intensity[my_line.nλ ÷ 2 + 2:end]) .> 0)  # smooth increase

        # now calculate extreme cases
        calc_line_1D!(my_line, buf, atm, n_u, n_u, σ_itp, voigt_itp) # emission line
        @test buf.intensity[1] < buf.intensity[my_line.nλ ÷ 2 + 1]

        calc_line_1D!(my_line, buf, atm, n_u * 0, n_u * 0, σ_itp, voigt_itp)  # no line
        @test all(buf.intensity .== buf.intensity[1])
        @test isapprox(buf.intensity[1], 28.538631, rtol=1e-3)
    end
    @testset "calc_τ_cont!" begin
        σ_itp = get_σ_itp(atm, 500f0, empty([""]))
        tau1 = Vector{Float32}(undef, atm.nz)
        tau2 = Vector{Float32}(undef, atm.nz)
        calc_τ_cont!(atm, tau1, σ_itp)
        @test tau1[1] == 0
        # Testing against implementation
        @test tau1[end] ≈ 19.67434
        @test minimum(log10.(tau1[2:end])) > -9
        # Testing with reversed atmosphere, final integration should be the same
        rev_atm = deepcopy(atm)
        rev_atm.temperature[:] = reverse(atm.temperature)
        rev_atm.electron_density[:] = reverse(atm.electron_density)
        rev_atm.proton_density[:] = reverse(atm.proton_density)
        rev_atm.hydrogen1_density[:] = reverse(atm.hydrogen1_density)
        rev_atm.z[:] = reverse(atm.z)
        calc_τ_cont!(rev_atm, tau2, σ_itp)
        @test tau1[end] ≈ tau2[end]
    end
end
