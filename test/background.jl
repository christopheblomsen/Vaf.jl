using AtomicData
using Interpolations
using Unitful
import Muspel: create_σ_itp_LTE, create_σ_itp_NLTE, get_atoms_bf_interpolant, σH_continuum, σH_atoms_bf

@testset "background.jl" begin
    abund = get_solar_abundances()
    He = read_atom("test_atoms/He_test.yaml")
    H = read_atom("test_atoms/H_test.yaml")
    H_empty = read_atom("test_atoms/H_test_empty.yaml")
    atoms = Vector{AtomicModel}([He, H, H_empty])
    atoms_empty = Vector{AtomicModel}(undef, 0)
    σ_atom_itp = get_atoms_bf_interpolant(atoms)
    σ_atom_itp_empty = get_atoms_bf_interpolant(atoms_empty)
    λ = [92., 500., 850., 1600,  2000.]
    log_ne = 15.0:0.05:20.0
    log_temp = 3.0:0.015:6.0
    itp_lte = create_σ_itp_LTE(λ[1], log_temp, log_ne, H, atoms, σ_atom_itp)
    itp_nlte = create_σ_itp_NLTE(λ[1], log_temp, log_ne, atoms, σ_atom_itp)
    itp_nlte_empty = create_σ_itp_NLTE(λ[1], log_temp, log_ne, atoms_empty, σ_atom_itp_empty)
    temp = [2050.0, 5770.0, 10100.0]
    n = [1.0e15, 7.1e17, 1.0e20]

    @testset "σ_itp" begin
        @test isa(itp_lte, ExtinctionItpLTE{Float64})
        @test isa(itp_nlte, ExtinctionItpNLTE{Float64})
        @test itp_nlte_empty.σ_atoms.(log_temp, 19) ≈ zeros(Float64, length(log_temp))
        @test_throws MethodError create_σ_itp_LTE(λ[2], [3.1, 3.3], log_ne, H, atoms, σ_atom_itp)
        @test_throws MethodError create_σ_itp_LTE(λ[2], log_temp, [15, 19], H, atoms, σ_atom_itp)
        @test_throws MethodError create_σ_itp_NLTE(λ[2], [3.1, 3.3], log_ne, atoms, σ_atom_itp)
        @test_throws MethodError create_σ_itp_NLTE(λ[2], log_temp, [15, 19], atoms, σ_atom_itp)
        log_ne2 = 15.0:0.025:20.0 # make same number of elements as log_temp
        itp_nlte2 = create_σ_itp_NLTE(λ[2], log_temp, log_ne2, atoms, σ_atom_itp)
        atoms_list = [
            "test_atoms/He_test.yaml",
            "test_atoms/H_test.yaml",
            "test_atoms/H_test_empty.yaml"
        ]
        z_tmp = [1e5, 0., -1e5]
        atmos_test = Atmosphere1D(1, 1, length(log_temp), z_tmp, 10 .^ log_temp,
                                  [0., 0., 0.], 10 .^log_ne2, n, n)
        itp_test = get_σ_itp(atmos_test, 500., atoms_list; npts=101)
        @test isa(itp_test, ExtinctionItpNLTE{Float64})
        @test isapprox(
            α_cont(itp_nlte2, 5000., 1e20, 1e20, 1e20),
            α_cont(itp_test, 5000., 1e20, 1e20, 1e20),
            rtol=1e-3
        )
        @test_throws ErrorException get_σ_itp(atmos_test, 500., ["nofile.yaml"])
    end

    @testset "α_cont" begin
        for ni in n
            ion_frac = [saha_boltzmann(H, t, ni, 1.)[end] for t in temp]
            nHI = ni .* (1 .- ion_frac)
            nHII = ni .* ion_frac
            # Check if LTE and NLTE give same result, assuming Saha for hydrogen
            @test isapprox(
                α_cont.(Ref(itp_lte), temp, ni, ni),
                α_cont.(Ref(itp_nlte), temp, ni, nHI, nHII),
                rtol=1e-5,
            )
            # Check if interpolant with no atoms gives same result as α_cont_no_itp
            @test isapprox(
                α_cont_no_itp.(λ[1], temp, ni, nHI, nHII),
                α_cont.(Ref(itp_nlte_empty), temp, ni, nHI, nHII),
                rtol=1e-5,
            )
        end
        # Some checks against implementation
        prev = [3.166260014050373e-9   1.2390038437211058e-10  2.8355199724497864e-10
                2.3365757985473882e-8  4.1969276633548743e-10  1.4792957612236907e-9
                3.19501480342208e-8    5.232736686830469e-10   2.039376943876063e-9
                1.1834781970475661e-9  2.1924407940343387e-10  5.4690898709933755e-9
                2.0918382712344928e-10 2.886350125152959e-10   9.258069388938863e-9]
        for (i, λi) in enumerate(λ)
            itp = create_σ_itp_LTE(λi, log_temp, log_ne, H, atoms, σ_atom_itp)
            @test α_cont.(Ref(itp), temp, 1e18, 1e20) ≈ prev[i, :]
        end
    end

    @testset "α_cont_no_itp" begin
        # Against previous implementation:
        prev = [1.052784352703514, 3.5727512640936836, 4.628599312906391,
               1.9725067321692635, 2.5417412522755836] * 1e-8
        @test all(α_cont_no_itp.(λ, 6e3, 1e20, 1e20, 4e15) .≈ prev)
        # Consistency check, Hmin maximum extinction
        @test argmax(α_cont_no_itp.(λ, 6e3, 1e20, 1.0e20, 4e15)) == 3
    end

    @testset "σH" begin
        # Against previous implementation:
        prev = [0.38752066260764906, 2.9073809748885355, 3.963188644121838,
                1.3072036044702444, 1.8764158215614224] * 1e-28
        @test all(σH_continuum.(λ, 6e3, 1e20, 4e-5) .≈ prev)
        @test argmax(σH_continuum.(λ, 6e3, 1e20, 4e-5)) == 3
        @test σH_atoms_bf(σ_atom_itp_empty, atoms_empty, 500., 6000., 1e20) == 0.0
        # Check that cross section matches data from atom with no stimulated emission
        Ly_cont = atoms[2].continua[1]  # Lyman continuum
        @test σH_atoms_bf(σ_atom_itp, atoms, Ly_cont.λ[end], 0., 1e20) ≈ Ly_cont.σ[end]
        @test σH_atoms_bf(σ_atom_itp, atoms, λ[1], 6e3, 1e20) ≈ 2.8831654074700022e-31
        @test σH_atoms_bf(σ_atom_itp, atoms, λ[end], 6e3, 1e20) ≈ 4.787807147547797e-31
    end

    @testset "σ_atoms_bf_tables" begin
        @test length(σ_atom_itp) == length(atoms)
        @test length(σ_atom_itp_empty) == 0
        @test length.(σ_atom_itp) == length.([atom.continua for atom in atoms])
        # Test against values from atomfile * abundances:
        # Using current abundances:
        @test σ_atom_itp[1][1](50.427) / abund[atoms[1].element] == 6.783e-22
        @test σ_atom_itp[1][2](14.609) / abund[atoms[1].element] == 4.844e-23
        @test σ_atom_itp[2][2](364.70521515693) / abund[atoms[2].element] ≈ 1.3788716766e-21
        # Using previous abundances:
        @test σ_atom_itp[1][1](50.427) ≈ 6.783e-22 * 0.08203515443298176
        @test σ_atom_itp[1][2](14.609) ≈ 4.844e-23 * 0.08203515443298176
        @test σ_atom_itp[2][2](364.70521515693) ≈ 1.3788716752197066e-21
        # A few edges:
        @test σ_atom_itp[1][1](50.428) == 0
        @test σ_atom_itp[1][1](12.7) == 0
        @test σ_atom_itp[2][1](22.793) == 0
        @test σ_atom_itp[2][2](364.8) == 0
        @test σ_atom_itp[2][2](91.175) == 0
    end
    @testset "σH_atoms_bf" begin
        # Against previous implementation:
        @test σH_atoms_bf(σ_atom_itp, atoms, 251.0, 6000.0, 1.0e19) ≈ 5.261436909864e-30
        @test σH_atoms_bf(σ_atom_itp, atoms, 10.0, 6000.0, 1.0e19) == 0
        @test σH_atoms_bf(σ_atom_itp, atoms, 370.0, 6000.0, 1.0e19) ≈ 1.5363426195935e-31
    end
end
