@testset "read_atmos.jl" begin
    FALC_RH_file = joinpath(@__DIR__, "..", "data", "atmospheres", "FALC.hdf5")
    FALC_RH_nHtot_file = joinpath(@__DIR__, "..", "data", "atmospheres", "FALC_nHtot.hdf5")
    FALC_multi3d = joinpath(@__DIR__, "..", "data", "atmospheres", "atm3d.FALC.3x3x82")
    FALC_multi3d_mesh = joinpath(@__DIR__, "..", "data", "atmospheres", "mesh.FALC.3x3x82")
    FALC_multi3d_mesh2 = joinpath(@__DIR__, "..", "data", "atmospheres", "mesh.FALC.3x3x82_lines")

    @testset "RH" begin
        # Atmosphere1D tests
        atm = read_atmos_rh(FALC_RH_file)
        @test typeof(atm) == Atmosphere1D{3, Float32, Array{Float32, 3}, Vector{Float32}}
        @test typeof(atm[1,1]) <: Atmosphere1D{1, <:AbstractFloat}
        @test atm[1,1].temperature == atm.temperature[:, 1, 1]
        @test atm[1,1].velocity_z == atm.velocity_z[:, 1, 1]
        @test atm[1,1].electron_density == atm.electron_density[:, 1, 1]
        @test atm[1,1].hydrogen1_density == atm.hydrogen1_density[:, 1, 1]
        @test atm[1,1].proton_density == atm.proton_density[:, 1, 1]
        @test atm[1,1].z == atm.z
        @test maximum(atm[1,1].temperature) == 1e5
        @test minimum(atm[1,1].temperature) == 4.5f3
        # Compare with version with nHtot
        atmH = read_atmos_rh(FALC_RH_nHtot_file)
        @test all(atmH.velocity_z .== atm.velocity_z)
        @test all(atmH.temperature .== atm.temperature)
        @test all(atmH.electron_density .== atm.electron_density)
        nH1 = atmH.proton_density[:, 1, 1] .+ atmH.hydrogen1_density[:, 1, 1]
        nH2 = atm.proton_density[:, 1, 1] .+ atm.hydrogen1_density[:, 1, 1]
        @test nH2 ≈ nH1
    end

    @testset "Multi3D" begin
        atm = read_atmos_multi3d(FALC_multi3d_mesh, FALC_multi3d)
        @test typeof(atm) == Atmosphere3D{Float32, Array{Float32, 3}, Vector{Float32}}
        @test typeof(atm[1,1]) <: Atmosphere1D{1, <:AbstractFloat}
        @test atm[1,1].temperature == atm.temperature[:, 1, 1]
        @test atm[1,1].velocity_z == atm.velocity_z[:, 1, 1]
        @test atm[1,1].electron_density == atm.electron_density[:, 1, 1]
        @test atm[1,1].hydrogen1_density == atm.hydrogen1_density[:, 1, 1]
        @test atm[1,1].proton_density == atm.proton_density[:, 1, 1]
        @test atm[1,1].z == atm.z
        @test maximum(atm[1,1].temperature) == 1e5
        @test minimum(atm[1,1].temperature) == 4.5f3
        # Compare against FALC in RH format
        atm_RH = read_atmos_rh(FALC_RH_file)
        @test atm.velocity_z == atm_RH.velocity_z
        @test atm.temperature == atm_RH.temperature
        @test atm.electron_density ≈ atm_RH.electron_density
        @test atm.velocity_z == atm_RH.velocity_z
        atm_RH = read_atmos_rh(FALC_RH_nHtot_file)
        @test atm[1,1].proton_density ≈ atm_RH[1,1].proton_density
        @test atm[1,1].hydrogen1_density ≈ atm_RH[1,1].hydrogen1_density
        # Tests of mesh, arrays in single line vs split lines
        z1 = Muspel.read_mesh(FALC_multi3d_mesh)[end]
        z2 = Muspel.read_mesh(FALC_multi3d_mesh2)[end]
        @test z1 == z2
    end
end
