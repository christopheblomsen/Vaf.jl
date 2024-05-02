@testset "lte.jl" begin
    H_empty = read_atom("test_atoms/H_test_empty.yaml")
    @test_throws MethodError saha_boltzmann(H_empty, 1, 1e15, 1.)
    @test_throws MethodError saha_boltzmann(H_empty, 1e6, 1, 1.)
    # Limits:
    @test all(saha_boltzmann(H_empty, 1e2, 1e15, 1.0) .== [1.0, 0, 0, 0, 0, 0])
    @test all(isapprox.(saha_boltzmann(H_empty, 1e20, 1e15, 1.), [0, 0, 0, 0, 0, 1.0], atol=1e-20))
    @test sum(saha_boltzmann(H_empty, 1e3, 1e20, 1.)) ≈ 1
    @test sum(saha_boltzmann(H_empty, 1e6, 1e20, 1.)) ≈ 1
    # Inplace vs allocating version
    tmp = Array{Float64, 1}(undef, H_empty.nlevels)
    saha_boltzmann!(H_empty, 1e3, 1e20, 1., tmp)
    @test tmp ≈ saha_boltzmann(H_empty, 1e3, 1e20, 1.)
    # Two versions:
    @test saha_boltzmann(
        H_empty.χ, H_empty.g, H_empty.stage, 6e3, 1e18, 1.
    ) == saha_boltzmann(H_empty, 6e3, 1e18, 1.)
    # Output against previous implementation:
    temp = [6e3, 1e4, 1e6]
    ne = [1e15, 1e17, 1e20]
    @test saha_boltzmann(H_empty, temp[1], ne[1], 1.) ≈ [
        0.19060546323810693,
        2.06799903712367e-9,
        1.2059252253378388e-10,
        5.969788074265383e-11,
        5.161507262903375e-11,
        0.8093945344619885,
    ]
    @test saha_boltzmann(H_empty, temp[2], ne[1], 1.) ≈ [
        2.9537642920495862e-6,
        8.559508453727741e-11,
        2.151710608057376e-11,
        1.776306769277307e-11,
        1.945971871991794e-11,
        0.9999970460913731,
    ]
    @test saha_boltzmann(H_empty, temp[3], ne[1], 1.) ≈ [
        4.849223942917004e-16,
        1.7231868116925775e-15,
        3.793118454516546e-15,
        6.691792178786093e-15,
        1.0418866099361604e-14,
        0.9999999999999768,
    ]
    @test saha_boltzmann(H_empty, temp[1], ne[2], 1.) ≈ [
        0.9592653666733548,
        1.0407675734606001e-8,
        6.069093108938104e-10,
        3.0044316929513423e-10,
        2.597645981926035e-10,
        0.040734621751852364,
    ]
    @test saha_boltzmann(H_empty, temp[1], ne[3], 1.) ≈ [
        0.9999575253446236,
        1.0849170660937385e-8,
        6.326544809332608e-10,
        3.1318800668987306e-10,
        2.707838454354065e-10,
        4.24625895794503e-5,
    ]
    @test Muspel.h_ionfrac_saha(0f0, 1f10) == 0
    @test Muspel.h_ionfrac_saha(9500f0, 1f20) ≈ 0.57733727
    @test Muspel.h_ionfrac_saha(9500f0, 1f22) ≈ 0.013475458
    @test isa(Muspel.h_ionfrac_saha(9500f0, 1f22), Float32)
    @test isa(Muspel.h_ionfrac_saha(9500., 1e22), Float64)
end
