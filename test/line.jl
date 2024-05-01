import SpecialFunctions: erfcx

@testset "line.jl" begin
    @test doppler_width(500., 1f-20, 5f3) / doppler_width(500., 1e-20, 6f3) ≈ sqrt(5f3/6f3)
    @test doppler_width(0., 1f-20, 5f3) == 0
    @test doppler_width(500., 1f-20, 0.) == 0
    @test typeof(doppler_width(5e3, 1e-20, 5f3)) == Float32  # Follow type of temperature
    @test typeof(doppler_width(5f3, 1f-20, 5e3)) == Float64

    @test isapprox(damping(1e8, 500., 0.01), 0.0006636047, rtol=1e-6)
    @test damping(0.0, 500., 0.01) == 0.

    na = 10
    nv = 10
    a = LinRange(1f-4, 1f1, na)
    v = LinRange(-5f2, 5f2, nv)
    tmp = create_voigt_itp(a, v)
    results = zeros(Bool, na, nv)
    # check that table points are indeed the Voigt function
    for i in eachindex(a)
        for j in eachindex(v)
            results[i, j] = tmp(a[i], v[j]) ≈ erfcx(a[i] - v[j]*im)
        end
    end
    @test all(results)
end
