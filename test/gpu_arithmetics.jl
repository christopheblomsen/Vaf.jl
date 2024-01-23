using CUDA

@testset "gpu_arithmetics.jl" begin
    N = 10
    x_d = CuArray{Float32}(ones(N))
    y_d = CuArray{Float32}(ones(N))

    CUDA.@sync @cuda addition!(x_d, y_d)
    @test all(Array(x_d) .== 2f0)

    x_d = CUDA.fill(3f0, N)
    CUDA.@sync @cuda multiply!(x_d, y_d)
    @test all(Array(x_d) .== 3f0)

    CUDA.@sync @cuda subtract!(x_d, y_d)
    @test all(Array(x_d) .== 1f0)
    
    CUDA.@sync @cuda divide!(x_d, y_d)
    @test all(Array(x_d) .== 1f0)

end