"""
IMPORTANT:
* This function assumes that data_in is of shape (nz, ny, nx).
* If ϕ is not a multiple of 90 deg, data_in will be overwritten.

For the (467, 252, 252) case, the optimal CUDA options were:
nthreads=(4,64), nblocks=(64,8).
"""
function incline_data_gpu!(
        data_in::AbstractArray{<: Real, 3},
        data_out::AbstractArray{<: Real, 3},
        z::AbstractVector,
        dx::Real,
        dy::Real,
        μ::Real,
        ϕ::Real,
        nthreads::Tuple,
        nblocks::Tuple,
)
    # Invert θ so that inclination goes towards increasing x or y values
    μ = -μ
    θ = acos(μ)
    sinθ = sin(θ)
    tanθ = sinθ / μ
    cosϕ = cos(ϕ)
    sinϕ = sin(ϕ)
    ∂x∂z = tanθ * cosϕ
    ∂y∂z = tanθ * sinϕ
    nz, ny, nx = size(data_in)
    @assert nz == length(z)
    @assert dx != 0
    @assert dy != 0
    ε = 1.0e-6
    changed = false

    if abs(∂x∂z) > ε   # μ shift in the x dimension
        CUDA.@sync @cuda threads=nthreads blocks=nblocks incline_x!(data_in, data_out, ∂x∂z, z, dx)
        changed = true
    end

    if abs(∂y∂z) > ε   # ϕ shift in the y dimension
        if changed
            copy!(data_in, data_out)
        end
        CUDA.@sync @cuda threads=nthreads blocks=nblocks incline_y!(data_in, data_out, ∂y∂z, z, dy)
    end

    return nothing
end

"""
Kernel for inclining in x dimension.
"""
function incline_x!(data_in, data_out, ∂x∂z, z, dx)
    nz, ny, nx = size(data_in)
    l = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (l <= nx) && (n <= nz)
        shift_x = ∂x∂z * z[n] / (nx*dx)
        (k, ac, bc, ad, bd) = Muspel._spline_coeffs(shift_x, nx)
        m1, p0, p1, p2 = Muspel._spline_stencil(l, k, nx)
        for m in 1:ny
            data_out[n,m,l] = ac*data_in[n,m,p0] + bc*data_in[n,m,p1] -
                              ad*data_in[n,m,m1] + bd*data_in[n,m,p2]
        end
    end
    return nothing
end


"""
Kernel for inclining in y dimension.

For the (467, 252, 252) case, the optimal CUDA options were:
nthreads=(4,4,32),  nblocks=(64,64,16).
"""
function incline_y!(data_in, data_out, ∂y∂z, z, dy)
    nz, ny, nx = size(data_in)
    m = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    n = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    if (m <= ny) && (n <= nz)
        shift_y = ∂y∂z * z[n] / (ny*dy)
        (k, ac, bc, ad, bd) = Muspel._spline_coeffs(shift_y, ny)
        m1, p0, p1, p2 = Muspel._spline_stencil(m, k, ny)
        for l in 1:nx
            data_out[n,m,l] = ac*data_in[n,p0,l] + bc*data_in[n,p1,l] -
                              ad*data_in[n,m1,l] + bd*data_in[n,p2,l]
        end
    end
    return nothing
end


"""
Projects the line of sight velocity into v_los, given vx, vy, vz, and μ = cosθ, φ.
"""
function project_vlos_gpu!(
    vx::A,
    vy::A,
    vz::A,
    v_los::A,
    μ::Real,
    φ::Real,
    nthreads::Tuple,
    nblocks::Tuple,
) where A <: AbstractArray{<:Real}
    cosθ = μ
    sinθ = sqrt(1 - μ^2)
    cosφ = cos(φ)
    sinφ = sin(φ)
    CUDA.@sync @cuda threads=nthreads blocks=nblocks project_kernel!(vx, vy, vz, v_los, cosθ, sinθ, cosφ, sinφ)
    return nothing
end


function project_kernel!(vx, vy, vz, v_los, cosθ, sinθ, cosφ, sinφ)
    nz, ny, nx = size(vx)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    k = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    @inbounds if (i <= nx) && (j <= ny) && (k <= nz)
        v_los[k, j, i] = vx[k, j, i]*sinθ*cosφ + vy[k, j, i]*sinθ*sinφ + vz[k, j, i]*cosθ
    end
    return nothing
end
