
"""
Tools that are used in the calculation of the radiative transfer equation.
on the GPU
"""

"""
The adapt functions are needed to pass the Atmosphere1D and Atmosphere3D
onto the GPU.
"""
function Adapt.adapt_structure(to, atmos::Atmosphere1D)
    Atmosphere1D(
        atmos.nx,
        atmos.ny,
        atmos.nz,
        adapt(to, atmos.z),
        adapt(to, atmos.temperature),
        adapt(to, atmos.velocity_z),
        adapt(to, atmos.electron_density),
        adapt(to, atmos.hydrogen1_density),
        adapt(to, atmos.proton_density)
    )
end

function Adapt.adapt_structure(to, atmos::Atmosphere3D)
    Atmosphere3D(
        atmos.nx,
        atmos.ny,
        atmos.nz,
        adapt(to, atmos.x),
        adapt(to, atmos.y),
        adapt(to, atmos.z),
        adapt(to, atmos.temperature),
        adapt(to, atmos.velocity_x),
        adapt(to, atmos.velocity_y),
        adapt(to, atmos.velocity_z),
        adapt(to, atmos.electron_density),
        adapt(to, atmos.hydrogen1_density),
        adapt(to, atmos.proton_density)
    )
end


function Adapt.adapt_structure(to, itp::ExtinctionItpNLTE)
    ExtinctionItpNLTE(
        adapt(to, itp.σ_atoms),    
        adapt(to, itp.σ_hminus),   
        adapt(to, itp.σ_h2plus),   
        adapt(to, itp.σ_h_ff),
        itp.λ,
    )
end


function Adapt.adapt_structure(to, line::AtomicLine)
    AtomicLine(
        line.nλ,
        line.χup,
        line.χlo,
        line.gup,
        line.glo,
        line.Aul,
        line.Blu,
        line.Bul,
        line.λ0,
        line.f_value,
        line.mass,
        adapt(to, line.λ),
        line.PRD,
        line.Voigt,
        nothing,
        nothing,
        line.γ,
    )
end

"""
Modified version of voigt.jl from Muspel.jl but with Float32.

Compute scaled complex complementary error function
using [Humlicek (1982, JQSRT 27, 437)](https://doi.org/10.1016/0022-4073(82)90078-4)
W4 rational approximations.
Here, z is defined as z = v + i * a, and returns w(z) = H(a,v) + i * L(a, v).
"""
function voigt_humlicek(a, v)
    z = v + a * im
    s = abs(real(z)) + imag(z)
    if s > 15.0f0
        # region I
        w = im * invSqrtPi * z / (z * z - 0.5f0)
    elseif s > 5.5f0
        # region II
        zz = z * z
        w = im * (z * (zz * invSqrtPi - 1.4104739589f0)) / (0.75f0 + zz * (zz - 3.0f0))
    else
        x, y = real(z), imag(z)
        t = y - im * x
        if y >= 0.195f0 * abs(x) - 0.176f0
            # region III
            w = ((16.4955f0 + t * (20.20933f0 + t * (11.96482f0 + t * (3.778987f0 + 0.5642236f0 * t))))
               / (16.4955f0 + t * (38.82363f0 + t * (39.27121f0 + t * (21.69274f0 + t * (6.699398f0 + t))))))
        else
            # region IV
            u = t * t
            nom = t * (36183.31f0 - u * (3321.99f0 - u * (1540.787f0 -  u *
                   (219.031f0 - u * (35.7668f0 - u * (1.320522f0 - u * .56419f0))))))
            den = 32066.6f0 - u * (24322.8f0 - u * (9022.23f0 - u * (2186.18f0 -
                    u * (364.219f0 - u * (61.5704f0 - u * (1.84144f0 - u))))))
            w = exp(u) - nom / den
        end
    end
    return real(w)
end

function read_pops_multi3d(pops_file, nx, ny, nz, nlevels)
    u_l = ustrip(1f0u"cm" |> u"m")
    pops = Array{Float32}(undef, nx, ny, nz, nlevels)
    read!(pops_file, pops)
    # convert to SI
    for i in eachindex(pops)
        pops[i] = pops[i] / u_l^3
    end
    return pops
end

"""
    function calc_line_gpu!(
        line: AtomicLine,
        atm: Atmosphere1D,
        n_up: CuArray{Float32, 3},
        n_lo: CuArray{Float32, 3},
        σ_itp: ExtinctionItpNLTE,
        intensity: CuArray{Float32, 3},

Calculate the intensity of a spectral line on the GPU
"""
function calc_line_gpu!(
    line::AtomicLine,
    atm::Atmosphere1D{3, T},
    n_up::AbstractArray{T, 3},
    n_lo::AbstractArray{T, 3},
    σ_itp::ExtinctionItpNLTE{<:Real},
    intensity::AbstractArray{T, 3},
) where T <: AbstractFloat
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    nz = atm.nz
    ny = atm.ny
    nx = atm.nx
    γ_energy = γ_mult / line.λ0
    
    int_tmp = @MVector zeros(Float32, global_ndep)
    α_c = @MVector zeros(Float32, global_ndep)
    α_line = @MVector zeros(Float32, global_ndep)
    S_cont = @MVector zeros(Float32, global_ndep)
    S_line = @MVector zeros(Float32, global_ndep)
    damp = @MVector zeros(Float32, global_ndep)
    ΔλD = @MVector zeros(Float32, global_ndep)
   
    @inbounds if (i <= nx) && (j <= ny)
        # before wave loop, calculate
        for k in 1:nz
            α_c[k] = α_cont(
                σ_itp,
                atm.temperature[k, j, i],
                atm.electron_density[k, j, i],
                atm.hydrogen1_density[k, j, i],
                atm.proton_density[k, j, i],
            )
            S_cont[k] = blackbody_λ(σ_itp.λ, atm.temperature[k, j, i])
            
            ΔλD[k] = doppler_width(line.λ0, line.mass, atm.temperature[k, j, i])
            γ = calc_broadening(
                line.γ,
                atm.temperature[k, j, i],
                atm.electron_density[k, j, i],
                atm.hydrogen1_density[k, j, i],
            )
            damp[k] = damping(γ, line.λ0, ΔλD[k]) 
            α_line[k] = γ_energy * (
                    n_lo[k, j, i] * line.Blu - n_up[k, j, i] * line.Bul) * 1f9  # to m^-1
            S_line[k] = γ_energy * n_up[k, j, i] * line.Aul * 1f-3 / α_line[k]   # to kW m^2 nm^-1
        end

        for (w, λ) in enumerate(line.λ)
            v = (λ - line.λ0 + line.λ0 * atm.velocity_z[nz, j, i] / c_0u) / ΔλD[nz]
            profile = voigt_humlicek(damp[nz], abs(v)) / ΔλD[nz] * invSqrtPi
            α_old = α_c[nz] + α_line[nz] * profile
            S_old = S_cont[nz]  # at depth, S_total = S_cont = B because of LTE
            int_tmp[nz] = S_old  # correct for line source function in LTE, CRD

            # piecewise explicitly
            incr = -1 
            for k in nz-1:incr:1 
                # calculate all of these, all of the time:
                v = (λ - line.λ0 + line.λ0 * atm.velocity_z[k, j, i] / c_0u) / ΔλD[k]
                profile = voigt_humlicek(damp[k], abs(v)) / ΔλD[k] * invSqrtPi
                η = α_line[k] * profile / α_c[k]
                
                α_new = α_c[k] + α_line[k] * profile
                S_new = (η * S_line[k] + S_cont[k]) / (1 + η)
                
                Δτ = abs(atm.z[k] - atm.z[k-incr]) * (α_new + α_old) / 2
                ΔS = (S_old - S_new) / Δτ
                w1, w2 = Muspel._w2(Δτ)
                int_tmp[k] = (1 - w1)*int_tmp[k-incr] + w1*S_new + w2*ΔS
    
                S_old = S_new
                α_old = α_new
                

                intensity[j, i, w] = int_tmp[k]
            end
        end
    end
    return nothing
end

"""
    function precalc_values!(
        α_c::CuArray{Float32, 3},
        α_line::CuArray{Float32, 3},
        S_cont::CuArray{Float32, 3},
        S_line::CuArray{Float32, 3},
        damp::CuArray{Float32, 3},
        ΔλD::CuArray{Float32, 3},
        line::AtomicLine,
        atm::Atmosphere3D,
        n_up::CuArray{Float32, 3},
        n_lo::CuArray{Float32, 3},
        σ_itp::ExtinctionItpNLTE{<:Real},
    )

Precalculating all the values needed for the radiative transfer equation
used in the GPU calculation of the spectral line. 
Used with calc_line_inclined_gpu!.
"""
function precalc_values!(
    α_c::AbstractArray{T, 3},
    α_line::AbstractArray{T, 3},
    S_cont::AbstractArray{T, 3},
    S_line::AbstractArray{T, 3},
    damp::AbstractArray{T, 3},
    ΔλD::AbstractArray{T, 3},
    line::AtomicLine,
    atm,
    n_up,
    n_lo,
    σ_itp,
    ) where T <: AbstractFloat
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y

    nz = atm.nz
    ny = atm.ny
    nx = atm.nx

    α_c_tmp = @MVector zeros(Float32, global_ndep)
    α_line_tmp = @MVector zeros(Float32, global_ndep)
    S_cont_tmp = @MVector zeros(Float32, global_ndep)
    S_line_tmp = @MVector zeros(Float32, global_ndep)
    damp_tmp = @MVector zeros(Float32, global_ndep)
    ΔλD_tmp = @MVector zeros(Float32, global_ndep)


    γ_energy = γ_mult / line.λ0

    @inbounds if (i <= nx) && (j <= ny)
        # before wave loop, calculate
        for k in 1:nz
            α_c_tmp[k] = α_cont(
                σ_itp,
                atm.temperature[k, j, i],
                atm.electron_density[k, j, i],
                atm.hydrogen1_density[k, j, i],
                atm.proton_density[k, j, i],
            )
            S_cont_tmp[k] = blackbody_λ(σ_itp.λ, atm.temperature[k, j, i])
            
            ΔλD_tmp[k] = doppler_width(line.λ0, line.mass, atm.temperature[k, j, i])
            γ = calc_broadening(
                line.γ,
                atm.temperature[k, j, i],
                atm.electron_density[k, j, i],
                atm.hydrogen1_density[k, j, i],
            )
            damp_tmp[k] = damping(γ, line.λ0, ΔλD_tmp[k]) 
            α_line_tmp[k] = γ_energy * (
                    n_lo[k, j, i] * line.Blu - n_up[k, j, i] * line.Bul) * 1f9  # to m^-1
            S_line_tmp[k] = γ_energy * n_up[k, j, i] * line.Aul * 1f-3 / α_line_tmp[k]   # to kW m^2 nm^-1

            α_c[k, j, i] = α_c_tmp[k]
            α_line[k, j, i] = α_line_tmp[k]
            S_cont[k, j, i] = S_cont_tmp[k]
            S_line[k, j, i] = S_line_tmp[k]
            ΔλD[k, j, i] = ΔλD_tmp[k]
            damp[k, j, i] = damp_tmp[k]
        end
    end
end

"""
function calc_line_inclined_gpu!(
    line::AtomicLine,
    atm,
    α_c,
    α_line,
    S_cont,
    S_line,
    damp,
    ΔλD,
    v_los,
    μ,
    intensity,
)

Calculate the intensity over the line of sight for an inclined atmosphere.
Need to incline all the data before running this function.
"""
function calc_line_inclined_gpu!(
    line::AtomicLine,
    atm,
    α_c,
    α_line,
    S_cont,
    S_line,
    damp,
    ΔλD,
    v_los,
    μ,
    intensity,
)
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    j = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    nz = atm.nz
    ny = atm.ny
    nx = atm.nx
    
    int_tmp = @MVector zeros(Float32, global_ndep)
   
    @inbounds if (i <= nx) && (j <= ny)
        # before wave loop, calculate
        
        for (w, λ) in enumerate(line.λ)
            if λ == line.λ[end]
                λ_new = line.λ[end] + abs(line.λ[end] - line.λ[end-1])
            else
                λ_new = line.λ[w+1]
            end
            # Boundary condition
            v = (λ - line.λ0 + line.λ0 * v_los[nz, j, i] / c_0u) / ΔλD[nz, j, i]
            profile = voigt_humlicek(damp[nz, j, i], abs(v)) / ΔλD[nz, j, i] * invSqrtPi
            α_old = α_c[nz, j, i] + α_line[nz, j, i] * profile
            S_old = S_cont[nz, j, i]  # at depth, S_total = S_cont = B because of LTE
            int_tmp[nz] = S_old  # correct for line source function in LTE, CRD
            int_old = S_old

            # piecewise explicitly
            incr = -1 
            for k in nz-1:incr:1 
                # calculate all of these, all of the time:
                v = (λ - line.λ0 + line.λ0 * v_los[k, j, i] / c_0u) / ΔλD[k, j, i]
                profile = voigt_humlicek(damp[k, j, i], abs(v)) / ΔλD[k, j, i] * invSqrtPi
                η = α_line[k, j, i] * profile / α_c[k, j, i]
                
                α_new = α_c[k, j, i] + α_line[k, j, i] * profile
                S_new = (η * S_line[k, j, i] + S_cont[k, j, i]) / (1 + η)
                
                Δτ = abs(atm.z[k] - atm.z[k-incr])/μ * (α_new + α_old) / 2
                ΔS = (S_old - S_new) / Δτ
                w1, w2 = Muspel._w2(Δτ)
                int_tmp[k] = (1 - w1)*int_tmp[k-incr] + w1*S_new + w2*ΔS
    
                S_old = S_new
                α_old = α_new
                
                int_new = int_tmp[k]
                Δλ = (λ_new - λ)/2

                intensity[k, j, i] = (int_new + int_old)*Δλ*profile  # this needs to go into k loop, and multiply by the weights and profile
                int_old = int_new
            end

            # Above is tested and working
            # Boundary condition for I- (I=0, S=0, α=0)
            α_old = 0
            S_old = 0  
            int_tmp[nz] = S_old 
            int_old = S_old

            incr = 1
            for k in 2:incr:1 
                # calculate all of these, all of the time:
                v = (λ - line.λ0 + line.λ0 * v_los[k, j, i] / c_0u) / ΔλD[k, j, i]
                profile = voigt_humlicek(damp[k, j, i], abs(v)) / ΔλD[k, j, i] * invSqrtPi
                η = α_line[k, j, i] * profile / α_c[k, j, i]
                
                α_new = α_c[k, j, i] + α_line[k, j, i] * profile
                S_new = (η * S_line[k, j, i] + S_cont[k, j, i]) / (1 + η)
                
                Δτ = -abs(atm.z[k] - atm.z[k-incr])/μ * (α_new + α_old) / 2
                ΔS = (S_old - S_new) / Δτ
                w1, w2 = Muspel._w2(Δτ)
                int_tmp[k] = (1 - w1)*int_tmp[k-incr] + w1*S_new + w2*ΔS
    
                S_old = S_new
                α_old = α_new
                
                int_new = int_tmp[k]
                Δλ = (λ_new - λ)/2

                intensity[k, j, i] = (int_new + int_old)*Δλ*profile  # this needs to go into k loop, and multiply by the weights and profile
                int_old = int_new
            end
        end
    end
    return nothing
end

Adapt.@adapt_structure Atmosphere1D
Adapt.@adapt_structure Atmosphere3D
Adapt.@adapt_structure ExtinctionItpNLTE