# Functions to calculate intensity and related quantities

using CUDA

const invSqrtPi = 1 / sqrt(π)


"""
    function calc_line_1D!(
        line::AtomicLine,
        buf::RTBuffer{T},
        atm::Atmosphere1D{T},
        n_up::AbstractVector{T},
        n_lo::AbstractVector{T},
        σ_itp::ExtinctionItpNLTE{<:Real},
        voigt_itp::Interpolations.AbstractInterpolation{<:Real, 2},
    )

Calculate emerging disk-centre intensity for a given line in a 1D atmosphere.
"""
function calc_line_1D!(
    line::AtomicLine,
    buf::RTBuffer{T},
    atm::Atmosphere1D{1, T},
    n_up::AbstractVector{T},
    n_lo::AbstractVector{T},
    σ_itp::ExtinctionItpNLTE{<:Real},
    voigt_itp::Interpolations.AbstractInterpolation{<:Real, 2},
) where T <: AbstractFloat
    γ_energy = ustrip((h * c_0 / (4 * π * line.λ0 * u"nm")) |> u"J")

    # wav>elength-independent part (continuum + broadening + Doppler width)
    # Could be done before in 
    for i in 1:atm.nz
        buf.α_c[i] = α_cont(
            σ_itp,
            atm.temperature[i],
            atm.electron_density[i],
            atm.hydrogen1_density[i],
            atm.proton_density[i]
        )
        buf.j_c[i] = buf.α_c[i] * blackbody_λ(line.λ0, atm.temperature[i])
        buf.γ[i] = calc_broadening(
            line.γ,
            atm.temperature[i],
            atm.electron_density[i],
            atm.hydrogen1_density[i]
        )
        # This can also go in the kernel
        buf.ΔλD[i] = doppler_width(line.λ0, line.mass, atm.temperature[i])
    end
    # Calculate line opacity and intensity
    for (i, λ) in enumerate(line.λ)
        # A kernel here instead
        for iz in 1:atm.nz
            # All of this in kernel that takes 3D arrays
            # Wavelength-dependent part
            a = damping(buf.γ[iz], λ, buf.ΔλD[iz])  # very small dependence on λ
            v = (λ - line.λ0 + line.λ0 * atm.velocity_z[iz] / ustrip(c_0)) / buf.ΔλD[iz]
            #profile = voigt_profile(a, v, buf.ΔλD[iz])  # units nm^-1
            profile = voigt_itp(a, abs(v)) / (sqrt(π) * buf.ΔλD[iz])  # units nm^-2
            # Part that only multiplies by wavelength:
            α_tmp = γ_energy * profile
            j_tmp = α_tmp
            α_tmp *= n_lo[iz] * line.Blu - n_up[iz] * line.Bul
            j_tmp *= n_up[iz] * line.Aul
            α_tmp = α_tmp * 1f9 + buf.α_c[iz]   # convert α_tmp to m^-1
            j_tmp = j_tmp * 1f-3 + buf.j_c[iz]  # convert j_tmp to kW m^3 nm^-1
            buf.source_function[iz] = j_tmp / α_tmp
            buf.α_total[iz] = α_tmp
        end
        # long run 
        piecewise_1D_linear!(atm.z, buf.α_total, buf.source_function, buf.int_tmp)
        buf.intensity[i] = buf.int_tmp[1]
    end
    return nothing
end


"""
Calculate continuum optical depth in the vertical direction,
from the observer to the stellar interior. The wavelength
is defined by σ_itp.
"""
function calc_τ_cont!(
    atm::Atmosphere1D{1, T},
    τ::AbstractVector{<:Real},
    σ_itp::ExtinctionItpNLTE{<:Real},
) where T <: AbstractFloat
    τ[1] = zero(T)
    α = α_cont(
        σ_itp,
        atm.temperature[1],
        atm.electron_density[1],
        atm.hydrogen1_density[1],
        atm.proton_density[1]
    )
    for i in 2:atm.nz
        α_next = α_cont(
            σ_itp,
            atm.temperature[i],
            atm.electron_density[i],
            atm.hydrogen1_density[i],
            atm.proton_density[i]
        )
        τ[i] = τ[i-1] + abs(atm.z[i] - atm.z[i-1]) * (α + α_next) / 2
        α = α_next
    end
    return nothing
end

function calc_line_3D!(
    intensity::AbstractArray{T, 3},
    line::AtomicLine,
    buf::RTBuffer{T},
    atm::Atmosphere1D{1, T},
    n_up::AbstractVector{T},
    n_lo::AbstractVector{T},
    σ_itp::ExtinctionItpNLTE{<:Real},
    threads::Tuple{Int, Int},
    blocks::Tuple{Int, Int},
    ) where T <: AbstractFloat

    γ_energy_d = ustrip((Unitful.h * Unitful.c_0 / (4 * π * line.λ0 * u"nm")) |> u"J")
    n_lo_d = CuArray(n_lo)
    n_up_d = CuArray(n_up)
    # wavelength-independent part (continuum + broadening + Doppler width)
    for i in 1:atm.nz
        buf.α_c[i] = α_cont(
            σ_itp,
            atm.temperature[i],
            atm.electron_density[i],
            atm.hydrogen1_density[i],
            atm.proton_density[i]
        )
        buf.j_c[i] = buf.α_c[i] * blackbody_λ(line.λ0, atm.temperature[i])
        buf.γ[i] = calc_broadening(
            line.γ,
            atm.temperature[i],
            atm.electron_density[i],
            atm.hydrogen1_density[i]
        )
        buf.ΔλD[i] = doppler_width(line.λ0, line.mass, atm.temperature[i])
    end
    c_0 = Float32(ustrip(Unitful.c0))
    k_B = Float32(ustrip(Unitful.k))
    λ0 = Float32(line.λ0)
    
    α_c_d = CuArray(buf.α_c)
    j_c_d = CuArray(buf.j_c)
    γ_d = CuArray(buf.γ)
    temperature_d = CuArray(atm.temperature)
    velocity_z_d = CuArray(atm.velocity_z)
    
    α_tot_d = CUDA.fill(0f0, (atm.nz, atm.nz, atm.nz))
    source_d = CUDA.fill(0f0, (atm.nz, atm.nz, atm.nz))
    profile_d = CUDA.fill(0f0, (atm.nz, atm.nz, atm.nz))

    constants_d = GPUinfo(c_0, k_B, λ0, mass, γ_energy_d, 
                    Float32(line.Bul), Float32(line.Blu), Float32(line.Aul));
    
    # Calculate line opacity and intensity
    for λ in line.λ

        @cuda threads=threads blocks=blocks inner_loop!(α_tot_d, source_d, α_c_d,
                            j_c_d, temperature_d, γ_d,
                            velocity_z_d, constants_d, λ, 
                            profile_d, n_lo_d, n_up_d)
        
        buf.source_function = Array(source_d)
        buf.α_total = Array(α_tot_d)
        
        #piecewise_1D_linear!(atm.x, buf.α_total, buf.source_function, buf.int_tmp)
        #piecewise_1D_linear!(atm.y, buf.α_total, buf.source_function, buf.int_tmp)
        piecewise_1D_linear!(atm.z, buf.α_total, buf.source_function, buf.int_tmp)
        intensity[i] = buf.int_tmp[1]
    end

    return nothing
end

function calc_line_1D_GPU!(
    intensity::AbstractArray{T, 3},
    line::AtomicLine,
    buf::RTBuffer{T},
    atm::Atmosphere1D{1, T},
    n_up::AbstractVector{T},
    n_lo::AbstractVector{T},
    σ_itp::ExtinctionItpNLTE{<:Real},
    threads::Tuple{Int, Int, Int},
    blocks::Tuple{Int, Int, Int},
    ) where T <: AbstractFloat

    γ_energy = ustrip((Unitful.h * Unitful.c0 / (4 * π * line.λ0 * u"nm")) |> u"J")
    #n_lo_d = CuArray(n_lo)
    #n_up_d = CuArray(n_up)
    # wavelength-independent part (continuum + broadening + Doppler width)
    for i in 1:atm.nz
        buf.α_c[i] = α_cont(
            σ_itp,
            atm.temperature[i],
            atm.electron_density[i],
            atm.hydrogen1_density[i],
            atm.proton_density[i]
        )
        buf.j_c[i] = buf.α_c[i] * blackbody_λ(line.λ0, atm.temperature[i])
        buf.γ[i] = calc_broadening(
            line.γ,
            atm.temperature[i],
            atm.electron_density[i],
            atm.hydrogen1_density[i]
        )
        #buf.ΔλD[i] = doppler_width(line.λ0, line.mass, atm.temperature[i])
    end
    n_lo_d = CuArray(n_lo)
    n_up_d = CuArray(n_up)
    c_0 = Float32(ustrip(Unitful.c0))
    k_B = Float32(ustrip(Unitful.k))
    λ0 = Float32(line.λ0)
    
    α_c_d = CuArray(buf.α_c)
    j_c_d = CuArray(buf.j_c)
    γ_d = CuArray(buf.γ)
    temperature_d = CuArray(atm.temperature)
    velocity_z_d = CuArray(atm.velocity_z)
    
    α_tot_d = CUDA.fill(0f0, atm.nz)
    source_d = CUDA.fill(0f0, atm.nz)
    profile_d = CUDA.fill(0f0, atm.nz)

    constants = GPUinfo(c_0, k_B, λ0, Float32(line.mass), Float32(γ_energy), 
                    Float32(line.Bul), Float32(line.Blu), Float32(line.Aul));
    
    # Calculate line opacity and intensity
    for (i, λ) in enumerate(line.λ)
        @cuda threads=threads blocks=blocks Vaf.inner_loop_1D!(α_tot_d, source_d, α_c_d,
                            j_c_d, temperature_d, γ_d,
                            velocity_z_d, constants, 
                            profile_d, n_lo_d, n_up_d, Float32(λ))
        # This can be prettier, but with the current structure, must be like this
        # NO you are just dumb and forgot about the COPYTO!
        #=
        for j in size(source_d, 1)
            buf.source_function[j] = Float32(source_d[j])
            buf.α_total[j] = Float32(α_tot_d[j])
        end
        =#
        copyto!(buf.source_function, source_d)
        copyto!(buf.α_total, α_tot_d)
        
        for iz in 1:atm.nz
            piecewise_1D_linear!(atm.z, buf.α_total[iz], buf.source_function[iz], buf.int_tmp[iz])
        end
        intensity[i] = buf.int_tmp[1]
    end

    return nothing
end
