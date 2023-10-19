using CUDA

const invSqrtPi = 1f0 / CUDA.sqrt(π)
# Functions to calculate intensity.
#=
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
=#
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
    # Calculate line opacity and intensity
    for (i, λ) in enumerate(line.λ)
        for iz in 1:atm.nz
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
        piecewise_1D_linear!(atm.z, buf.α_total, buf.source_function, buf.int_tmp)
        buf.intensity[i] = buf.int_tmp[1]
    end
    return nothing
end

function calc_line_1D!(
    line::AtomicLine,
    buf::RTBuffer{T},
    atm::Atmosphere1D{1, T},
    n_up::AbstractVector{T},
    n_lo::AbstractVector{T},
    σ_itp::ExtinctionItpNLTE{<:Real},
    voigt_itp::Interpolations.AbstractInterpolation{<:Real, 2},
    threads::Tuple{Int, Int},
    blocks::Tuple{Int, Int},
) where T <: AbstractFloat
    γ_energy = ustrip((h * c_0 / (4 * π * line.λ0 * u"nm")) |> u"J")

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
    # Calculate line opacity and intensity
    for (i, λ) in enumerate(line.λ)
        for iz in 1:atm.nz
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
        piecewise_1D_linear!(atm.z, buf.α_total, buf.source_function, buf.int_tmp)
        buf.intensity[i] = buf.int_tmp[1]
    end
    a = damping(buf.γ[1], λ, buf.ΔλD[1])  # very small dependence on λ
    v = (λ - line.λ0 + line.λ0 * atm.velocity_z[1] / ustrip(c_0)) / buf.ΔλD[1]
    profile = voigt_profile(a, v, buf.ΔλD[1], threads, blocks)  # units nm^-1

    return profile
end

function damping(γ, λ, ΔλD)
    #ix = 
    c1 = 1 / (4 * π * c_0)
    damping_parameter = c1 * γ * λ^2 / ΔλD
    return nothing
end

function wavelength_independent_part()
    #=
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
    =#
    # Must calculate α_cont, blackbody_λ and calc_broadening on the gpu

    return nothing
end

function α_cont(
    itp::ExtinctionItpLTE{<: Real},
    temperature::T,
    electron_density::T,
    hydrogen_density::T,
    )::T where T <: AbstractFloat
    log_temp = CUDA.log10(temperature)
    log_ne = CUDA.log10(electron_density)
    α = itp.σ_H(log_temp, log_ne) * hydrogen_density
    α += (itp.σ_H2(log_temp, log_ne) * hydrogen_density) * hydrogen_density
    α += σ_THOMSON * electron_density
    return α
end
