# Functions to calculate intensity and related quantities


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
            profile = voigt_itp(a, abs(v)) / (sqrt(π) * buf.ΔλD[iz])  # units nm^-1
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
