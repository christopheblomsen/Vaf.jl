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
    #=
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
    =#
    a = damping(buf.γ[1], λ, buf.ΔλD[1])  # very small dependence on λ
    v = (λ - line.λ0 + line.λ0 * atm.velocity_z[1] / ustrip(c_0)) / buf.ΔλD[1]
    profile = CuArray{Float32, 2}(undef, length(a), length(v))
    @cuda threads=threads blocks=blocks voigt_profile!(
        profile,
        a,
        v,
    )
    return nothing
end

function humlicek(z::Complex)
    s = abs(real(z)) + imag(z)
    if s > 15.0
        # region I
        w = im * invSqrtPi * z / (z * z - 0.5)
    elseif s > 5.5
        # region II
        zz = z * z
        w = im * (z * (zz * invSqrtPi - 1.4104739589)) / (0.75 + zz * (zz - 3.0))
    else
        x, y = real(z), imag(z)
        t = y - im * x
        if y >= 0.195 * abs(x) - 0.176
            # region III
            w = ((16.4955 + t * (20.20933 + t * (11.96482 + t * (3.778987 + 0.5642236 * t))))
               / (16.4955 + t * (38.82363 + t * (39.27121 + t * (21.69274 + t * (6.699398 + t))))))
        else
            # region IV
            u = t * t
            nom = t * (36183.31 - u * (3321.99 - u * (1540.787 -  u *
                   (219.031 - u * (35.7668 - u * (1.320522 - u * .56419))))))
            den = 32066.6 - u * (24322.8 - u * (9022.23 - u * (2186.18 -
                    u * (364.219 - u * (61.5704 - u * (1.84144 - u))))))
            w = exp(u) - nom / den
        end
    end
    return w
end

function humlicek!(a::AbstractArray{T},
                v::AbstractArray{T},
                ans::AbstractArray{T}) where {T <: AbstractFloat}
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x

    w = Complex(0f0, 0f0)
    if (ix > 1 && ix < size(a, 1) && ix < size(v, 1))
        z = CUDA.Complex(a[ix], v[ix])
        s = abs(CUDA.real(a[ix])) + CUDA.imag(v[ix])
        if s > 15.0
            # region I
            w = im * invSqrtPi * z / (z * z - 0.5)
        elseif s > 5.5
            # region II
            zz = z * z
            w = im * (z * (zz * invSqrtPi - 1.4104739589)) / (0.75 + zz * (zz - 3.0))
        else
            x, y = real(z), imag(z)
            t = y - im * x
            if y >= 0.195 * abs(x) - 0.176
                # region III
                w = ((16.4955 + t * (20.20933 + t * (11.96482 + t * (3.778987 + 0.5642236 * t))))
                   / (16.4955 + t * (38.82363 + t * (39.27121 + t * (21.69274 + t * (6.699398 + t)))                )))
            else
                # region IV
                u = t * t
                nom = t * (36183.31 - u * (3321.99 - u * (1540.787 -  u *
                       (219.031 - u * (35.7668 - u * (1.320522 - u * .56419))))))
                den = 32066.6 - u * (24322.8 - u * (9022.23 - u * (2186.18 -
                        u * (364.219 - u * (61.5704 - u * (1.84144 - u))))))
                w = exp(u) - nom / den
            end
        end
        ans[ix] = w
    end
    return nothing
end

function voigt_profile(a, v, ΔD, threads::Tuple, blocks::Tuple)
    profile = CuArray{ComplexF32, 1}(undef, length(v))
    
    @cuda threads=threads blocks=blocks humlicek!(a, v, profile)

    return profile * invSqrtPi / ΔD
end
