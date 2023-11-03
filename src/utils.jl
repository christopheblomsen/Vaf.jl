"""
Various utility functions.
"""


"""
Calculates the Blackbody (Planck) function per wavelength in nm
and temperature in K. Outputs in kW m^-2 nm^-1.
"""
function blackbody_λ(λ::T1, temperature::T2)::T2 where {T1 <: Real, T2 <: AbstractFloat}
    mult = ustrip((2h * c_0^2) |>  u"kW * m^-2 * nm^4")
    return mult * λ^-5 / (exp(h * c_0 / k_B / (λ * temperature * u"K * nm")) - 1)
end

function blackbody_λ(BB, λ, temperature, h, c_0, k_B)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    mult = 2h * c_0^2
    BB[ix] = mult * λ[ix]^-5 / (exp(h * c_0 / k_B / (λ[ix] * temperature[ix])) - 1) 
    return nothing
end

function source_function(source, 
                        γ_energy, 
                        profile,
                        n_lo,
                        n_up,
                        Blu, Bul,
                        Aul,
                        j_c, α_c)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if (ix <= size(source, 1))
        source[ix] = ( ( (γ_energy*profile)*(n_up[ix] * Aul)*1f-3 + j_c[ix]) 
                    / ( ((γ_energy*profile)*(n_lo[ix] * Blu - n_up[ix] * Bul)*1f9 + α_c[ix] ) ))
    end
    return nothing
end

"""
    inner_loop!(α_c, source, α_total)

Calculates the inner loop of the formal solver. This is the part that
that is paralized over wavelength and the arrays are 3D
"""

function inner_loop!(α_tot, source, α_c, j_c, temperature,
                    γ, velocity_z, constants, profile, 
                    n_lo, n_up, λ)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y - 1) * blockDim().y + threadIdx().y
    iz = (blockIdx().z - 1) * blockDim().z + threadIdx().z
    if (ix <= size(α_c, 1) && iy <= size(α_c, 2) && iz <= size(α_c, 3))
        ΔλD = constants.λ0 / constants.c_0 * sqrt(2 * constants.k_B * temperature[ix, iy, iz] / constants.mass)
        a = ( γ[ix, iy, iz] * λ^2 ) / (4 * π * constants.c_0) / ΔλD
        v = (λ - constants.λ0 + constants.λ0 * velocity_z[ix, iy, iz] / constants.c_0) / ΔλD
        profile[ix, iy, iz] = voigt_humlicek(a, v) / (sqrt(π) * ΔλD)
        #profile = voigt_humlicek(a, v)
        
        α_tmp = constants.γ_energy * profile[ix, iy, iz]
        j_tmp = α_tmp
        α_tmp *= n_lo[ix, iy, iz] * constants.Blu - n_up[ix, iy, iz] * constants.Bul
        j_tmp *= n_up[ix, iy, iz] * constants.Aul
        α_tmp = α_tmp * 1f9 + α_c[ix, iy, iz]   # convert α_tmp to m^-1
        j_tmp = j_tmp * 1f-3 + j_c[ix, iy, iz]  # convert j_tmp to kW m^3 nm^-1

        source[ix, iy, iz] = j_tmp / α_tmp
        α_tot[ix, iy, iz] = α_tmp
    end 
    return nothing 
end

function inner_loop_cpu!(line, buf, atm)
    for iz in 1:atm.nz
        # All of this in kernel that takes 3D arrays
        # Wavelength-dependent part
        a = damping(buf.γ[iz], λ, buf.ΔλD[iz])  # very small dependence on λ
        v = (λ - line.λ0 + line.λ0 * atm.velocity_z[iz] / ustrip(c_0)) / buf.ΔλD[iz]
        #profile = voigt_profile(a, v, buf.ΔλD[iz])  # units nm^-1
        profile = voigt_humlicek(a, v) / (sqrt(π) * buf.ΔλD[iz])  # units nm^-2 
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
end