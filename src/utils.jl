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

function blackbody_λ(λ, temperature, h, c_0, k_B)
    ix = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    if (ix > 1 && size(λ) < 1)

    end
    return nothing
end
