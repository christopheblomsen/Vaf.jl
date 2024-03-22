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

"""
Returns the nodes and weights form a Gaussian
quadrature with k points. Rescaled to an interval
from [0, 1].
"""
function gauss_quadrature(n)
    nodes, weights = gausslegendre(n)
    # Scaling from -1 to 1 to 0 to 1    
    return (nodes ./2 .+ 0.5, weights ./ 2)
end