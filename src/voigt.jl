"""
Computes line profiles and associated quantites

Modified version of voigt.jl from Transparency.jl
"""

const invSqrtPi = 1. / sqrt(π)

"""
Compute scaled complex complementary error function
using [Humlicek (1982, JQSRT 27, 437)](https://doi.org/10.1016/0022-4073(82)90078-4)
W4 rational approximations.
Here, z is defined as z = v + i * a, and returns w(z) = H(a,v) + i * L(a, v).
"""
function humlizek(z::Complex)
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

"""
    voigt_profile(a::T, v::AbstractFloat, ΔD::T)::T

Compute the normalised Voigt profile, given a damping constant `a`, dimensionless
wavelength or frequency `v`, and Doppler width `ΔD` (wavelength or frequency).
In the case of wavelength, v = (λ - λ0) / ΔD.
Uses Humlicek's W4 approximation. Returns in inverse units of ΔD.
"""
function voigt_profile(a::T, v::AbstractFloat, ΔD::T)::T where T <: AbstractFloat
    z = v + im * a
    w = humlizek(z)
    return real(w) / ΔD
end

"""
    voigt_profile(a::T, v::AbstractVector{<:AbstractFloat}, ΔD::T)::T
"""
function humlicek!(a::AbstractArray{T},
                v::AbstractArray{T},
                ans::CuDeviceVector{ComplexF32, 1}) where {T <: AbstractFloat}
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
