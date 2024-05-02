"""
Collection of types.
"""

abstract type AbstractAtmos{T <: Real} end

abstract type AbstractAtmos1D{N, T <: Real} <: AbstractAtmos{T} end

abstract type AbstractAtmos3D{T <: Real} <: AbstractAtmos{T} end


"""
Type for 1D atmospheres. Can contain both 1D only, or 1.5D atmospheres (multiple
columns of 1D atmospheres).
"""
struct Atmosphere1D{
    N,
    T <: Real,
    A <: AbstractArray{T, N},
    V <:AbstractVector{T},
} <: AbstractAtmos1D{N, T}
    nx::Int64
    ny::Int64
    nz::Int64
    z::V
    temperature::A
    velocity_z::A
    electron_density::A
    hydrogen1_density::A  # neutral hydrogen across all levels
    proton_density::A
end


struct Atmosphere3D{
    T <: Real,
    A <: AbstractArray{T, 3},
    V <:AbstractVector{T}
} <: AbstractAtmos3D{T}
    nx::Int64
    ny::Int64
    nz::Int64
    x::V
    y::V
    z::V
    temperature::A
    velocity_x::A
    velocity_y::A
    velocity_z::A
    electron_density::A
    hydrogen1_density::A  # neutral hydrogen across all levels
    proton_density::A
end

Base.getindex(a::AbstractAtmos1D{3}, i, j) = Atmosphere1D(
    1,
    1,
    a.nz,
    a.z,
    a.temperature[:, i, j],
    a.velocity_z[:, i, j],
    a.electron_density[:, i, j],
    a.hydrogen1_density[:, i, j],
    a.proton_density[:, i, j],
)

Base.getindex(a::AbstractAtmos1D{2}, i) = Atmosphere1D(
    1,
    1,
    a.nz,
    a.z,
    a.temperature[:, i],
    a.velocity_z[:, i],
    a.electron_density[:, i],
    a.hydrogen1_density[:, i],
    a.proton_density[:, i],
)

Base.getindex(a::AbstractAtmos3D, i, j) = Atmosphere1D(
    1,
    1,
    a.nz,
    a.z,
    a.temperature[:, i, j],
    a.velocity_z[:, i, j],
    a.electron_density[:, i, j],
    a.hydrogen1_density[: ,i, j],
    a.proton_density[:, i, j],
)


abstract type AbstractBroadening{T <: AbstractFloat} end


struct LineBroadening{N, T} <: AbstractBroadening{T}
    natural::T
    coeff::SVector{N, T}
    temp_exp::SVector{N, T}
    hydrogen_exp::SVector{N, T}
    electron_exp::SVector{N, T}
end


struct AtomicLine{
    N,
    FloatT <: AbstractFloat,
    IntT <: Integer,
    V <:AbstractVector{FloatT},
    S <: Union{Nothing, String},
}
    nλ::IntT
    χup::FloatT
    χlo::FloatT
    gup::IntT
    glo::IntT
    Aul::FloatT
    Blu::FloatT
    Bul::FloatT
    λ0::FloatT  # in nm
    f_value::FloatT
    mass::FloatT
    λ::V
    PRD::Bool
    Voigt::Bool
    label_up::S
    label_lo::S
    γ::LineBroadening{N, FloatT}
end


struct AtomicContinuum{Nλ, FloatT <: AbstractFloat, IntT <: Integer}
    up::IntT
    lo::IntT
    nλ::IntT
    λedge::FloatT  # in nm
    σ::SVector{Nλ, FloatT}  # m^-2
    λ::SVector{Nλ, FloatT}  # nm
end


struct AtomicModel{Nlevel, FloatT <: AbstractFloat, IntT <: Integer}
    element::Symbol
    nlevels::IntT
    nlines::IntT
    ncontinua::IntT
    Z::IntT
    mass::FloatT
    χ::SVector{Nlevel, FloatT}  # Energy in J or aJ?
    g::SVector{Nlevel, IntT}
    stage::SVector{Nlevel, IntT}
    label::Vector{String}
    lines::Vector{AtomicLine}
    continua::Vector{AtomicContinuum}
end


struct RTBuffer{T <: AbstractFloat}
    ndep::Int
    nλ::Int
    intensity::Vector{T}
    source_function::Vector{T}
    α_total::Vector{T}
    α_c::Vector{T}
    j_c::Vector{T}
    ΔλD::Vector{T}
    γ::Vector{T}
    int_tmp::Vector{T}
    function RTBuffer(ndep, nλ, f_type)
        T = f_type
        intensity = Vector{T}(undef, nλ)
        source_function = Vector{T}(undef, ndep)
        α_total = Vector{T}(undef, ndep)
        α_c = Vector{T}(undef, ndep)
        j_c = Vector{T}(undef, ndep)
        ΔλD = Vector{T}(undef, ndep)
        γ = Vector{T}(undef, ndep)
        int_tmp = Vector{T}(undef, ndep)
        new{T}(ndep, nλ, intensity, source_function, α_total, α_c, j_c, ΔλD, γ, int_tmp)
    end
end
