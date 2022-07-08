"""
Reading functions.
"""

"""
Return a value with a unit given a dictionary read from a YAML file.
"""
_assign_unit(data::Dict) = data["value"] * uparse(data["unit"])


"""
Helper function to parse transition upper and lower levels and compute
wavelength. To be used with fields from the YAML atom format.
"""
function _read_transition(data::Dict, χ, level_ids)
    nlevels = length(χ)
    i = (1:nlevels)[level_ids .== data["transition"][1]][1]
    j = (1:nlevels)[level_ids .== data["transition"][2]][1]
    λ0 = ustrip(h * c_0/ (abs(χ[i] - χ[j]))u"J" |> u"nm")
    up = max(i, j)
    lo = min(i, j)
    return λ0, up, lo
end


"""
Calculate line wavelengths using recipe from RH.
"""
function calc_λline_RH(λ0::Unitful.Length{T}, nλ, qcore::T, qwing::T, vξ::Unitful.Velocity{T};
                       asymm=true) where T <: Real
    q_to_λ = convert(typeof(λ0), (λ0 * vξ / c_0))
    nhalf = nλ ÷ 2
    if !asymm
        nhalf *= 2
    end
    if qwing <= 2 * qcore
        β = one(T)
    else
        β = qwing / (2 * qcore)
    end
    y = β + sqrt(β * β + (β - 1) * nhalf + 2 - 3 * β)
    b = 2 * log(y) / (nhalf - 1)
    a = qwing / (nhalf - 2 + y*y)
    Δλ = a * ((1:nhalf) .+ exp.(b * ((1:nhalf) .- 1))) .* q_to_λ
    if asymm
        λ = vcat(reverse(λ0 .- Δλ), λ0 .+ Δλ)
    else
        λ = λ0 .+ Δλ
    end
    return λ
end


"""
Calculate line wavelengths using recipe from MULTI.
"""
function calc_λline_MULTI(λ0::Unitful.Length{T}, nλ, q0::T, qmax::T, vξ::Unitful.Velocity{T};
                          asymm=true) where T <: Real
    ν0 = convert(typeof(one(T) * u"Hz"), (c_0 / λ0))
    q = Vector{T}(undef, nλ)

    ten = 10 * one(T)
    al10 = log(ten)
    half = one(T) / 2
    a = ten ^ (q0 + half)
    xmax = log10(a * max(half, qmax - q0 - half))
    if qmax <= q0
        # Linear spacing
        dq = 2 * qmax / (nλ - 1)
        q[1] = -qmax
        for i in 2:nλ
            q[i] = q[i - 1] + dq
        end
    elseif (qmax >= 0) & (q0 >=0)
        if asymm
            dx = 2 * xmax / (nλ - 1)
            for i in 1:nλ
                x = -xmax + (i - 1) * dx
                x10 = ten ^ x
                q[i] = x + (x10 - 1 / x10) / a
            end
        else
            dx = xmax / (nλ - 1)
            for i in 1:nλ
                x = (i - one(T)) * dx
                x10 = ten ^ x
                # Set negative (contrary to MULTI) to ensure consistency
                # with the RH values, which increase from line centre
                q[i] = -(x + (x10 - one(T)) / a)
            end
        end
    end
    ν = ν0 * (1 .+ q * vξ / c_0)
    λ = sort!(convert.(typeof(λ0), c_0 ./ ν))
    return λ
end



"""
Reads atom in YAML format, returns AtomicModel structure
"""
function read_atom(atom_file)
    data = YAML.load_file(atom_file)
    element = Symbol(data["element"]["symbol"])
    Z = elements[element].number
    mass = ustrip(elements[element].atomic_mass |> u"kg")
    levels = data["atomic_levels"]
    nlevels = length(levels)
    level_ids = collect(keys(levels))
    # Load and sort levels
    χ = [_assign_unit(level["energy"]) for (_, level) in levels]
    χ = ustrip(Transparency.wavenumber_to_energy.(χ) .|> u"J")
    idx = sortperm(χ)  # argsort
    FloatT = eltype(χ)
    IntT = typeof(Z)
    χ = SVector{nlevels, FloatT}(χ[idx])
    g = SVector{nlevels, IntT}([level["g"] for (_, level) in levels][idx])
    stage = SVector{nlevels, IntT}([level["stage"] for (_, level) in levels][idx])
    label = [level["label"] for (_, level) in levels][idx]
    level_ids = level_ids[idx]
    # Continua
    ncont = length(data["radiative_bound_free"])
    continua = Vector{AtomicContinuum}(undef, ncont)
    for (index, cont) in enumerate(data["radiative_bound_free"])
        continua[index] = read_continuum(cont, χ, stage, level_ids)
    end
    # Lines
    # ...
    # Collisions
    # ...
    return AtomicModel{nlevels, FloatT, IntT}(element, nlevels, 1, ncont, Z, mass,
                                              χ, g, stage, label, continua)
end


"""
Reads continuum transition data in a Dict read from a YAML-formatted atom file.
Needs level energies χ, ionisation stages, and level ids from atom file.
"""
function read_continuum(cont::Dict, χ, stage, level_ids)
    λedge, up, lo = _read_transition(cont, χ, level_ids)
    # Explicit and hydrogenic cases
    if "cross_section" in keys(cont)
        tmp = reduce(hcat, cont["cross_section"]["value"])
        λ = ustrip.((tmp[1, :] .* uparse(cont["cross_section"]["unit"][1])) .|> u"nm")
        σ = ustrip.((tmp[2, :] .* uparse(cont["cross_section"]["unit"][2])) .|> u"m^2")
        idx = sortperm(λ)
        λ = λ[idx]
        σ = σ[idx]
    elseif "cross_section_hydrogenic" in keys(cont)
        tmp = cont["cross_section_hydrogenic"]
        nλ = tmp["nλ"]
        λmin = ustrip(_assign_unit(tmp["λ_min"]) |> u"nm")
        @assert λmin < λedge "Minimum wavelength not shorter than bf edge"
        σ0 = _assign_unit(tmp["σ_peak"])
        λ = LinRange(λmin, λedge, nλ)
        Z_eff = stage[up] - 1
        n_eff = Transparency.n_eff(χ[up] * u"J", χ[lo] * u"J", Z_eff)
        σ = hydrogenic_bf_σ_scaled.(σ0, λ * u"nm", λedge * u"nm", Z_eff, n_eff)
        σ = ustrip(σ .|> u"m^2")
    else
        error("Photoionisation cross section data missing")
    end
    return AtomicContinuum(λedge, up, lo, σ, λ)
end
