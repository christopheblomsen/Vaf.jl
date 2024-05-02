"""
Reading functions.
"""


"""
Reads atom in YAML format, returns AtomicModel structure
"""
function read_atom(atom_file; FloatT=Float64, IntT=Int)
    data = YAML.load_file(atom_file)
    element = Symbol(data["element"]["symbol"])
    Z = elements[element].number
    if "atomic_mass" in keys(data["element"])
        mass = _assign_unit(data["element"]["atomic_mass"]) |> u"kg"
    else
        mass = elements[element].atomic_mass |> u"kg"
    end
    levels = data["atomic_levels"]
    nlevels = length(levels)
    level_ids = collect(keys(levels))
    # Load and sort levels
    χ = [_assign_unit(level["energy"]) for (_, level) in levels]
    χ = Transparency.wavenumber_to_energy.(χ) .|> u"J"
    idx = sortperm(χ)  # argsort
    χ = χ[idx]
    g = [level["g"] for (_, level) in levels][idx]
    stage = [level["stage"] for (_, level) in levels][idx]
    label = [level["label"] for (_, level) in levels][idx]
    level_ids = level_ids[idx]
    # Continua
    ncont = length(data["radiative_bound_free"])
    continua = Vector{AtomicContinuum}(undef, ncont)
    for (index, cont) in enumerate(data["radiative_bound_free"])
        continua[index] = read_continuum(cont, χ, stage, level_ids;
                                         FloatT=FloatT, IntT=IntT)
    end
    # Lines
    nlines = length(data["radiative_bound_bound"])
    lines = Vector{AtomicLine}(undef, nlines)
    for (index, line) in enumerate(data["radiative_bound_bound"])
        lines[index] = read_line(line, χ, g, stage, level_ids, label, mass)
    end
    # ...
    # Collisions
    # ...
    # Package things nicely and convert to types
    return AtomicModel{nlevels, FloatT, IntT}(element, nlevels, nlines, ncont, Z,
                                              ustrip(mass), ustrip.(χ), g, stage,
                                              label, lines, continua)
end


"""
Reads continuum transition data in a Dict read from a YAML-formatted atom file.
Needs level energies χ, ionisation stages, and level ids from atom file.
"""
function read_continuum(cont::Dict, χ, stage, level_ids; FloatT=Float64, IntT=Int)
    λedge, up, lo = _read_transition(cont, χ, level_ids)
    # Explicit and hydrogenic cases
    if "cross_section" in keys(cont)
        tmp = reduce(hcat, cont["cross_section"]["value"])
        λ = (tmp[1, :] .* uparse(cont["cross_section"]["unit"][1])) .|> u"nm"
        σ = (tmp[2, :] .* uparse(cont["cross_section"]["unit"][2])) .|> u"m^2"
        idx = sortperm(λ)
        λ = λ[idx]
        σ = σ[idx]
        nλ = length(λ)
    elseif "cross_section_hydrogenic" in keys(cont)
        tmp = cont["cross_section_hydrogenic"]
        nλ = tmp["nλ"]
        λmin = _assign_unit(tmp["λ_min"]) |> u"nm"
        @assert λmin < λedge "Minimum wavelength not shorter than bf edge"
        σ0 = _assign_unit(tmp["σ_peak"])
        λ = LinRange(λmin, λedge, nλ)
        Z_eff = stage[up] - 1
        n_eff = Transparency.n_eff(χ[up], χ[lo], Z_eff) |> NoUnits
        σ = σ_hydrogenic_bf_scaled.(σ0, λ, λedge, Z_eff, n_eff) .|> u"m^2"
    else
        error("Photoionisation cross section data missing")
    end
    return AtomicContinuum{nλ, FloatT, IntT}(
        up, lo, nλ, ustrip(λedge), ustrip.(σ), ustrip.(λ))
end


"""
Reads spectral line data in a Dict read from a YAML-formatted atom file.
Needs level energies χ, ionisation stages, labels, level ids and
atomic mass from atom file.
"""
function read_line(line::Dict, χ, g, stage, level_ids, label, mass)
    λ0, up, lo = _read_transition(line, χ, level_ids)
    f_value = line["f_value"]
    Aul = calc_Aul(λ0, g[lo] / g[up], f_value)
    Bul = calc_Bul(λ0, Aul) |> u"m^3 / J"
    Blu = g[up] / g[lo] * Bul
    waves = line["wavelengths"]
    if "data" in keys(waves)
        λ = _assign_unit(waves["data"]) .|> u"nm"
        nλ = length(λ)
    elseif "type" in keys(waves)
        if waves["type"] == "RH"
            qcore = waves["qcore"]
            qwing = waves["qwing"]
            nλ = waves["nλ"]
            vξ = _assign_unit(waves["vmicro_char"])
            asymm = waves["asymmetric"]
            λ = calc_λline_RH(λ0, nλ, qcore, qwing, vξ; asymm=asymm)
            nλ = length(λ)  # when asymm=true, RH forces an even nλ
        elseif waves["type"] == "MULTI"
            q0 = waves["q0"]
            qmax = waves["qmax"]
            nλ = waves["nλ"]
            if iseven(nλ)   # ensure odd number of wavelengths per line
                nλ += 1
            end
            vξ = _assign_unit(waves["qnorm"])
            λ = calc_λline_MULTI(λ0, nλ, q0, qmax, vξ; asymm=true)
        else
            error("Unrecognised wavelength type")
        end
    end
    prof = lowercase(line["profile_type"])
    prd = false
    if prof in ["voigt", "prd"]
        voigt = true
        if prof == "prd"
            prd = true
        end
    elseif prof in ["gaussian", "gauss", "doppler"]
        voigt = false
    else
        error("Unsupported profile type $prof")
    end
    # Energy of the first ionised stage above upper level
    χ∞ = minimum(χ[stage .== stage[up] + 1])
    broadening = _read_broadening(line, mass, χ[up], χ[lo], χ∞, stage[up])
    return AtomicLine(
        nλ,
        ustrip(χ[up]),
        ustrip(χ[lo]),
        g[up],
        g[lo],
        ustrip(Aul),
        ustrip(Blu),
        ustrip(Bul),
        ustrip(λ0),
        f_value,
        ustrip(mass),
        ustrip.(λ),
        prd, voigt,
        label[up],
        label[lo],
        broadening,
    )
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

#=----------------------------------------------------------------------------
                            Utility functions
----------------------------------------------------------------------------=#
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
    λ0 = (h * c_0/ abs(χ[i] - χ[j])) |> u"nm"
    up = max(i, j)
    lo = min(i, j)
    return λ0, up, lo
end


function _read_broadening(data::Dict, mass, χup, χlo, χ∞, Z; T=Float64)
    γ_rad = zero(T)
    coeff = zeros(T, 0)
    temp_exp = zeros(T, 0)
    hydrogen_exp = zeros(T, 0)
    electron_exp = zeros(T, 0)

    electron_perturb = ["stark_quadratic"]
    hydrogen_perturb = ["vanderwaals_abo", "vanderwaals_unsold", "vanderwaals_deridder_rensbergen"]

    if "broadening" in lowercase.(keys(data))
        for mechanism in data["broadening"]
            keys_mech = lowercase.(keys(mechanism))
            @assert "type" in keys_mech "Missing broadening type"
            type = lowercase(mechanism["type"])
            if type == "natural"
                γ_rad = ustrip(_assign_unit(mechanism) |> u"s^-1")
            elseif type == "stark_linear_sutton"
                # Exponent on n_e, no dependence on temperature nor on n_H
                tmp_c, tmp_e = _read_broadening_single(mechanism, mass, χup, χlo, χ∞, Z)
                append!(coeff, tmp_c)
                append!(temp_exp, zero(T))
                append!(hydrogen_exp, zero(T))
                append!(electron_exp, tmp_e)
            elseif type in electron_perturb
                # Exponent on temperature, linear on n_e, no dependence on n_H
                tmp_c, tmp_e = _read_broadening_single(mechanism, mass, χup, χlo, χ∞, Z)
                append!(coeff, tmp_c)
                append!(temp_exp, tmp_e)
                append!(hydrogen_exp, zero(T))
                append!(electron_exp, one(T))
            elseif type in hydrogen_perturb
                # Exponent on temperature, linear on n_H, no dependence on n_E
                tmp_c, tmp_e = _read_broadening_single(mechanism, mass, χup, χlo, χ∞, Z)
                append!(coeff, tmp_c)
                append!(temp_exp, tmp_e)
                append!(hydrogen_exp, one(T))
                append!(electron_exp, zero(T))
            else
                @warn "Unsupported line broadening type $type, ignoring"
            end
        end
    end
    n_processes = length(coeff)
    coeff = SVector{n_processes, T}(coeff)
    temp_exp = SVector{n_processes, T}(temp_exp)
    hydrogen_exp = SVector{n_processes, T}(hydrogen_exp)
    electron_exp = SVector{n_processes, T}(electron_exp)
    broadening = LineBroadening{n_processes, T}(
        γ_rad,
        coeff,
        temp_exp,
        hydrogen_exp,
        electron_exp,
    )
    return broadening
end



"""
Reads individual broadening mechanisms and converts each to a multiplier
constant and temperature exponent based on different recipes.
"""
function _read_broadening_single(data::Dict, mass, χup, χlo, χ∞, Z)
    keys_input = lowercase.(keys(data))
    type = lowercase(data["type"])
    if type == "vanderwaals_unsold"
        if "h_coefficient" in keys_input
            h_scaling = data["h_coefficient"]
        else
            h_scaling = 0.0
        end
        if "he_coefficient" in keys_input
            he_scaling = data["he_coefficient"]
        else
            he_scaling = 0.0
        end
        tmp_const = const_unsold(mass, χup, χlo, χ∞, Z;
                                 H_scaling=h_scaling, He_scaling=he_scaling) * u"m^3 / s"
        tmp_exp = 0.3
    elseif type == "vanderwaals_abo"  # Anstee, Barklem, & O'Mara formalism
        α = data["α"]["value"]  # Trusting the units are in a_0^2
        σ = data["σ"]["value"]
        tmp_const = const_barklem(mass, α, σ)
        tmp_exp = (1 - α)/2
    elseif type == "vanderwaals_deridder_rensbergen"
        data = data["h"]
        α = data["α"]["value"] * ustrip(uparse(data["α"]["unit"]) * 1e8 |> u"cm^3/s")
        β = data["β"]
        h_mass = elements[:H].atomic_mass |> u"kg"
        # Assuming only perturbation by hydrogen
        tmp_const = const_deridder_rensbergen(mass, h_mass, α, β)
        tmp_exp = β
    elseif type == "stark_quadratic"
        if "coefficient" in keys_input
            coefficient = data["coefficient"]
        else
            coefficient = 1.0
        end
        if "c_4" in keys_input  # C_4 provided explicitly, no temp dependence
            C_4 = _assign_unit(data["c_4"])
            tmp_exp = 0.0
        else                    # Use C_4 recipe from Traving 1960, prop to T^1/6
            C_4 = const_stark_quadratic(mass, χup, χlo, χ∞, Z)
            tmp_exp = 1/6
        end
        tmp_const = coefficient * C_4
    elseif type == "stark_linear_sutton"
        n_u = data["n_upper"]
        n_l = data["n_lower"]
        tmp_const = data["coefficient"] * γ_stark_linear(1.0u"m^-3", n_u, n_l) * 1.0u"m^3"
        tmp_exp = 2/3
    else
        error("Unsupported type $type")
    end
    return (ustrip(tmp_const |> u"m^3 / s"), tmp_exp)
end
