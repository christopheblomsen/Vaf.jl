"""
Functions for reading model atmospheres from various formats.
"""

using HDF5
using Mmap
using FortranFiles
using DelimitedFiles


"""
Reads RH atmosphere. Returns always in single precision.
"""
function read_atmos_rh(atmos_file; index=1)
    temperature = h5read(atmos_file, "temperature", (:, :, :, index))
    electron_density = h5read(atmos_file, "electron_density", (:, :, :, index))
    hydrogen_density = h5read(atmos_file, "hydrogen_populations", (:, :, :, :, index))
    vz = h5read(atmos_file, "velocity_z", (:, :, :, index))
    z = h5read(atmos_file, "z", (:, index))
    x = h5read(atmos_file, "x")
    y = h5read(atmos_file, "y")

    nz, ny, nx, nhydr = size(hydrogen_density)
    # must define proton_density
    if nhydr == 1
        hydrogen1_density = hydrogen_density[:, :, :, 1]
        proton_density = similar(hydrogen1_density)
        Threads.@threads for i in eachindex(temperature)
            ionfrac = h_ionfrac_saha(temperature[i], electron_density[i])
            proton_density[i] = hydrogen1_density[i] * ionfrac
            hydrogen1_density[i] *= (1 - ionfrac)
        end
    elseif nhydr == 2
        hydrogen1_density = hydrogen_density[:, :, :, 1]
        proton_density = hydrogen_density[:, :, :, 2]
    elseif nhydr > 2
        proton_density = hydrogen_density[:, :, :, end]
        hydrogen1_density = dropdims(
            sum(view(hydrogen_density, :, :, :, 1:nhydr-1), dims=4);
            dims=4
        )
    end
    return Atmosphere1D(
        nx,
        ny,
        nz,
        z,
        temperature,
        vz,
        Float32.(electron_density),
        hydrogen1_density,
        proton_density
    )
end


"""
Reads RH atmosphere. Returns always in single precision.
"""
function read_atmos_hpops_rh(atmos_file, aux_file; index=1)
    temperature = h5read(atmos_file, "temperature", (:, :, :, index))
    electron_density = h5read(atmos_file, "electron_density", (:, :, :, index))
    #hydrogen_density = h5read(atmos_file, "hydrogen_populations", (:, :, :, :, index))
    vz = h5read(atmos_file, "velocity_z", (:, :, :, index))
    z = h5read(atmos_file, "z", (:, index))
    x = h5read(atmos_file, "x")
    y = h5read(atmos_file, "y")
    hydrogen_density = read_pops_rh(aux_file, "H")
    nz, ny, nx, nhydr = size(hydrogen_density)
    proton_density = hydrogen_density[:, :, :, end]
    hydrogen1_density = dropdims(
        sum(view(hydrogen_density, :, :, :, 1:nhydr-1), dims=4);
        dims=4
    )
    return Atmosphere1D(
        nx,
        ny,
        nz,
        z,
        temperature,
        vz,
        Float32.(electron_density),
        hydrogen1_density,
        proton_density
    ), hydrogen_density
end


"""
Reads array with populations for a given species.
"""
function read_pops_rh(aux_file, species)::Array{Float32, 4}
    atom = uppercase(species)
    try
        populations = h5read(aux_file, "atom_$atom/populations")
        return populations
    catch e
        if isa(e, KeyError)
            error("Could not find $species populations in $aux_file")
        else
            throw(e)
        end
    end
end


"""
Reads atmosphere in the input format of MULTI3D. Only Float32 atmospheres
are supported at the moment.
"""
function read_atmos_multi3d(mesh_file, atmos_file; grph=2.380491f-24)
    # Get parameters and height scale
    u_l = ustrip(1f0u"cm" |> u"m")
    u_v = ustrip(1f0u"km" |> u"m")
    nx::Int64, ny::Int64, nz::Int64, x, y, z = read_mesh(mesh_file)
    x .*= u_l
    y .*= u_l
    z .*= u_l
    # Get atmosphere and transpose
    fobj = open(atmos_file, "r")
    tmp = Array{Float32}(undef, nx, ny, nz)
    read!(fobj, tmp)
    electron_density = permutedims(tmp, (3, 2, 1))
    read!(fobj, tmp)
    temperature = permutedims(tmp, (3, 2, 1))
    read!(fobj, tmp)
    vx = permutedims(tmp, (3, 2, 1))
    read!(fobj, tmp)
    vy = permutedims(tmp, (3, 2, 1))
    read!(fobj, tmp)
    vz = permutedims(tmp, (3, 2, 1))
    read!(fobj, tmp)
    nH = permutedims(tmp, (3, 2, 1))
    close(fobj)
    proton_density = similar(temperature)

    # unit conversion and ion frac
    rho_to_nH = 1 / (grph * u_l^3)

    Threads.@threads for i in eachindex(temperature)
        electron_density[i] = electron_density[i] / u_l^3
        ionfrac = Muspel.h_ionfrac_saha(temperature[i], electron_density[i])
        proton_density[i] = nH[i] * rho_to_nH * ionfrac
        nH[i] *= rho_to_nH * (1 - ionfrac)
        vx[i] *= u_v
        vy[i] *= u_v
        vz[i] *= u_v
    end

    return Atmosphere3D(
        nx,
        ny,
        nz,
        x,
        y,
        z,
        temperature,
        vx,
        vy,
        vz,
        electron_density,
        nH,
        proton_density,
    )
end


"""
Reads atmosphere in the input format of MULTI3D, at the same time as the
hydrogen populations. Only works for a H NLTE run. Only Float32 files
are supported at the moment.
"""
function read_atmos_hpops_multi3d(
        mesh_file, atmos_file, hpops_file;
        nlevels=6, grph=2.380491f-24
)
    # Get parameters and height scale
    u_l = ustrip(1f0u"cm" |> u"m")
    u_v = ustrip(1f0u"km" |> u"m")
    nx::Int64, ny::Int64, nz::Int64, x, y, z = read_mesh(mesh_file)
    x .*= u_l
    y .*= u_l
    z .*= u_l
    # Get hydrogen populations
    h_pops = Array{Float32}(undef, nx, ny, nz, nlevels)
    read!(hpops_file, h_pops)
    Threads.@threads for i in eachindex(h_pops)
        h_pops[i] = h_pops[i] / u_l^3
    end
    h1_pops = sum(h_pops[:, :, :, 1:end-1], dims=4)[:, :, :, 1]
    # Get atmosphere and transpose
    fobj = open(atmos_file, "r")
    tmp = Array{Float32}(undef, nx, ny, nz)
    read!(fobj, tmp)
    electron_density = permutedims(tmp, (3, 2, 1))
    read!(fobj, tmp)
    temperature = permutedims(tmp, (3, 2, 1))
    read!(fobj, tmp)
    vx = permutedims(tmp, (3, 2, 1))
    read!(fobj, tmp)
    vy = permutedims(tmp, (3, 2, 1))
    read!(fobj, tmp)
    vz = permutedims(tmp, (3, 2, 1))
    close(fobj)
    proton_density = permutedims(h_pops[:, :, :, end], (3, 2, 1))
    HI_density = permutedims(h1_pops, (3, 2, 1))

    Threads.@threads for i in eachindex(temperature)
        electron_density[i] /= u_l^3
        vx[i] *= u_v
        vy[i] *= u_v
        vz[i] *= u_v
    end

    atm = Atmosphere3D(
        nx,
        ny,
        nz,
        x,
        y,
        z,
        temperature,
        vx,
        vy,
        vz,
        electron_density,
        HI_density,
        proton_density,
    )
    return atm, PermutedDimsArray(h_pops, (3, 2, 1, 4))
end


"""
Reads NLTE populations from MULTI3D output. Does NOT permute dims. Only Float32
files are supported at the moment.
"""
function read_pops_multi3d(pop_file, nx, ny, nz, nlevels)::Array{Float32, 4}
    u_l = ustrip(1f0u"cm" |> u"m")
    pops = Array{Float32}(undef, nx, ny, nz, nlevels)
    read!(pop_file, pops)
    Threads.@threads for i in eachindex(pops)
        pops[i] /= u_l^3
    end
    return PermutedDimsArray(pops, (3, 2, 1, 4))
end


"""
Reads mesh file from Bifrost or MULTI3D.
"""
function read_mesh(mesh_file)
    # Read all values into a single 1D array
    tmp::Vector{Float32} = Float32.(
        [a for a in vec(permutedims(readdlm(mesh_file))) if a != ""]
    )
    inc = 1
    nx = Int64(tmp[inc])
    inc += 1
    x = tmp[inc:inc + nx - 1]
    inc += nx
    ny = Int64(tmp[inc])
    inc += 1
    y = tmp[inc:inc + ny - 1]
    inc += ny
    nz = Int64(tmp[inc])
    inc += 1
    z = tmp[inc:end]
    return (nx, ny, nz, x, y, z)
end
