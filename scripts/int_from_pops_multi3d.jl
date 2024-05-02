using Vaf
using AtomicData
using HDF5
using ProgressMeter
using Base.Threads


"""
Sample script to calculate Ca II 8542 profiles with added microturbulence,
using populations from a Multi3D run. The `atmos_file` should point to
a Multi3D *input* atmosphere file (`atmos3d.*`), not an output atmosphere file.

Assumes an atom with a similar structure to CaII_PRD.yaml is used.
"""
function calc_multi3d_8542(mesh_file, atmos_file, pops_file, atom_file)
    ca_atom = read_atom(atom_file)
    my_line = ca_atom.lines[5]  #  index 5 for Ca II 854.2 nm

    atmos = read_atmos_multi3d(mesh_file, atmos_file)
    pops = read_pops_multi3d(pops_file, atmos.nx, atmos.ny, atmos.nz, ca_atom.nlevels)
    n_u = pops[:, :, :, 5]
    n_l = pops[:, :, :, 3]

    # Continuum opacity structures
    bckgr_atoms = [
        "Al.yaml",
        "C.yaml",
        "Ca.yaml",
        "Fe.yaml",
        "H_6.yaml",
        "He.yaml",
        "KI.yaml",
        "Mg.yaml",
        "N.yaml",
        "Na.yaml",
        "NiI.yaml",
        "O.yaml",
        "S.yaml",
        "Si.yaml",
    ]
    atom_files = [joinpath(AtomicData.get_atom_dir(), a) for a in bckgr_atoms]
    σ_itp = get_σ_itp(atmos, my_line.λ0, atom_files)

    a = LinRange(1f-4, 1f1, 20000)
    v = LinRange(0f0, 5f2, 2500)
    voigt_itp = create_voigt_itp(a, v)

    intensity = Array{Float32, 3}(undef, my_line.nλ, atmos.nx, atmos.ny)
    p = ProgressMeter.Progress(atmos.nx)

    Threads.@threads for i in 1:atmos.nx
        buf = RTBuffer(atmos.nz, my_line.nλ, Float32)  # allocate inside for local scope
        for j in 1:atmos.ny
            calc_line_1D!(my_line, buf, atmos[j, i], n_u[:, j, i], n_l[:, j, i], σ_itp, voigt_itp)
            intensity[:, j, i] = buf.intensity
        end
        ProgressMeter.next!(p)
    end

    return intensity
end


"""
Sample script to calculate Ca II 8542 profiles with added microturbulence,
using populations from a Multi3D run. The `atmos_file` should point to
a Multi3D *input* atmosphere file (`atmos3d.*`), not an output atmosphere file.

The difference from the previous function is that the NLTE hydrogen
populations are read at the same time as the atmosphere, and used to get the
densities of H I and H II that go into the background opacity calculations.

Assumes an atom with a similar structure to CaII_PRD.yaml is used.
"""
function calc_multi3d_hα(mesh_file, atmos_file, pops_file, atom_file)
    h_atom = read_atom(atom_file)
    my_line = h_atom.lines[5]  #  index 5 for Halpha

    atmos, h_pops = read_atmos_hpops_multi3d(mesh_file, atmos_file, pops_file)
    n_u = h_pops[:, :, :, 3]
    n_l = h_pops[:, :, :, 2]

    # Continuum opacity structures
    bckgr_atoms = [
        "Al.yaml",
        "C.yaml",
        "Ca.yaml",
        "Fe.yaml",
        "H_6.yaml",
        "He.yaml",
        "KI.yaml",
        "Mg.yaml",
        "N.yaml",
        "Na.yaml",
        "NiI.yaml",
        "O.yaml",
        "S.yaml",
        "Si.yaml",
    ]
    atom_files = [joinpath(AtomicData.get_atom_dir(), a) for a in bckgr_atoms]
    σ_itp = get_σ_itp(atmos, my_line.λ0, atom_files)

    a = LinRange(1f-4, 1.5f1, 20000)
    v = LinRange(0f2, 5f2, 2500)
    voigt_itp = create_voigt_itp(a, v)

    intensity = Array{Float32, 3}(undef, my_line.nλ, atmos.nx, atmos.ny)
    p = ProgressMeter.Progress(atmos.nx)

    Threads.@threads for i in 1:atmos.nx
        buf = RTBuffer(atmos.nz, my_line.nλ, Float32)  # allocate inside for local scope
        for j in 1:atmos.ny
            calc_line_1D!(my_line, buf, atmos[j, i], n_u[:, j, i], n_l[:, j, i], σ_itp, voigt_itp)
            intensity[:, j, i] = buf.intensity
        end
        ProgressMeter.next!(p)
    end

    return intensity
end
