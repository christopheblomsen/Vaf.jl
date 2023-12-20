# file for the init parameters for testing
using Vaf
using CUDA 
using Unitful 
const invSqrtPi = Float32(1/sqrt(π))
nx = ny = nz = 32*4;
α_c = CUDA.rand(Float32, nx, ny, nz);
j_c = CUDA.rand(Float32, nx, ny, nz);
source = CUDA.rand(Float32, nx, ny, nz);
α_tot = CUDA.zeros(Float32, nx, ny, nz);
profile = CUDA.zeros(Float32, nx, ny, nz);
temperature = CUDA.rand(Float32, nx, ny, nz);
α_c_1D = CUDA.rand(Float32, nx);
j_c_1D = CUDA.rand(Float32, nx);
source_1D = CUDA.rand(Float32, nx);
α_tot_1D = CUDA.zeros(Float32, nx);
profile_1D = CUDA.zeros(Float32, nx);
temperature_1D = CUDA.rand(Float32, nx);
c_0 = Float32(ustrip(Unitful.c0));
k_B = Float32(ustrip(Unitful.k));
λ0 = rand(Float32);
mass = rand(Float32);
λ = rand(Float32);
γ = CUDA.rand(Float32, nx, ny, nz);
γ_1D = CUDA.rand(Float32, nx);
threads = (32, 8, 1);
blocks = (nx÷threads[1], ny÷threads[2], nz÷threads[3]);

HOME_PATH = "/uio/hume/student-u67/chriskbl"
#include(joinpath(HOME_PATH, "Muspel.jl/scripts/int_from_pops_multi3d.jl"))
include(joinpath(HOME_PATH, "Documents/master/Vaf.jl/scripts/int_from_pops_multi3d.jl"))
#include("Muspel.jl/scripts/int_from_pops_multi3d.jl")

# /mn/stornext/u3/tiago/data/rhout/cb24bih/Halpha/s385/output 
BASE_DIR = "/mn/stornext/u3/tiago/data/rhout/cb24bih/Halpha/s385/output"
mesh_file = joinpath(BASE_DIR, "mesh.cb24bih_s0385")
atmos_file = joinpath(BASE_DIR, "atm3d.cb24bih_s0385")
pops_file = joinpath(BASE_DIR, "out_pop")
ATOM_DIR = joinpath(HOME_PATH, ".julia/packages/AtomicData/Po3lP/data/atoms")
atom_file = joinpath(AtomicData.get_atom_dir(), "H_6.yaml")
#atom_file = joinpath(ATOM_DIR, "H_6.yaml")
# read atom file manually
aa = read_atom(atom_file)
line = aa.lines[5]  #  index 5 for Ca II 854.2 nm

atmos, h_pops = read_atmos_hpops_multi3d(mesh_file, atmos_file, pops_file)
#atmos = read_atmos_multi3d(mesh_file, atmos_file)
#pops = read_pops_multi3d(pops_file, atmos.nx, atmos.ny, atmos.nz, aa.nlevels)
n_up = CuArray(h_pops[:, :, :, 5]);
n_lo = CuArray(h_pops[:, :, :, 3]);

n_up_1D = n_up[:, 1, 1];               
n_lo_1D = n_lo[:, 1, 1];

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
σ_itp = get_σ_itp(atmos, line.λ0, atom_files)
γ_energy = Float32(ustrip((Unitful.h * Unitful.c0 / (4 * π * line.λ0 * u"nm")) |> u"J"))
buf = RTBuffer(atmos.nz, line.nλ, Float32)
velocity_z = CuArray(atmos.velocity_z);
velocity_z_1D = CuArray(atmos.velocity_z[:, 1, 1]);
intensity = CuArray{Float32, 3}(undef, line.nλ, atmos.nx, atmos.ny)
# wavelength-independent part (continuum + broadening + Doppler width)
constants = GPUinfo(c_0, k_B, λ0, mass, γ_energy, 
                    Float32(line.Bul), Float32(line.Blu), Float32(line.Aul));

for i in 1:atmos.nz
    buf.α_c[i] = α_cont(
        σ_itp,
        atmos.temperature[i],
        atmos.electron_density[i],
        atmos.hydrogen1_density[i],
        atmos.proton_density[i]
    )
    buf.j_c[i] = buf.α_c[i] * blackbody_λ(line.λ0, atmos.temperature[i])
    buf.γ[i] = calc_broadening(
        line.γ,
        atmos.temperature[i],
        atmos.electron_density[i],
        atmos.hydrogen1_density[i]
    )
    # This can also go in the kernel
    buf.ΔλD[i] = doppler_width(line.λ0, line.mass, atmos.temperature[i])
end