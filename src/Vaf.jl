module Vaf

export Atmosphere1D, Atmosphere3D, GPUinfo
export AtomicLine, AtomicContinuum, AtomicModel
export RTBuffer
export AbstractBroadening, LineBroadening
export read_atom, h_ionfrac_saha
export saha_boltzmann, saha_boltzmann!
export AbstractExtinctionItp, ExtinctionItpLTE, ExtinctionItpNLTE
export α_cont, α_cont_no_itp, get_σ_itp
export piecewise_1D_nn, piecewise_1D_linear, piecewise_1D_linear!, feautrier
export read_atmos_rh, read_atmos_hpops_rh, read_pops_rh
export read_atmos_hpops_multi3d, read_atmos_multi3d, read_pops_multi3d
export doppler_width, damping, calc_broadening, create_voigt_itp
export blackbody_λ, voigt_humlicek
export calc_line_1D!, inner_loop!
export incline_atmos, incline_data!, incline_data_inv!, project_vector!
export addition!, subtract!, multiply!, divide!
export humlicek!, voigt_profile
export precalc_values!, calc_line_gpu!, calc_line_inclined_gpu!
export voigt_humlicek, incline_data_gpu!, incline_x!, incline_y!
export project_vlos_gpu!, project_kernel!
export gauss_quadrature, calc_J_global!, average_J!


using AtomicData
using CUDA
using Adapt
using AtomicData
using Interpolations
using PeriodicTable
using StaticArrays
using Transparency
using Unitful
using YAML
import PhysicalConstants.CODATA2018: h, k_B, R_∞, c_0, m_e, m_u, e, ε_0, a_0
import SpecialFunctions: erfcx
using ProgressMeter

# Write your package code here.

include("types.jl")
include("read_utils.jl")
include("lte.jl")
include("line.jl")
include("background.jl")
include("formal_solvers.jl")
include("read_atmos.jl")
include("utils.jl")
include("intensity.jl")
include("incline.jl")
include("gpu_arithmetics.jl")
include("gpu_solution.jl")
include("voigt.jl")
include("calc_line_gpu.jl")
include("gpu_incline.jl")

end
