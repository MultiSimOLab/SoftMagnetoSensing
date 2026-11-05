#-----------------------------------------------------
#  Parametric study of the magneto-mechanical binary sensor
#
#  Sweep over the gripper width w ∈ [5, 7] (5 values) and the gap e ∈ [2, 3] (5 values), with
#  αgrip = 1.0 and particle concentrations c = 20 % and c = 30 %.
#
#  Run from the root folder of the project with N Julia threads:
#     julia --project=. -t N src/parametric_study.jl
#-----------------------------------------------------
using HyperFEM, HyperFEMKernels
using HyperFEM.TensorAlgebra: regularize_hessian
using Gridap, GridapGmsh
using Gridap.FESpaces, Gridap.CellData
using LinearAlgebra, JSON

# One BLAS thread: the sparse LU (UMFPACK) is dominated by small dense kernels, where a multi-threaded
# BLAS is slower and erratic (measured ×4 on this problem); the parallelism is in the assembly kernels
BLAS.set_num_threads(1)

const ROOT = joinpath(@__DIR__, "..")
include(joinpath(ROOT, "src", "gripper_sensor.jl"))
include(joinpath(ROOT, "data", "models", "mesh_generation.jl"))

meshdir = joinpath(ROOT, "data", "models")
simdir  = joinpath(ROOT, "data", "sims", "SmallMagnet_ParametricStudy")
mkpath(simdir)

αgrip = 1.0
w     = range(5, 7, 5)
e     = range(2, 3, 5)
cases = [(wi, ei, c) for c in (20.0, 30.0) for ei in e for wi in w]

# Cases solved at the same time (one Julia task each). The assembly kernels of all of them share the Julia
# threads, while the sparse LU of every case, which is serial, overlaps with the work of the others.
concurrent = max(1, Threads.nthreads() ÷ 2)

# Meshes (generated once; the ones already present are reused) and their discrete models, read here
# because the Gmsh library is not thread-safe
geometries = unique((c[1], c[2]) for c in cases)
for (wi, ei) in geometries
   isfile(mesh_file(wi, ei, αgrip; dir=meshdir)) || gripper_mesh(wi, ei, αgrip; dir=meshdir)
end
geomodels = Dict(g => GmshDiscreteModel(mesh_file(g..., αgrip; dir=meshdir)) for g in geometries)

semaphore = Base.Semaphore(concurrent)
@sync for (wi, ei, c) in cases
   Threads.@spawn begin
      Base.acquire(semaphore)
      try
         t = @elapsed run_case(geomodels[(wi, ei)], wi, ei, αgrip, c; T₀=0.1, uapplied=15.5, nsteps=5000, simdir, verbose=0)
         println("case w = $wi, e = $ei, c = $c %: done in $(round(t / 60; digits=1)) min")
      finally
         Base.release(semaphore)
      end
   end
end
