# One configuration with a timing and memory report:
#   julia --project=. -t N bench/run_case.jl iw ie concentration nsteps backend nmax [meshdir]
using HyperFEM, HyperFEMKernels
using HyperFEM.TensorAlgebra: regularize_hessian
using Gridap, GridapGmsh
using Gridap.FESpaces, Gridap.CellData
using LinearAlgebra, JSON

# One BLAS thread: the sparse LU (UMFPACK) is dominated by small dense kernels, where a multi-threaded
# BLAS is slower and erratic (measured ×4 on this problem); the parallelism is in the assembly kernels
BLAS.set_num_threads(1)

include(joinpath(@__DIR__, "..", "src", "gripper_sensor.jl"))
include(joinpath(@__DIR__, "..", "data", "models", "mesh_generation.jl"))

iw, ie  = parse(Int, ARGS[1]), parse(Int, ARGS[2])
conc    = parse(Float64, ARGS[3])
nsteps  = parse(Int, ARGS[4])
backend = Symbol(ARGS[5])
nmax    = parse(Int, ARGS[6])        # stop after this many converged steps
meshdir = length(ARGS) >= 7 ? ARGS[7] : joinpath(@__DIR__, "..", "data", "models")
rtol    = parse(Float64, get(ENV, "SMS_RTOL", "1e-6"))   # Newton relative tolerance
grow    = parse(Float64, get(ENV, "SMS_GROW", "2.0"))    # growth of the load increment after an easy step
easy    = parse(Int, get(ENV, "SMS_EASY", "5"))         # Newton iterations of an easy step
proj    = get(ENV, "SMS_PROJ", "1") == "1"               # Dirichlet projection (predictor) at every step
lmax    = parse(Float64, get(ENV, "SMS_LMAX", "1.0"))    # stop at this load factor
paraview = get(ENV, "SMS_VTK", "0") == "1"               # write the .vtu files of every step
tangent = Symbol(get(ENV, "SMS_TANGENT", "hybrid"))   # :projected or :hybrid

e = range(2, 3, 5); w = range(5, 7, 5); αgrip = 1.0
simdir = joinpath(@__DIR__, "..", "data", "sims", "bench_$(backend)_t$(Threads.nthreads())_nsteps$(nsteps)_nmax$(nmax)_rtol$(rtol)_grow$(grow)_easy$(easy)_proj$(Int(proj))_lmax$(lmax)_$(tangent)")
mkpath(simdir)
geomodel = GmshDiscreteModel(mesh_file(w[iw], e[ie], αgrip; dir=meshdir))

struct StopRun <: Exception end
step_times = Float64[]
verbose = parse(Int, get(ENV, "SMS_VERBOSE", "1"))
function stop_after(step, Λ)
   push!(step_times, time())
   flush(stdout)
   (step >= nmax || Λ >= lmax - 1e-12) && throw(StopRun())
end

GC.gc()
t0 = time()
m = sensor_model(geomodel, w[iw], e[ie], αgrip, conc; simdir, backend, verbose, callback=stop_after,
                 newton=(maxiter=40, atol=1e-9, rtol), tangent, paraview)
t = @timed try
   solve!(m.comp_model; stepping=LoadStepping(; nsteps, maxbisec=10, grow, easy), ProjectDirichlet=proj, post=m.post)
catch err
   err isa StopRun || rethrow()
end
HyperFEM.Solvers.print_summary(m.nls.log)
paraview && HyperFEM.ComputationalModels.vtk_save(m.post)
open(io -> JSON.print(io, step_times .- t0), joinpath(simdir, "step_times.json"), "w")
n = length(step_times)
println("=== steps=$n  t_first_step_s=$(step_times[1]-t0)  t_per_step_after_first_s=$((step_times[end]-step_times[1])/max(n-1,1))")
println("=== NEW backend=$backend case iw=$iw ie=$ie c=$conc nsteps=$nsteps threads=$(Threads.nthreads()) rtol=$rtol grow=$grow easy=$easy proj=$proj lmax=$lmax tangent=$tangent paraview=$paraview")
println("=== time_s=$(t.time) alloc_GB=$(t.bytes/1e9) gctime_s=$(t.gctime) maxrss_GB=$(Sys.maxrss()/1e9)")
println("=== julia=$(VERSION) HyperFEM=$(pkgversion(HyperFEM)) HyperFEMKernels=$(pkgversion(HyperFEMKernels)) Gridap=$(pkgversion(Gridap))")
