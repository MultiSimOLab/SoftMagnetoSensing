# Finite element simulations of the binary sensor

Finite element model and parametric study of the magneto-mechanical binary sensor of

> J. Crespo-Miguel, R. Ortigosa, M. L. Lopez-Donaire, J. Pernas-Sanchez, J. Martinez-Frutos, D. Garcia-Gonzalez.
> *Piezoresistive Binary Sensors Through Residual Magnetisation.* Submitted to Nature Communications.

The database these simulations produce trains the neural-network surrogate of [`../NN_surrogate`](../NN_surrogate).

Two hard-magnetic grippers with residual magnetisation face each other across a gap inside a
deformable air domain. The device is compressed by a prescribed displacement; the magnetic attraction
between the grippers closes the gap, and the reaction forces on the loaded faces and the displacement
of the gap corners are recorded along the load path. The study sweeps the gripper width `w`, the gap
`e` and the particle concentration of the magnetic material.

The problem is a monolithic magneto-mechanical formulation (displacement and scalar magnetic
potential) solved with [HyperFEM](https://github.com/MultiSimOLab/HyperFEM.jl) on
[Gridap](https://github.com/gridap/Gridap.jl), with the assembly delegated to the compiled, threaded
kernels of [HyperFEMKernels](https://github.com/MultiSimOLab/HyperFEMKernels.jl).

## Contents

```
Project.toml / Manifest.toml      Julia environment (Julia ≥ 1.12, HyperFEM 0.2, HyperFEMKernels 0.1)
README.md
src/
  gripper_sensor.jl               sensor_model(...): constitutive models, weak forms, solver and
                                  post-processing of one configuration; run_case(...) solves it
  parametric_study.jl             driver of the study: meshes, 25 geometries × 2 concentrations
data/models/
  mesh_generation.jl              Gmsh geometry and mesh of the device (the .msh files are written here)
data/sims/                        results, one JSON file per configuration
bench/                            scripts of the comparison with the original version (see below)
```

## Installation

Julia 1.12 or later is required. From this folder (`FE_simulations`):

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

All the dependencies, HyperFEM and HyperFEMKernels included, are registered Julia packages.

## Running the study

```bash
julia --project=. -t auto src/parametric_study.jl
```

The script

1. generates the 25 meshes in `data/models/` (`w ∈ {5, 5.5, 6, 6.5, 7}`, `e ∈ {2, 2.25, 2.5, 2.75, 3}`,
   `αgrip = 1.0`), reusing the ones already present;
2. solves the 50 configurations (25 geometries × 2 concentrations, `c = 20 %` and `c = 30 %`), several at a
   time (`concurrent` in the script, half the number of Julia threads by default): the assembly kernels of
   all of them share the Julia threads (`-t N`) and the sparse LU of each case, which is serial, overlaps
   with the others;
3. writes the results to `data/sims/SmallMagnet_ParametricStudy/results_SmallDevice_w_*_e_*_αgrip_1.0_c_*.json`.

Every JSON file holds, per converged load step `step` (the load factor Λ ∈ [0, 1]), the reactions on
the top and bottom loaded faces `t_top`, `t_bottom` (vectors `[Rx, Ry]`) and the vertical displacement of
the four corners of the gap `uy_top_right`, `uy_bottom_right`, `uy_top_left`, `uy_bottom_left`.

A single configuration can be run from the REPL (`include` the `using` block and the two `include`
lines of `src/parametric_study.jl` first; `verbose=1` prints a line per load step, `verbose=2` every
Newton iteration):

```julia
geomodel = GmshDiscreteModel(gripper_mesh(5.0, 2.0, 1.0; dir="data/models"))
run_case(geomodel, 5.0, 2.0, 1.0, 20.0; simdir="data/sims", paraview=true, verbose=1)
```

With `paraview=true` a `.vtu` file (displacement, potential and Jacobian of the deformation) is
written at every converged step.

## Model and solver

- Grippers: Mooney–Rivlin hyperelasticity; the magnetic parts carry a hard-magnetic model with a
  remanent magnetisation that scales with the particle concentration, ramped up over the first 10 %
  of the load path. The soft (non-magnetic) part of the grippers is 20 times softer.
- Air: ideal magnetic medium; the Maxwell stress of the air acts on the solid as a traction on the
  solid–air interface, evaluated at the start of every load step. The mesh of the air follows the solid
  with a fictitious ARAP hyperelastic model stabilised by a gradient-jump penalty on its interior facets.
- The material tangents are projected (eigenvalues clamped from below) for robustness.
- Newton–Raphson with an injectivity-preserving line search (`det F > 0` at every quadrature point)
  and a sparse LU solver, 5000 load increments with bisection on failure.

## Comparison with the original scripts

The simulations of the paper were run with the original version of these scripts (Julia 1.10, HyperFEM
0.0.2, Gridap's lazy assembly). This version keeps the model, the mesh and the load path and changes the
software stack: HyperFEM 0.2 (new boundary-condition API, `NewtonRaphson` with the `InjectivityLS` line
search, `regularize_hessian` for the projected tangents), the compiled assembly kernels of HyperFEMKernels,
and a single BLAS thread for the sparse LU. `bench/run_case.jl` runs one configuration with a timing and
memory report, `bench/compare.py` compares two result files.

Measured on the configuration `w = 5, e = 2, c = 20 %` (31 683 dofs), 100 of the 5000 load steps, Apple M5 Max:

| | original (1 thread) | this version, `:lazy` (1 thread) | this version, `:compiled` (1 thread) | this version, `:compiled` (18 threads) |
|---|---|---|---|---|
| time per load step | 1.70 s | 1.16 s | 0.61 s | 0.53 s |
| Newton iterations per step | 1 | 2.2 | 2.2 | 2.2 |
| assembly per Newton iteration | ≈ 0.6 s | 0.39 s | 0.12 s | 0.076 s |
| LU factorisation per iteration | ≈ 0.5 s | 0.09 s | 0.11 s | 0.12 s |
| memory allocated per step | 2.6 GB | 1.3 GB | 0.6 GB | 0.6 GB |
| peak resident memory | 5.0 GB | 6.3 GB | 7.3 GB | 6.8 GB |

The Newton solver of the original scripts accepted every load step after a single iteration (its absolute
curvature criterion `|b·dx| < 1e-7` was always met at the force scale of the problem), with relative residuals
between 3·10⁻³ and 10⁻¹; this version iterates to the stated tolerance. With the same single iteration per step
(`newton = (maxiter=40, atol=1e-9, rtol=0.5)`) the two versions agree to 10⁻⁷ (displacements) and 2·10⁻⁵
(reactions) in relative terms. With the converged Newton the gap displacements agree with the original results
to 0.2 % and the reaction on the fixed face to 0.6 % over the magnetisation phase and the onset of compression
(Λ ≤ 0.113); the reaction on the loaded face is sensitive to the residual left by the Newton (see `newton`
in `sensor_model`).

## Citation

If you use these scripts, please cite the paper above (BibTeX entry in the [main README](../README.md#citation)).

## Contact

Jesús Martínez-Frutos (jesus.martinez@upct.es), MultiSimO Lab, Universidad Politécnica de Cartagena.
