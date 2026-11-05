#***********************************************************************************************
# Magneto-mechanical binary sensor: two hard-magnetic grippers with residual magnetisation facing
# each other across a gap, embedded in a deformable vacuum (air) domain. The device is compressed by
# a prescribed displacement; the magnetic attraction between the grippers closes the gap (contact is
# resolved through the deformable air layer) and the reaction forces on the top and bottom faces
# are recorded.
#
# Monolithic magneto-mechanical formulation (displacement u, scalar magnetic potential φ) solved
# with HyperFEM on a Gridap mesh. The Maxwell stress of the air acts on the solid as a traction on
# the solid–air interface, updated at the start of every load step (staggered within the step).
# The load path is followed with an adaptive increment (bisection on failure, growth after easy steps)
# and a Dirichlet predictor; the Newton uses the consistent tangent, with a projected (positive-definite)
# mechanical tangent only as a fallback when the line search fails.
#
# "Piezoresistive Binary Sensors Through Residual Magnetisation" (submitted to Nature Communications)
#***********************************************************************************************

"""
    sensor_model(geomodel, w, e, αgrip, concentration; T₀=0.1, uapplied=15.5, simdir, backend=:compiled,
                 paraview=false, verbose=1, callback=nothing, newton=(maxiter=40, atol=1e-9, rtol=1e-6),
                 tangent=:hybrid, atol_projected=1e-8)

Build the finite element model of one sensor configuration on the Gmsh model `geomodel` (see
`data/models/mesh_generation.jl`): gripper width `w`, gap `e`, loaded-height parameter `αgrip` and
particle concentration `concentration` (in %). The remanent magnetisation ramps up over the first
fraction `T₀` of the load path; afterwards the compression `uapplied` is applied with a smooth ramp.

Returns a named tuple with the computational model `comp_model` (a `StaticNonlinearModel` of HyperFEM),
its Newton solver `nls`, the post-processor `post` (reactions on the loaded faces and vertical
displacement of the gap corners, written after every converged load step to the JSON file `jsonfile`
in `simdir`; with `paraview=true` a `.vtu` file per step, collected in `Results.pvd`, with the displacement,
the magnetic potential, the spatial magnetic field `h = F⁻ᵀℋ₀`, its magnitude and the material region
of every cell, see `data/visualization/`), the multi-field space `U` and the state `xh`.

`backend` is the assembly backend of HyperFEM (`:compiled` for the fused, threaded kernels of
HyperFEMKernels, `:lazy` for Gridap's assembly); `newton` the stopping criteria of the Newton–Raphson
solver (`‖R‖ < atol` or `‖R‖/‖R₀‖ < rtol`, at most `maxiter` iterations); `tangent` the mechanical
tangents of the Jacobian: `:hybrid` (consistent tangents, switched to the projected ones only when the
line search fails: `HybridTangent` of HyperFEM, quadratic convergence; in projected mode the Newton stops
at `‖R‖ < atol_projected`, above the round-off floor of the residual of the compressed device) or
`:projected` (eigenvalues clamped from below at every quadrature point, robust but linearly convergent). `callback(step, Λ)`, if given, is
called after every converged load step (e.g. to monitor or stop a run).
"""
function sensor_model(geomodel, w, e, αgrip, concentration;
                      T₀=0.1, uapplied=15.5, simdir=pwd(), backend=:compiled, paraview=false, verbose=1,
                      callback=nothing, newton=(maxiter=40, atol=1e-9, rtol=1e-6), tangent=:hybrid,
                      atol_projected=1e-8)
   #********************
   # Constitutive models
   #********************
   μ0     = 4π * 1e-4                 # magnetic permeability of vacuum (units of the paper)
   αr_max = 3.0e-3 / μ0               # remanent magnetisation of the reference (20 %) material
   μ      = 1e-6 * 12000              # shear modulus of the magnetic material (20 %)
   f      = 1 / 20                    # stiffness ratio of the soft (non-magnetic) part of the grippers
   μv     = μ * 1e-4                  # fictitious stiffness of the air

   # Air: ideal (linear) magnetic medium; Maxwell stress from the magneto-mechanical model with no
   # mechanical energy; fictitious ARAP mechanics for the mesh motion
   model_air_mag     = IdealMagnetic2D(μ0=μ0, χe=0.0)
   model_air_maxwell = MagnetoMechModel(magneto=model_air_mag, mechano=ARAP2D(μ=0.0))
   model_air_mech    = ARAP2D(μ=μv)

   # Solid 1: soft, non-magnetic part of the grippers
   μ1           = μ * f
   model_solid1 = MagnetoMechModel(magneto=model_air_mag,
                                   mechano=NonlinearMooneyRivlin2D(λ=μ1 * 1e2, μ1=0.0, μ2=μ1, α1=1.0, α2=1.0))
   # Solid 2: hard-magnetic material at the reference concentration (20 %)
   μ2           = μ
   model_mag2   = HardMagnetic2D(μ0=μ0, αr=αr_max, χe=0.0, χr=8.0)
   model_solid2 = MagnetoMechModel(magneto=model_mag2,
                                   mechano=NonlinearMooneyRivlin2D(λ=μ2 * 1e2, μ1=0.0, μ2=μ2, α1=1.0, α2=1.0))
   # Solid 3: hard-magnetic material at the study concentration (stiffness and remanence scale with it)
   μ3           = μ * (0.08 * concentration - 0.6)
   αr3          = αr_max * (concentration / 20.0)
   model_mag3   = HardMagnetic2D(μ0=μ0, αr=αr3, χe=0.0, χr=8.0)
   model_solid3 = MagnetoMechModel(magneto=model_mag3,
                                   mechano=NonlinearMooneyRivlin2D(λ=μ3 * 1e2, μ1=0.0, μ2=μ3, α1=1.0, α2=1.0))

   #**********************
   # Finite elements and domains
   #**********************
   order  = 1
   reffeφ = ReferenceFE(lagrangian, Float64, order)
   reffeu = ReferenceFE(lagrangian, VectorValue{2,Float64}, order)

   Ωdomain    = Triangulation(geomodel, tags=["Air", "MagnetoOmega1", "MagnetoOmega2", "MagnetoOmega3"])
   Ωair       = Triangulation(geomodel, tags=["Air"])
   Ωsolid     = Triangulation(geomodel, tags=["MagnetoOmega1", "MagnetoOmega2", "MagnetoOmega3"])
   Ωsolid_mec = Triangulation(geomodel, tags=["MechanoOmega1", "MechanoOmega2", "MechanoOmega3"])
   Ωmec1, Ωmec2, Ωmec3 = (Triangulation(geomodel, tags=[t]) for t in ("MechanoOmega1", "MechanoOmega2", "MechanoOmega3"))
   Ωmag1, Ωmag2, Ωmag3 = (Triangulation(geomodel, tags=[t]) for t in ("MagnetoOmega1", "MagnetoOmega2", "MagnetoOmega3"))

   degree   = 2 * order
   dΩdomain = Measure(Ωdomain, degree)
   dΩmec1, dΩmec2, dΩmec3 = Measure(Ωmec1, degree), Measure(Ωmec2, degree), Measure(Ωmec3, degree)
   dΩmag1, dΩmag2, dΩmag3 = Measure(Ωmag1, degree), Measure(Ωmag2, degree), Measure(Ωmag3, degree)
   dΩair    = Measure(Ωair, degree)

   # Solid–air interface (traction of the Maxwell stress) and interior facets of the air (gradient-jump
   # stabilisation of the fictitious air mechanics)
   Γsolid_int = BoundaryTriangulation(Ωsolid, tags="Interface")
   dΓs        = Measure(Γsolid_int, 4 * order)
   Γsk        = SkeletonTriangulation(geomodel, tags=["Air"])
   dΓsk       = Measure(Γsk, degree)
   αj         = μ * 0.24                  # gradient-jump penalty of the air mechanics

   #************************************
   # Dirichlet boundary conditions
   #************************************
   evolφ(Λ) = Λ
   Dφ = BoundaryConditions(BoundaryCondition("phi_Dirichlet", 0.0, evolφ))

   # Compression: zero while the magnetisation ramps up (Λ ≤ T₀), then a smooth ramp to uapplied
   p = 10
   tt(Λ)    = max(0.0, (Λ - T₀) / (1 - T₀))
   evolu(Λ) = 1 - (1 - tt(Λ))^p
   Du = BoundaryConditions(
      BoundaryCondition("usolid_fixed",    [0.0, 0.0],       evolu),
      BoundaryCondition("usolid_nonzero",  [0.0, -uapplied], evolu),
      BoundaryCondition("usolid_symmetry", [0.0, 0.0],       evolu; mask=(true, false)),
      BoundaryCondition("uair_fixed",      [0.0, 0.0],       evolu),
      BoundaryCondition("uair_nonzero",    [0.0, -uapplied], evolu),
      BoundaryCondition("uair_symmetry",   [0.0, 0.0],       evolu; mask=(true, false)))

   #******************************************
   # Finite element spaces and state
   #******************************************
   Vφ = TestFESpace(Ωdomain, reffeφ, Dφ, conformity=:H1)
   Uφ = TrialFESpace(Vφ, Dφ)
   Vu = TestFESpace(Ωdomain, reffeu, Du, conformity=:H1)
   Uu = TrialFESpace(Vu, Du)

   V    = MultiFieldFESpace([Vu, Vφ])
   U    = MultiFieldFESpace([Uu, Uφ])
   D_bc = MultiFieldBC([Du, Dφ])
   xh   = FEFunction(U, zero_free_values(U))

   #******************************************************
   # Maxwell stress in the air and its traction on the interface
   #******************************************************
   V_maxwell = TestFESpace(Ωair, reffeu)
   Pmaxw     = FEFunction(V_maxwell, zero_free_values(V_maxwell))
   Pmaxw_    = get_free_dof_values(Pmaxw)

   Usolid_int = FESpace(Γsolid_int, reffeu)
   V_traction = TestFESpace(Γsolid_int, reffeu)
   U_traction = TrialFESpace(V_traction)
   tmaxw      = interpolate_everywhere(Pmaxw, Usolid_int)
   tmaxw_     = get_free_dof_values(tmaxw)
   mass_int   = StaticLinearModel((u, v) -> ∫(u ⋅ v)dΓs, U_traction, V_traction, NothingBC(); backend)

   #******************************************************
   # Weak forms
   #******************************************************
   _, ∂ΨMax∂F, ∂ΨMax∂H0, _, ∂ΨMax∂H0F, ∂ΨMax∂H0H0           = model_air_maxwell()
   _, ∂Ψair∂F, ∂Ψair∂FF_                                   = model_air_mech()
   _, ∂Ψs1∂F, ∂Ψs1∂H0, ∂Ψs1∂FF_, ∂Ψs1∂H0F, ∂Ψs1∂H0H0        = model_solid1()
   _, ∂Ψs2∂F, ∂Ψs2∂H0, ∂Ψs2∂FF_, ∂Ψs2∂H0F, ∂Ψs2∂H0H0        = model_solid2()
   _, ∂Ψs3∂F, ∂Ψs3∂H0, ∂Ψs3∂FF_, ∂Ψs3∂H0F, ∂Ψs3∂H0H0        = model_solid3()

   # Stabilised (projected) mechanical tangents: eigenvalues clamped from below, always (`:projected`) or
   # only while the Newton is in projected mode (`:hybrid`, the switch is flipped by the solver)
   tangent in (:projected, :hybrid) || error("tangent must be :projected or :hybrid (got $tangent)")
   switch = tangent === :hybrid ? TangentSwitch() : nothing
   project(C, δ) = (switch === nothing || is_projected(switch)) ? regularize_hessian(C, δ) : C
   ∂Ψair∂FF(F)     = project(∂Ψair∂FF_(F), μv * 1e-2)
   ∂Ψs1∂FF(F, H, N) = project(∂Ψs1∂FF_(F, H, N), μ1 * 1e-4)
   ∂Ψs2∂FF(F, H, N) = project(∂Ψs2∂FF_(F, H, N), μ2 * 1e-6)
   ∂Ψs3∂FF(F, H, N) = project(∂Ψs3∂FF_(F, H, N), μ3 * 1e-6)

   # Kinematics and magnetisation direction of the grippers
   F, H, J = get_Kinematics(Kinematics(Mechano, Solid))
   ℋ₀      = get_Kinematics(Kinematics(Magneto, Solid))
   V_N = TestFESpace(Ωdomain, reffeu)
   Nh  = interpolate_everywhere(x -> VectorValue(0.0, 1.0), V_N)

   # Remanent magnetisation ramp over [0, T₀]
   function set_magnetisation!(Λ)
      model_mag2.αr = αr_max * min(Λ / T₀, 1.0)
      model_mag3.αr = αr3 * min(Λ / T₀, 1.0)
   end

   # Residual: at every load step the Maxwell stress of the air is evaluated at the last converged
   # state, interpolated onto the interface and L2-projected to the traction tmaxw
   res(Λ) = begin
      assemble_vector!(v -> ∫(∇(v)' ⊙ (∂ΨMax∂F ∘ (F ∘ (∇(xh[1])'), ℋ₀ ∘ (∇(xh[2])), Nh)))dΩair, Pmaxw_, V_maxwell)
      interpolate_everywhere!(Interpolable(Pmaxw), tmaxw_, tmaxw.dirichlet_values, Usolid_int)
      mass_int(tmaxw_)
      set_magnetisation!(Λ)

      ((u, φ), (v, vφ)) ->
         -1.0 * ∫(∇(vφ) ⋅ (∂Ψs1∂H0 ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)))dΩmag1 -
                ∫(∇(vφ) ⋅ (∂Ψs2∂H0 ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)))dΩmag2 -
                ∫(∇(vφ) ⋅ (∂Ψs3∂H0 ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)))dΩmag3 -
                ∫(∇(vφ) ⋅ (∂ΨMax∂H0 ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)))dΩair +
                ∫(∇(v)' ⊙ (∂Ψs1∂F ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)))dΩmec1 +
                ∫(∇(v)' ⊙ (∂Ψs2∂F ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)))dΩmec2 +
                ∫(∇(v)' ⊙ (∂Ψs3∂F ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)))dΩmec3 +
                ∫(∇(v)' ⊙ (∂Ψair∂F ∘ (F ∘ (∇(u)'))))dΩair +
                ∫(v ⋅ tmaxw)dΓs +
                ∫(αj * (jump(∇(u)) ⊙ jump(∇(v))))dΓsk
   end

   jac(Λ) = begin
      set_magnetisation!(Λ)

      ((u, φ), (du, dφ), (v, vφ)) ->
         ∫(∇(vφ) ⋅ ((∂Ψs1∂H0H0 ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)) ⋅ ∇(dφ)))dΩmag1 +
         ∫(∇(vφ) ⋅ ((∂Ψs2∂H0H0 ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)) ⋅ ∇(dφ)))dΩmag2 +
         ∫(∇(vφ) ⋅ ((∂Ψs3∂H0H0 ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)) ⋅ ∇(dφ)))dΩmag3 +
         ∫(∇(vφ) ⋅ ((∂ΨMax∂H0H0 ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)) ⋅ ∇(dφ)))dΩair +
         ∫(∇(v)' ⊙ ((∂Ψs1∂FF ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)) ⊙ (∇(du)')))dΩmec1 +
         ∫(∇(v)' ⊙ ((∂Ψs2∂FF ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)) ⊙ (∇(du)')))dΩmec2 +
         ∫(∇(v)' ⊙ ((∂Ψs3∂FF ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)) ⊙ (∇(du)')))dΩmec3 +
         ∫(∇(v)' ⊙ ((∂Ψair∂FF ∘ (F ∘ (∇(u)'))) ⊙ (∇(du)')))dΩair -
         ∫(∇(vφ) ⋅ ((∂Ψs1∂H0F ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)) ⊙ ∇(du)'))dΩmag1 -
         ∫(∇(vφ) ⋅ ((∂Ψs2∂H0F ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)) ⊙ ∇(du)'))dΩmag2 -
         ∫(∇(vφ) ⋅ ((∂Ψs3∂H0F ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)) ⊙ ∇(du)'))dΩmag3 -
         ∫(∇(vφ) ⋅ ((∂ΨMax∂H0F ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)) ⊙ ∇(du)'))dΩair -
         # mechanical equations of the grippers with respect to φ (the air carries no Maxwell stress in its
         # mechanical equations: it acts on the solid through the explicit traction tmaxw)
         ∫(∇(dφ) ⋅ ((∂Ψs1∂H0F ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)) ⊙ ∇(v)'))dΩmec1 -
         ∫(∇(dφ) ⋅ ((∂Ψs2∂H0F ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)) ⊙ ∇(v)'))dΩmec2 -
         ∫(∇(dφ) ⋅ ((∂Ψs3∂H0F ∘ (F ∘ (∇(u)'), ℋ₀ ∘ (∇(φ)), Nh)) ⊙ ∇(v)'))dΩmec3 +
         ∫(αj * (jump(∇(du)) ⊙ jump(∇(v))))dΓsk
   end

   #******************************************************
   # Nonlinear solver and computational model
   #******************************************************
   # 2D, first-order elements: a sparse LU is the fastest linear solver at this size. The injectivity-
   # preserving line search keeps det F > 0 at every quadrature point of the displacement field.
   linesearch = InjectivityLS(; β=1 - 1e-4, ρ=0.5, c=0.95, maxiter=15, αmin=1e-16)
   strategy   = switch === nothing ? ConsistentTangent() : HybridTangent(; atol_projected)
   nls        = NewtonRaphson(LUSolver(); newton..., verbose, linesearch, tangent=strategy)
   comp_model = StaticNonlinearModel(res, jac, U, V, D_bc; nls, xh, backend,
                                     injectivity=(dΩdomain, 1), tangent_switch=switch)

   #******************************************************
   # Post-processing: reactions on the loaded faces and vertical displacement of the gap corners
   #******************************************************
   V_solid          = TestFESpace(Ωsolid_mec, reffeu)
   solid_traction   = FEFunction(V_solid, zero_free_values(V_solid))
   solid_traction_  = get_free_dof_values(solid_traction)
   Utop             = FESpace(BoundaryTriangulation(Ωsolid_mec, tags="top_traction_boundary"), reffeu)
   Ubottom          = FESpace(BoundaryTriangulation(Ωsolid_mec, tags="bottom_traction_boundary"), reffeu)
   top_traction     = interpolate_everywhere(solid_traction, Utop)
   bottom_traction  = interpolate_everywhere(solid_traction, Ubottom)

   results = (step=Float64[], t_top=Vector{Float64}[], t_bottom=Vector{Float64}[],
              uy_top_right=Float64[], uy_bottom_right=Float64[], uy_top_left=Float64[], uy_bottom_left=Float64[])
   tag      = "w_$(w)_e_$(e)_αgrip_$(αgrip)_c_$(concentration)"
   jsonfile = joinpath(simdir, "results_SmallDevice_$(tag).json")

   # Fields of the .vtu files: material region of every cell (0 air, 1 soft part of the grippers, 2 and 3
   # the two hard-magnetic materials) and the spatial magnetic field h = F⁻ᵀℋ₀ in the deformed configuration
   cell_tags = Gridap.Geometry.get_face_tag_index(get_face_labeling(geomodel),
                  ["Air", "MagnetoOmega1", "MagnetoOmega2", "MagnetoOmega3"], num_cell_dims(geomodel))
   regions   = cell_tags[Gridap.Geometry.get_glue(Ωdomain, Val(num_cell_dims(geomodel))).tface_to_mface] .- 1
   hspatial(F, ℋ) = inv(F)' ⋅ ℋ
   vtkdir    = joinpath(simdir, "vtk_$(tag)")
   paraview && mkpath(vtkdir)

   function postprocess(post)
      Λ  = post.Λ[post.iter]
      xh = FEFunction(U, post.comp_model.caches.x)
      uh, φh = xh[1], xh[2]
      if paraview
         h = hspatial ∘ (F ∘ (∇(uh)'), ℋ₀ ∘ (∇(φh)))
         post.cachevtk[3][Λ] = createvtk(Ωdomain, joinpath(vtkdir, "step_$(lpad(post.iter, 4, '0'))"),
                                          cellfields=["u" => uh, "φ" => φh, "h" => h, "|h|" => norm ∘ h,
                                                      "J" => J ∘ (F ∘ (∇(uh)'))],
                                          celldata=["region" => regions])
      end
      # internal forces of the grippers, summed on the loaded faces (reaction = -internal force)
      lsolid(v) = ∫(∇(v)' ⊙ (∂Ψs1∂F ∘ (F ∘ (∇(uh)'), ℋ₀ ∘ (∇(φh)), Nh)))dΩmec1 +
                  ∫(∇(v)' ⊙ (∂Ψs2∂F ∘ (F ∘ (∇(uh)'), ℋ₀ ∘ (∇(φh)), Nh)))dΩmec2 +
                  ∫(∇(v)' ⊙ (∂Ψs3∂F ∘ (F ∘ (∇(uh)'), ℋ₀ ∘ (∇(φh)), Nh)))dΩmec3
      assemble_vector!(lsolid, solid_traction_, V_solid)
      t_top    = get_free_dof_values(interpolate_everywhere!(Interpolable(solid_traction), get_free_dof_values(top_traction), top_traction.dirichlet_values, Utop))
      t_bottom = get_free_dof_values(interpolate_everywhere!(Interpolable(solid_traction), get_free_dof_values(bottom_traction), bottom_traction.dirichlet_values, Ubottom))
      push!(results.t_top,    -vec(sum(reshape(t_top, 2, :), dims=2)))
      push!(results.t_bottom, -vec(sum(reshape(t_bottom, 2, :), dims=2)))
      push!(results.uy_top_right,    uh(VectorValue(4.5, 7.5))[2])
      push!(results.uy_bottom_right, uh(VectorValue(4.5, -7.5))[2])
      push!(results.uy_top_left,     uh(VectorValue(0.0, 7.5))[2])
      push!(results.uy_bottom_left,  uh(VectorValue(0.0, -7.5))[2])
      push!(results.step, Λ)
      open(io -> JSON.print(io, results), jsonfile, "w")
      callback === nothing || callback(post.iter, Λ)
   end

   post = PostProcessor(comp_model, postprocess; is_vtk=paraview, filepath=paraview ? vtkdir : simdir)
   return (; comp_model, nls, post, jsonfile, U, xh)
end

"""
    run_case(geomodel, w, e, αgrip, concentration; nsteps=500, maxbisec=10, grow=2.0, easy=5,
             ProjectDirichlet=true, kwargs...) -> jsonfile

Build the model of one configuration ([`sensor_model`](@ref), whose keywords are forwarded) and solve
it along the load path Λ ∈ [0, 1]. The load increment starts at `1/nsteps`; a failed increment is
bisected (at most `maxbisec` consecutive times) and, after an increment converged in at most `easy`
Newton iterations, the next one grows by `grow` (never above `1/nsteps`). With `ProjectDirichlet` every
increment starts from the Dirichlet projection of the new boundary values, cut by the injectivity-
preserving step (`det F > 0`). Returns the path of the JSON file with the results.
"""
function run_case(geomodel, w, e, αgrip, concentration; nsteps=500, maxbisec=10, grow=2.0, easy=5,
                  ProjectDirichlet=true, kwargs...)
   m = sensor_model(geomodel, w, e, αgrip, concentration; kwargs...)
   stepping = LoadStepping(; nsteps, maxbisec, grow, easy)
   solve!(m.comp_model; stepping, ProjectDirichlet, post=m.post)
   return m.jsonfile
end
