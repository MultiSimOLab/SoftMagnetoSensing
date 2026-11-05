#-----------------------------------------------------------------------
# fig4_sensitivity.jl
#
# Footprint-conditioned sensitivity analysis and response surfaces of main
# text Fig. 4c-e, from the released surrogate (no retraining).
#
# Outputs analysed: (u_On, u_Union, eps_On, eps_Union), with
#   eps_k = u_k / L_gauge,   L_gauge = 2 w (1 + d_BC).
#
# For each footprint, (W, h, c) is fixed and (w, e, phi_HM, d_BC) are
# sampled over the range covered by the FE designs of that footprint. Three
# estimates are computed:
#   (a) continuous box, N = 2000, same draws as si_analysis.jl
#       (Supplementary Table S8);
#   (b) continuous box, N = 200000;
#   (c) phi_HM restricted to the two manufactured loadings {20, 30} vol%,
#       N = 200000, 100 bootstrap replicates (Fig. 4d, Supplementary
#       Table S9).
# Indices are computed on physical outputs centred on their mean.
#
# Usage:  julia --project=. scripts/fig4_sensitivity.jl
# Output: results/fig4_sensitivity_data.json
#-----------------------------------------------------------------------

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "Surrogate.jl"))
using JSON

const DATAFILE  = joinpath(ROOT, "data", "fe_database.bson")
const MODELFILE = joinpath(ROOT, "models", "surrogate_L4_N8.bson")
const OUTFILE   = joinpath(ROOT, "results", "fig4_sensitivity_data.json")

const OUTNAMES   = ["u_On", "u_Union", "eps_On", "eps_Union"]
const FREENAMES  = INPUT_NAMES[1:4]
const PHI_LEVELS = [20.0, 30.0]
const E_FIX      = 2.0          # e (mm) used for the response surfaces

x_raw, y_raw, _ = load_database(DATAFILE)
cur  = curate(x_raw, y_raw)
x, y = cur.x, cur.y
n    = size(x, 2)

model, s, _ = load_surrogate(MODELFILE)

"""(u_On, u_Union, eps_On, eps_Union) for physical inputs X (7 x N)."""
function response(X)
    U = predict(model, s, X)
    return vcat(U, U ./ gauge_length(X)')
end

footprint_cols(tri) = findall(j -> x[5, j] == tri[1] && x[6, j] == tri[2] &&
                                   x[7, j] == tri[3], 1:n)

#-----------------------------------------------------------------------
# Sobol' indices per footprint
#-----------------------------------------------------------------------
fam_out = map(FOOTPRINTS) do fp
    cols = footprint_cols(fp.triplet)
    xf = x[:, cols]
    lo = [minimum(xf[i, :]) for i in 1:4]
    hi = [maximum(xf[i, :]) for i in 1:4]

    f(U) = response(vcat(U, repeat(fp.triplet, 1, size(U, 2))))

    draw_cont = uniform_box(lo, hi)
    function draw_mixed(rng, N)
        U = lo .+ (hi .- lo) .* rand(rng, 4, N)
        U[3, :] .= PHI_LEVELS[rand(rng, 1:2, N)]
        return U
    end

    si    = sobol_indices(f, draw_cont, 4; N = 2000, rng = MersenneTwister(SEED + 1), centre = true)
    cont  = sobol_indices(f, draw_cont, 4; N = 200_000, rng = MersenneTwister(SEED + 7), centre = true)
    mixed = sobol_indices(f, draw_mixed, 4; N = 200_000, rng = MersenneTwister(SEED + 11),
                          B = 100, second_order = true, centre = true)

    L = gauge_length(xf)
    Dict("name" => fp.name, "triplet" => fp.triplet, "n_samples" => length(cols),
         "lo" => lo, "hi" => hi,
         "sobol_continuous_N2000" => si,
         "sobol_continuous" => cont,
         "sobol" => mixed,
         "fe_w" => xf[1, :], "fe_e" => xf[2, :], "fe_phi" => xf[3, :],
         "fe_dBC" => xf[4, :], "fe_uOn" => y[1, cols],
         "fe_Lgauge" => L, "fe_epsOn" => y[1, cols] ./ L)
end

#-----------------------------------------------------------------------
# u_On and eps_On over the (w, d_BC) plane, e = E_FIX, at each
# manufactured phi_HM
#-----------------------------------------------------------------------
function surface(fp, phi; nw = 121, na = 101)
    cols = footprint_cols(fp.triplet)
    ws = collect(range(minimum(x[1, cols]), maximum(x[1, cols]), length = nw))
    as = collect(range(0.0, 1.0, length = na))
    Z = [response(reshape([w, E_FIX, phi, a, fp.triplet...], 7, 1)) for a in as, w in ws]
    return Dict("w" => ws, "d_BC" => as,
                "uOn" => [z[1] for z in Z], "epsOn" => [z[3] for z in Z])
end

surfaces = [Dict("name" => fp.name,
                 "maps" => [Dict("phi_HM" => p, "surface" => surface(fp, p)) for p in PHI_LEVELS])
            for fp in FOOTPRINTS]

#-----------------------------------------------------------------------
# Reported strain windows plotted in Fig. 4e (literature values; references
# in the figure caption): Escamilla 2012, Beynnon 1995; Rahman 2025;
# Byl 2002; Adam 2023; Meadows 2023; Garcia 2024; Weidlich 2023;
# Nagashima 2022.
#-----------------------------------------------------------------------
clinical = [
    Dict("label" => "ACL graft",            "site" => "knee",     "range" => [0.018, 0.036]),
    Dict("label" => "Cervical spinal cord", "site" => "neck",     "range" => [0.020, 0.060]),
    Dict("label" => "Median nerve",         "site" => "wrist",    "range" => [0.035, 0.076]),
    Dict("label" => "Achilles tendon",      "site" => "ankle",    "range" => [0.040, 0.090]),
    Dict("label" => "Lumbar disc, annulus", "site" => "lumbar",   "range" => [0.042, 0.079]),
    Dict("label" => "Supraspinatus tendon", "site" => "shoulder", "range" => [0.045, 0.056]),
    Dict("label" => "Patellar tendon",      "site" => "knee",     "range" => [0.047, 0.082]),
    Dict("label" => "Ulnar nerve",          "site" => "elbow",    "range" => [0.063, 0.136]),
]

payload = Dict(
    "seed" => SEED, "model_file" => basename(MODELFILE), "n_curated" => n,
    "outnames" => OUTNAMES, "varnames" => FREENAMES,
    "phi_levels" => PHI_LEVELS, "e_fixed_mm" => E_FIX,
    "families" => fam_out, "surfaces" => surfaces, "clinical" => clinical,
)
mkpath(dirname(OUTFILE))
open(io -> JSON.print(io, payload, 2), OUTFILE, "w")

#-----------------------------------------------------------------------
# Summary
#-----------------------------------------------------------------------
for f in fam_out
    println("\nFootprint ", f["name"], "  (W,h,c) = ", f["triplet"], "   n = ", f["n_samples"])
    for tag in ("sobol_continuous_N2000", "sobol_continuous", "sobol")
        r = f[tag]
        println("  ", rpad(tag, 24), "N = ", r["N"])
        for k in (1, 3)
            ord = sortperm(r["ST"][:, k], rev = true)
            println("    ", rpad(OUTNAMES[k], 8),
                    join([" $(rpad(FREENAMES[i], 7)) ST=$(round(r["ST"][i, k], digits = 3))" for i in ord]),
                    "   1-sum(S1) = ", round(r["interaction"][k], digits = 3))
        end
    end
end
large_map = surfaces[1]["maps"][2]["surface"]["uOn"]
println("\nu_On range, large footprint, phi_HM = 30, e = 2: ",
        round(minimum(large_map), digits = 2), " - ", round(maximum(large_map), digits = 2), " mm")
println("Written ", OUTFILE)
