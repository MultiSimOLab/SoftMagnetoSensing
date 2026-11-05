#-----------------------------------------------------------------------
# si_analysis.jl
#
# Analysis of the released surrogate (no retraining): curation statistics,
# accuracy on the stored training / held-out split, Sobol' indices over the
# full design box and per device footprint, GP baseline on the same split
# and evaluation cost. Every number of Supplementary Section S3 is written
# to results/si_nn_data.json, which plotting/si_figures.py turns into
# Supplementary Figs. S3-S7.
#
# Usage:  julia --project=. scripts/si_analysis.jl
#-----------------------------------------------------------------------

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "Surrogate.jl"))
using JSON

const DATAFILE  = joinpath(ROOT, "data", "fe_database.bson")
const MODELFILE = joinpath(ROOT, "models", "surrogate_L4_N8.bson")
const OUTFILE   = joinpath(ROOT, "results", "si_nn_data.json")

#-----------------------------------------------------------------------
# 1. Database and curation
#-----------------------------------------------------------------------
x_raw, y_raw, ids = load_database(DATAFILE)
cur   = curate(x_raw, y_raw; umax = 15.0, alpha = 0.1)
x, y  = cur.x, cur.y
n     = size(x, 2)
chi2_thr = quantile(Chisq(2), 0.9)

#-----------------------------------------------------------------------
# 2. Released network and its stored split
#-----------------------------------------------------------------------
model, s, md = load_surrogate(MODELFILE)
losses    = md["Losses"]
train_idx = md["Training_Indices"]
held_idx  = setdiff(1:n, train_idx)

#-----------------------------------------------------------------------
# 3. Accuracy. R2 in standardised space, jointly over both outputs and
#    per output.
#-----------------------------------------------------------------------
xs = scale_x(s, x)
ys = scale_y(s, y)
ys_hat = model(xs)

r2_joint(cols)  = r2_score(vec(ys[:, cols]), vec(ys_hat[:, cols]))
r2_out(k, cols) = r2_score(ys[k, cols], ys_hat[k, cols])
loss_on(cols)   = norm(ys_hat[:, cols] .- ys[:, cols]) / norm(ys[:, cols])

accuracy = Dict(
    "train_joint" => r2_joint(train_idx),
    "held_joint"  => r2_joint(held_idx),
    "all_joint"   => r2_joint(1:n),
    "train_out"   => [r2_out(k, train_idx) for k in 1:2],
    "held_out"    => [r2_out(k, held_idx) for k in 1:2],
    "all_out"     => [r2_out(k, 1:n) for k in 1:2],
)

y_hat = unscale_y(s, ys_hat)
resid = y_hat .- y

#-----------------------------------------------------------------------
# 4. Sobol' indices over the full 7-d box (standardised outputs),
#    N = 2000, 500 bootstrap replicates
#-----------------------------------------------------------------------
f_std(X) = model(scale_x(s, X))
sob = sobol_indices(f_std, uniform_box(s.xmin, s.xmax), 7;
                    N = 2000, B = 500, rng = MersenneTwister(SEED))

#-----------------------------------------------------------------------
# 5. Sobol' indices per footprint: (W, h, c) fixed, (w, e, phi_HM, d_BC)
#    sampled over the range populated by that footprint
#-----------------------------------------------------------------------
footprint_cols(X, tri) = findall(j -> X[5, j] == tri[1] && X[6, j] == tri[2] &&
                                      X[7, j] == tri[3], axes(X, 2))

fam_results = map(FOOTPRINTS) do fp
    cols = footprint_cols(x, fp.triplet)
    lo = [minimum(x[i, cols]) for i in 1:4]
    hi = [maximum(x[i, cols]) for i in 1:4]
    f(U) = f_std(vcat(U, repeat(fp.triplet, 1, size(U, 2))))
    r = sobol_indices(f, uniform_box(lo, hi), 4; N = 2000, rng = MersenneTwister(SEED + 1))
    (name = fp.name, triplet = fp.triplet, n = length(cols), lo = lo, hi = hi,
     S1 = r["S1"], ST = r["ST"])
end

#-----------------------------------------------------------------------
# 6. Design-of-experiments coverage: number of (phi_HM, d_BC) runs that
#    converged (raw) and that survived curation, per (w, e)
#-----------------------------------------------------------------------
function doe_grid(X, tri)
    cols = footprint_cols(X, tri)
    ws = sort(unique(X[1, cols]))
    es = sort(unique(X[2, cols]))
    counts = [count(j -> X[1, j] == w && X[2, j] == e, cols) for e in es, w in ws]
    return ws, es, counts
end
doe_raw = [doe_grid(x_raw, fp.triplet) for fp in FOOTPRINTS]
doe_cur = [doe_grid(x, fp.triplet) for fp in FOOTPRINTS]

#-----------------------------------------------------------------------
# 7. Gaussian-process baseline on the same split
#-----------------------------------------------------------------------
xs64  = 2 .* (x .- s.xmin) ./ (s.xmax .- s.xmin) .- 1
gp    = fit_gp(xs64[:, train_idx], ys[:, train_idx])
gp_tr = gp(xs64[:, train_idx])
gp_ho = gp(xs64[:, held_idx])
gp_metrics = Dict(
    "train_joint" => r2_score(vec(ys[:, train_idx]), vec(gp_tr)),
    "held_joint"  => r2_score(vec(ys[:, held_idx]), vec(gp_ho)),
    "held_out"    => [r2_score(ys[k, held_idx], gp_ho[k, :]) for k in 1:2],
)

#-----------------------------------------------------------------------
# 8. Cost of one surrogate evaluation (single thread)
#-----------------------------------------------------------------------
function time_eval(m, X; reps = 20)
    m(X)
    t = minimum(begin t0 = time_ns(); m(X); time_ns() - t0 end for _ in 1:reps)
    return t * 1e-9 / size(X, 2)
end
t_single = time_eval(model, xs[:, 1:1])
t_batch  = time_eval(model, xs)

#-----------------------------------------------------------------------
# 9. Trigger strains over the curated set
#-----------------------------------------------------------------------
eps = trigger_strain(x, y)

#-----------------------------------------------------------------------
# 10. Output
#-----------------------------------------------------------------------
payload = Dict(
    "seed" => SEED,
    "files" => Dict("data" => basename(DATAFILE), "model" => basename(MODELFILE)),
    "inputs" => INPUT_NAMES, "outputs" => OUTPUT_NAMES,
    "curation" => Dict(
        "n_raw" => size(x_raw, 2),
        "n_admissible" => size(cur.x_adm, 2),
        "n_curated" => n,
        "n_rejected_admissibility" => count(.!cur.keep_adm),
        "n_rejected_mahalanobis" => count(.!cur.keep_mah),
        "chi2_threshold" => chi2_thr,
        "d2" => cur.d2,
        "keep_mahalanobis" => cur.keep_mah,
        "y_admissible" => [cur.y_adm[1, :], cur.y_adm[2, :]],
        "rejected_identifiers" => ids[cur.keep_adm][.!cur.keep_mah],
    ),
    "dataset" => Dict(
        "y_min"  => [minimum(y[k, :]) for k in 1:2],
        "y_max"  => [maximum(y[k, :]) for k in 1:2],
        "y_mean" => [mean(y[k, :]) for k in 1:2],
        "y_std"  => [std(y[k, :]) for k in 1:2],
        "pearson_r" => cor(y[1, :], y[2, :]),
        "x_levels" => [sort(unique(x[i, :])) for i in 1:7],
        "eps_trigger_range" => [[minimum(eps[k, :]), maximum(eps[k, :])] for k in 1:2],
    ),
    "split" => Dict("n_train" => length(train_idx), "n_held" => length(held_idx),
                    "train_idx" => train_idx, "held_idx" => held_idx),
    "model" => Dict("n_layers" => length(md["weights"]) - 1,
                    "n_neurons" => size(md["weights"][1], 1),
                    "n_parameters" => n_parameters(model),
                    "mu" => s.mu, "sigma" => s.sigma, "xmin" => s.xmin, "xmax" => s.xmax),
    "training" => Dict(
        "losses" => losses,
        "n_epochs" => length(losses),
        "loss_best" => minimum(losses),
        "epoch_best" => argmin(losses),
        "loss_final" => losses[end],
        "loss_train" => loss_on(train_idx),
        "loss_held" => loss_on(held_idx),
        "loss_all" => loss_on(1:n),
    ),
    "accuracy" => accuracy,
    "parity" => Dict(
        "y_target" => [y[1, :], y[2, :]],
        "y_pred"   => [y_hat[1, :], y_hat[2, :]],
        "residual" => [resid[1, :], resid[2, :]],
        "rmse_mm"  => [sqrt(mean(resid[k, :] .^ 2)) for k in 1:2],
        "mae_mm"   => [mean(abs.(resid[k, :])) for k in 1:2],
        "max_abs_mm" => [maximum(abs.(resid[k, :])) for k in 1:2],
        "rmse_train_mm" => [sqrt(mean(resid[k, train_idx] .^ 2)) for k in 1:2],
        "rmse_held_mm"  => [sqrt(mean(resid[k, held_idx] .^ 2)) for k in 1:2],
    ),
    "sobol" => Dict(
        "N" => 2000, "n_bootstrap" => 500, "varnames" => INPUT_NAMES,
        "S1" => sob["S1"], "ST" => sob["ST"],
        "S1_lo" => sob["S1_lo"], "S1_hi" => sob["S1_hi"],
        "ST_lo" => sob["ST_lo"], "ST_hi" => sob["ST_hi"],
        "sum_S1" => sob["sum_S1"], "sum_ST" => sob["sum_ST"],
        "joint_footprint_S1" => [sum(sob["S1"][5:7, k]) for k in 1:2],
        "joint_footprint_ST" => [sum(sob["ST"][5:7, k]) for k in 1:2],
    ),
    "sobol_family" => [Dict(
        "name" => f.name, "triplet" => f.triplet, "n_samples" => f.n,
        "varnames" => INPUT_NAMES[1:4], "lo" => f.lo, "hi" => f.hi,
        "S1" => f.S1, "ST" => f.ST,
        "sum_S1" => [sum(f.S1[:, k]) for k in 1:2],
    ) for f in fam_results],
    "doe" => [Dict(
        "name" => FOOTPRINTS[i].name, "triplet" => FOOTPRINTS[i].triplet,
        "w" => doe_raw[i][1], "e" => doe_raw[i][2],
        "counts_converged" => doe_raw[i][3],
        "counts_curated" => doe_cur[i][3],
        "n_converged" => sum(doe_raw[i][3]),
    ) for i in eachindex(FOOTPRINTS)],
    "gp" => gp_metrics,
    "cost" => Dict("t_per_eval_single_s" => t_single,
                   "t_per_eval_batched_s" => t_batch,
                   "batch_size" => n,
                   "sobol_evaluations" => 2000 * (7 + 2)),
)

mkpath(dirname(OUTFILE))
open(io -> JSON.print(io, payload, 2), OUTFILE, "w")

#-----------------------------------------------------------------------
# Summary
#-----------------------------------------------------------------------
rd(v, k = 5) = round.(v, digits = k)
println("\nCuration   ", size(x_raw, 2), " raw -> ", size(cur.x_adm, 2), " admissible -> ", n, " curated")
println("Split      ", length(train_idx), " training / ", length(held_idx), " held out")
println("Model      ", n_parameters(model), " parameters, best loss ", rd(minimum(losses), 6),
        " at epoch ", argmin(losses))
println("R2 joint   train ", rd(accuracy["train_joint"]), "   held out ", rd(accuracy["held_joint"]),
        "   all ", rd(accuracy["all_joint"]))
println("R2 u_On    train ", rd(accuracy["train_out"][1]), "   held out ", rd(accuracy["held_out"][1]))
println("R2 u_Union train ", rd(accuracy["train_out"][2]), "   held out ", rd(accuracy["held_out"][2]))
println("RMSE (um)  held out ", rd(1e3 .* payload["parity"]["rmse_held_mm"], 1),
        "   all ", rd(1e3 .* payload["parity"]["rmse_mm"], 1),
        "   max |res| ", rd(1e3 .* payload["parity"]["max_abs_mm"], 1))
println("GP         train ", rd(gp_metrics["train_joint"], 6), "   held out ", rd(gp_metrics["held_joint"]))
println("Cost       ", rd(t_single * 1e6, 2), " us (single), ", rd(t_batch * 1e6, 3), " us/sample (batch of ", n, ")")
for k in 1:2
    ord = sortperm(sob["ST"][:, k], rev = true)
    println("Sobol ", OUTPUT_NAMES[k], ": ",
            join(["$(INPUT_NAMES[i]) $(rd(sob["ST"][i, k], 3))" for i in ord], ", "))
end
for f in fam_results
    println("Footprint ", f.name, " (n = ", f.n, ")  ST u_On: ",
            join(["$(INPUT_NAMES[i]) $(rd(f.ST[i, 1], 3))" for i in 1:4], ", "))
end
println("eps_On range ", rd(payload["dataset"]["eps_trigger_range"][1], 3))
println("\nWritten ", OUTFILE)
