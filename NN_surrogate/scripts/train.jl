#-----------------------------------------------------------------------
# train.jl
#
# Calibrates the surrogate from the FE database: curation, random
# 360 / 49 split of the 409 curated designs, 4 x 8 tanh network, Adam
# (lr 1e-2) with L2 penalty 1e-6, mini-batches of 250, 1e5 epochs, best
# iterate on the training loss.
#
# The released network (models/surrogate_L4_N8.bson) was obtained with this
# procedure; its split is stored in the file. A new run draws a new split
# and a new initialisation, so the weights differ and the accuracy is
# equivalent up to that randomness. Use the released file to reproduce the
# numbers of the paper exactly.
#
# Usage:  julia --project=. scripts/train.jl [epochs] [seed]
#         (defaults: 100000 epochs, seed 20250820; about 3 min on one core)
#-----------------------------------------------------------------------

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "Surrogate.jl"))

const EPOCHS  = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 100_000
const RUNSEED = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : SEED
const OUTFILE = joinpath(ROOT, "models", "retrained_L4_N8.bson")

x_raw, y_raw, _ = load_database(joinpath(ROOT, "data", "fe_database.bson"))
cur  = curate(x_raw, y_raw; umax = 15.0, alpha = 0.1)
x, y = cur.x, cur.y
n    = size(x, 2)

# 80 % of the 450 admissible designs, i.e. 360 of the 409 curated ones
n_train = round(Int, 0.8 * size(cur.x_adm, 2))

rng = MersenneTwister(RUNSEED)
Random.seed!(RUNSEED)                 # weight initialisation
train_idx = randperm(rng, n)[1:n_train]
held_idx  = setdiff(1:n, train_idx)
println("curated designs ", n, ", training ", length(train_idx), ", held out ", length(held_idx))

t0 = time()
model, s, losses = train_surrogate(x, y, train_idx; n_layers = 4, n_neurons = 8,
                                   epochs = EPOCHS, lr = 0.01, weight_decay = 1e-6,
                                   batchsize = 250, rng = rng,
                                   print_every = max(EPOCHS ÷ 20, 1))
println("training time ", round(time() - t0, digits = 1), " s")

ys, ys_hat = scale_y(s, y), model(scale_x(s, x))
r2_tr = r2_score(vec(ys[:, train_idx]), vec(ys_hat[:, train_idx]))
r2_ho = r2_score(vec(ys[:, held_idx]), vec(ys_hat[:, held_idx]))
res   = unscale_y(s, ys_hat) .- y
rmse_ho = [sqrt(mean(res[k, held_idx] .^ 2)) for k in 1:2]

println("best training loss ", round(minimum(losses), sigdigits = 5), " at epoch ", argmin(losses))
println("R2 (joint, standardised)  train ", round(r2_tr, digits = 5), "   held out ", round(r2_ho, digits = 5))
println("RMSE held out (um)  u_On ", round(1e3 * rmse_ho[1], digits = 1),
        "   u_Union ", round(1e3 * rmse_ho[2], digits = 1))

save_surrogate(OUTFILE, model, s, losses, train_idx; R2 = r2_score(vec(ys), vec(ys_hat)))
println("saved ", OUTFILE)
