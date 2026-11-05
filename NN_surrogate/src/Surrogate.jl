#-----------------------------------------------------------------------
# Surrogate.jl
#
# Neural-network surrogate of the magneto-mechanical FE model of the
# multi-domain binary sensor. Inputs and outputs:
#
#   x = (w, e, phi_HM, d_BC, W, h, c)      7 design variables
#   y = (u_On, u_Union)                    2 recovery displacements (mm)
#
# Contents: database loading and curation, input/output scaling, network
# definition and training, save/load, Sobol' indices (pick-and-freeze),
# Gaussian-process baseline and gradient-based inverse design.
#-----------------------------------------------------------------------

using BSON
using Flux
using Zygote
using Statistics
using LinearAlgebra
using Random
using Distributions
using GaussianProcesses

# GaussianProcesses v0.12 defines ldiv!(::PDMat, x) with an untyped second
# argument, which is ambiguous with recent PDMats releases.
const _PDMats = GaussianProcesses.PDMats
LinearAlgebra.ldiv!(cK::_PDMats.PDMat, x::AbstractVecOrMat) = LinearAlgebra.ldiv!(cK.chol, x)

const INPUT_NAMES  = ["w", "e", "phi_HM", "d_BC", "W", "h", "c"]
const OUTPUT_NAMES = ["u_On", "u_Union"]

# (W, h, c) of the two simulated device footprints, in mm
const FOOTPRINTS = [(name = "large", triplet = [15.0, 36.0, 3.0]),
                    (name = "small", triplet = [9.0, 17.0, 2.0])]

const SEED = 20250820


#-----------------------------------------------------------------------
# Database and curation
#-----------------------------------------------------------------------

"""
    load_database(path) -> x, y, ids

`x` is 7 x n (inputs), `y` is 2 x n (u_On, u_Union in mm), `ids` the FE
simulation identifiers.
"""
function load_database(path)
    raw = BSON.load(path)
    data = raw[:data]
    return Matrix(data[:, 1:7]'), Matrix(data[:, 8:9]'), String.(raw[:identifier])
end

"""
    mahalanobis_filter(y; alpha=0.1) -> y_kept, keep, d2

Keeps the columns of `y` whose squared Mahalanobis distance to the sample
mean is below the chi-square quantile of order 1 - alpha.
"""
function mahalanobis_filter(y; alpha::Float64 = 0.1)
    d, n = size(y)
    mu = mean(y, dims = 2)
    yc = y .- mu
    Sigma = (yc * yc') / (n - 1)
    t = inv(Sigma) * yc
    d2 = [dot(yc[:, i], t[:, i]) for i in 1:n]
    keep = d2 .< quantile(Chisq(d), 1 - alpha)
    return y[:, keep], keep, d2
end

"""
    curate(x, y; umax=15.0, alpha=0.1)

Two filters: (i) physical admissibility, u_On <= umax (reconnection within
the imposed stroke); (ii) Mahalanobis screening of the outputs.
"""
function curate(x, y; umax = 15.0, alpha = 0.1)
    keep_adm = y[1, :] .<= umax
    x_adm, y_adm = x[:, keep_adm], y[:, keep_adm]
    y_cur, keep_mah, d2 = mahalanobis_filter(y_adm; alpha = alpha)
    x_cur = x_adm[:, keep_mah]
    return (x = x_cur, y = y_cur, x_adm = x_adm, y_adm = y_adm,
            keep_adm = keep_adm, keep_mah = keep_mah, d2 = d2)
end


#-----------------------------------------------------------------------
# Scaling
#
# Inputs are mapped affinely onto [-1, 1] with the min/max of the training
# split; outputs are standardised with the training mean and std.
#-----------------------------------------------------------------------

struct Scaling
    xmin::Vector{Float64}
    xmax::Vector{Float64}
    mu::Vector{Float64}
    sigma::Vector{Float64}
end

function fit_scaling(x, y)
    xmin = vec(minimum(x, dims = 2))
    xmax = vec(maximum(x, dims = 2))
    any(xmax .== xmin) && error("an input has zero range on the training split")
    return Scaling(xmin, xmax, vec(mean(y, dims = 2)), vec(std(y, dims = 2)))
end

scale_x(s::Scaling, x)   = Float32.(2 .* (x .- s.xmin) ./ (s.xmax .- s.xmin) .- 1)
scale_y(s::Scaling, y)   = (y .- s.mu) ./ s.sigma
unscale_y(s::Scaling, z) = z .* s.sigma .+ s.mu

"""Surrogate prediction in physical units: 7 x N inputs -> 2 x N outputs (mm)."""
predict(model, s::Scaling, x) = unscale_y(s, model(scale_x(s, x)))

"""Effective gauge length 2 w (1 + d_BC), in mm."""
gauge_length(x) = 2 .* (x[4, :] .+ 1) .* x[1, :]

"""Dimensionless trigger strains (eps_On, eps_Union) = y / (2 w (1 + d_BC))."""
trigger_strain(x, y) = y ./ gauge_length(x)'


#-----------------------------------------------------------------------
# Network and training
#-----------------------------------------------------------------------

"""Fully connected network, `n_layers` hidden layers of `n_neurons` tanh units, linear output."""
function build_mlp(n_in, n_out, n_layers, n_neurons; activation = tanh_fast)
    layers = Any[Dense(n_in => n_neurons, activation)]
    for _ in 2:n_layers
        push!(layers, Dense(n_neurons => n_neurons, activation))
    end
    push!(layers, Dense(n_neurons => n_out))
    return Chain(layers...)
end

"""Relative Frobenius-norm residual, ||f(x) - y||_F / ||y||_F."""
relative_loss(m, x, y) = sqrt(sum(abs2, m(x) .- y) / sum(abs2, y))

"""Coefficient of determination of `p` against `t` (vectors)."""
r2_score(t, p) = 1 - sum(abs2, t .- p) / sum(abs2, t .- mean(t))

"""
    train_surrogate(x, y, train_idx; kwargs...) -> model, scaling, losses

Adam with decoupled L2 penalty, mini-batches, fixed number of epochs. The
loss is evaluated on the whole training split after every epoch and the
best iterate is returned. Scaling constants come from the training split
only.
"""
function train_surrogate(x, y, train_idx;
                         n_layers = 4, n_neurons = 8, epochs = 100_000,
                         lr = 0.01, weight_decay = 1e-6, batchsize = 250,
                         rng = Random.default_rng(), print_every = 0)
    s  = fit_scaling(x[:, train_idx], y[:, train_idx])
    xt = scale_x(s, x[:, train_idx])
    yt = Float32.(scale_y(s, y[:, train_idx]))

    model  = build_mlp(size(x, 1), size(y, 1), n_layers, n_neurons)
    opt    = Flux.setup(OptimiserChain(WeightDecay(weight_decay), Adam(lr)), model)
    loader = Flux.DataLoader((xt, yt), batchsize = batchsize, shuffle = true, rng = rng)

    losses = zeros(epochs)
    best_loss, best = Inf, deepcopy(model)
    for epoch in 1:epochs
        Flux.train!(relative_loss, model, loader, opt)
        L = relative_loss(model, xt, yt)
        losses[epoch] = L
        if L < best_loss
            best_loss, best = L, deepcopy(model)
        end
        if print_every > 0 && epoch % print_every == 0
            println("epoch ", epoch, "   loss ", round(L, sigdigits = 5),
                    "   best ", round(best_loss, sigdigits = 5))
        end
    end
    return best, s, losses
end

"""Stores weights, scaling constants, loss history and training indices."""
function save_surrogate(path, model, s::Scaling, losses, train_idx; R2 = NaN)
    dense = [l for l in model if l isa Dense]
    model_data = Dict(
        "weights" => [l.weight for l in dense],
        "bias" => [l.bias for l in dense],
        "xmin" => s.xmin, "xmax" => s.xmax, "μ" => s.mu, "sigma" => s.sigma,
        "Losses" => losses, "Training_Indices" => collect(train_idx), "R2" => R2)
    BSON.bson(path, Dict(:model_data => model_data))
    return path
end

"""
    load_surrogate(path) -> model, scaling, model_data

Rebuilds the network from a stored file. The architecture is read from the
weight shapes; hidden activations are tanh.
"""
function load_surrogate(path)
    md = BSON.load(path)[:model_data]
    W, b = md["weights"], md["bias"]
    model = build_mlp(size(W[1], 2), size(W[end], 1), length(W) - 1, size(W[1], 1))
    for i in eachindex(W)
        model[i].weight .= W[i]
        model[i].bias   .= b[i]
    end
    s = Scaling(Float64.(md["xmin"]), Float64.(md["xmax"]),
                Float64.(md["μ"]), Float64.(md["sigma"]))
    return model, s, md
end

n_parameters(model) = sum(length, Flux.trainables(model))


#-----------------------------------------------------------------------
# Sobol' indices
#
# Pick-and-freeze Monte Carlo: Saltelli (2010) estimator for S1, Jansen
# (1999) for ST, closed second-order indices from pair freezing, and
# percentile bootstrap intervals obtained by resampling the N base rows.
#
#   f(X)        d x N -> n_out x N
#   draw(rng,N) returns the d x N design matrix (the sampling law)
#   centre      subtract mean(f(A)) before applying the S1 estimator
#-----------------------------------------------------------------------
function sobol_indices(f, draw, d; N, rng, B = 0, second_order = false, centre = false)
    A  = draw(rng, N)
    Bm = draw(rng, N)
    fA, fB = f(A), f(Bm)
    n_out  = size(fA, 1)
    fAB = [begin AB = copy(A); AB[i, :] .= Bm[i, :]; f(AB) end for i in 1:d]

    # s1/st are local on purpose: the outer S1/ST must not be overwritten by
    # the bootstrap calls
    function estimate(cols)
        m  = length(cols)
        s1 = zeros(d, n_out)
        st = zeros(d, n_out)
        for k in 1:n_out
            a  = fA[k, cols]
            mk = centre ? mean(a) : 0.0
            vk = var(a)
            ac = a .- mk
            bc = fB[k, cols] .- mk
            for i in 1:d
                ab = fAB[i][k, cols] .- mk
                s1[i, k] = sum(bc .* (ab .- ac)) / m / vk
                st[i, k] = sum((ac .- ab) .^ 2) / (2m) / vk
            end
        end
        return s1, st
    end

    S1, ST = estimate(1:N)
    out = Dict{String,Any}("N" => N, "S1" => S1, "ST" => ST,
                           "sum_S1" => [sum(S1[:, k]) for k in 1:n_out],
                           "sum_ST" => [sum(ST[:, k]) for k in 1:n_out],
                           "interaction" => [1 - sum(S1[:, k]) for k in 1:n_out])

    if B > 0
        S1b = Array{Float64}(undef, d, n_out, B)
        STb = Array{Float64}(undef, d, n_out, B)
        for t in 1:B
            S1b[:, :, t], STb[:, :, t] = estimate(rand(rng, 1:N, N))
        end
        q(X, p) = [quantile(vec(X[i, k, :]), p) for i in 1:d, k in 1:n_out]
        out["n_bootstrap"] = B
        out["S1_lo"], out["S1_hi"] = q(S1b, 0.025), q(S1b, 0.975)
        out["ST_lo"], out["ST_hi"] = q(STb, 0.025), q(STb, 0.975)
    end

    if second_order
        S2 = zeros(d, d, n_out)
        for i in 1:d, j in (i + 1):d
            AB = copy(A); AB[i, :] .= Bm[i, :]; AB[j, :] .= Bm[j, :]
            fij = f(AB)
            for k in 1:n_out
                mk = centre ? mean(fA[k, :]) : 0.0
                vk = var(fA[k, :])
                closed = sum((fB[k, :] .- mk) .* (fij[k, :] .- fA[k, :])) / N / vk
                S2[i, j, k] = S2[j, i, k] = closed - S1[i, k] - S1[j, k]
            end
        end
        out["S2"] = S2
    end
    return out
end

"""Uniform sampler on the box [lo, hi]."""
uniform_box(lo, hi) = (rng, N) -> lo .+ (hi .- lo) .* rand(rng, length(lo), N)


#-----------------------------------------------------------------------
# Gaussian-process baseline: one GP per output, zero mean, ARD Matern 5/2
# kernel, hyperparameters by maximum marginal likelihood. Works on scaled
# inputs/outputs, like the network.
#-----------------------------------------------------------------------
struct GPModel
    gps::Vector{GaussianProcesses.GPE}
end

function (m::GPModel)(X::AbstractMatrix)
    Xd = Float64.(X)
    Y  = zeros(length(m.gps), size(Xd, 2))
    for j in eachindex(m.gps)
        Y[j, :] = predict_y(m.gps[j], Xd)[1]
    end
    return Y
end

function fit_gp(x, y)
    d  = size(x, 1)
    Xd = Float64.(x)
    gps = map(1:size(y, 1)) do j
        gp = GP(Xd, Float64.(y[j, :]), MeanZero(), Mat52Ard(zeros(d), 0.0), -2.0)
        optimize!(gp)
        gp
    end
    return GPModel(gps)
end


#-----------------------------------------------------------------------
# Gradient-based inverse design
#
# Finds x such that trigger_strain(x, predict(x)) matches `target` (eps_On,
# eps_Union). Gradients through scaling, network, de-scaling and strain
# conversion are obtained with Zygote; the update is Adam with projection on
# the box `lb <= x <= ub`. Variables in `fixed` keep their initial value.
# The geometric constraint w (1 + d_BC) <= h is added as a squared-hinge
# penalty.
#-----------------------------------------------------------------------
function inverse_design(model, s::Scaling, target, x0;
                        fixed = (5, 6, 7), lb = s.xmin, ub = s.xmax,
                        lr = 0.01, iterations = 50_000, tol = 1e-12,
                        constraint_weight = 1e4)
    lb = collect(Float64, lb)
    ub = collect(Float64, ub)
    for i in fixed
        lb[i] = ub[i] = x0[i]
    end

    function misfit(x)
        X = reshape(x, :, 1)
        eps = vec(predict(model, s, X)) ./ (2 * x[1] * (1 + x[4]))
        return sum(abs2, eps .- target)
    end
    violation(x) = max(0.0, x[1] * (1 + x[4]) - x[6])
    objective(x) = misfit(x) + constraint_weight * violation(x)^2

    x = collect(Float64, x0)
    state = Flux.setup(Adam(lr), x)
    n_it = 0
    for it in 1:iterations
        n_it = it
        misfit(x) < tol && violation(x) <= tol && break
        g = Zygote.gradient(objective, x)[1]
        for i in fixed
            g[i] = 0.0
        end
        Flux.update!(state, x, g)
        x .= clamp.(x, lb, ub)
    end
    violation(x) > 1e-6 && @warn "constraint w(1+d_BC) <= h violated by $(violation(x))"
    eps = vec(predict(model, s, reshape(x, :, 1))) ./ (2 * x[1] * (1 + x[4]))
    return (x = x, eps = eps, misfit = misfit(x), iterations = n_it)
end
