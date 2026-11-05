#-----------------------------------------------------------------------
# example.jl
#
# Short demonstration with the released surrogate (runs in under a
# minute once the packages are compiled):
#   1. prediction of the 49 held-out FE designs and their accuracy;
#   2. trigger strain of a new design;
#   3. gradient-based inverse design: recover a design whose predicted
#      trigger strains match those of a held-out FE simulation.
#
# Usage:  julia --project=. scripts/example.jl
# Expected output: results/example_output.txt
#-----------------------------------------------------------------------

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "Surrogate.jl"))
using Printf

x_raw, y_raw, ids = load_database(joinpath(ROOT, "data", "fe_database.bson"))
cur  = curate(x_raw, y_raw)
x, y = cur.x, cur.y
ids  = ids[cur.keep_adm][cur.keep_mah]
model, s, md = load_surrogate(joinpath(ROOT, "models", "surrogate_L4_N8.bson"))
held = setdiff(1:size(x, 2), md["Training_Indices"])

#-----------------------------------------------------------------------
# 1. Held-out designs
#-----------------------------------------------------------------------
y_hat = predict(model, s, x[:, held])
println("1. Held-out FE designs (", length(held), ")\n")
@printf("%-6s %-6s %-7s %-5s %-14s %-9s %-9s %-9s %-9s\n",
        "w", "e", "phi_HM", "d_BC", "(W,h,c)", "u_On FE", "u_On NN", "uUn FE", "uUn NN")
for (j, i) in enumerate(held[1:8])
    @printf("%-6.1f %-6.2f %-7.0f %-5.1f %-14s %-9.4f %-9.4f %-9.4f %-9.4f\n",
            x[1, i], x[2, i], x[3, i], x[4, i],
            "($(Int(x[5, i])),$(Int(x[6, i])),$(Int(x[7, i])))",
            y[1, i], y_hat[1, j], y[2, i], y_hat[2, j])
end
println("...")
for k in 1:2
    @printf("%-8s R2 = %.5f   RMSE = %.1f um\n", OUTPUT_NAMES[k],
            r2_score(y[k, held], y_hat[k, :]),
            1e3 * sqrt(mean((y_hat[k, :] .- y[k, held]) .^ 2)))
end

#-----------------------------------------------------------------------
# 2. A design not in the database
#-----------------------------------------------------------------------
xq = reshape([7.25, 2.5, 30.0, 0.25, 9.0, 17.0, 2.0], 7, 1)
yq = predict(model, s, xq)
eq = trigger_strain(xq, yq)
println("\n2. New design (w, e, phi_HM, d_BC, W, h, c) = ", vec(xq))
@printf("   u_On = %.3f mm, u_Union = %.3f mm, eps_On = %.4f, eps_Union = %.4f\n",
        yq[1], yq[2], eq[1], eq[2])

#-----------------------------------------------------------------------
# 3. Inverse design. Target: trigger strains of a held-out FE design of the
#    small footprint. Start: another design of the same footprint.
#    (W, h, c) and phi_HM are kept fixed; w, e and d_BC are free.
#-----------------------------------------------------------------------
small = [i for i in held if x[5, i] == 9.0]
i_t   = small[1]
target = vec(trigger_strain(x[:, i_t:i_t], y[:, i_t:i_t]))
x0 = copy(x[:, small[end]])
x0[3] = x[3, i_t]

sol = inverse_design(model, s, target, x0; fixed = (3, 5, 6, 7), lr = 0.01, iterations = 50_000)

println("\n3. Inverse design")
println("   target (eps_On, eps_Union) from FE design ", ids[i_t])
@printf("   target      %.5f  %.5f\n", target...)
@printf("   found       %.5f  %.5f   (misfit %.2e, %d iterations)\n", sol.eps..., sol.misfit, sol.iterations)
@printf("   start  w = %.3f  e = %.3f  d_BC = %.3f\n", x0[1], x0[2], x0[4])
@printf("   found  w = %.3f  e = %.3f  d_BC = %.3f\n", sol.x[1], sol.x[2], sol.x[4])
@printf("   FE     w = %.3f  e = %.3f  d_BC = %.3f\n", x[1, i_t], x[2, i_t], x[4, i_t])
println("   (the inverse problem is not unique: the design found need not coincide",
        "\n    with the FE design that produced the target)")
