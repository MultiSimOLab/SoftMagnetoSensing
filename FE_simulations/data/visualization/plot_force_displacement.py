#!/usr/bin/env python3
"""Force–displacement curve of one configuration from its result file(s).

    python3 plot_force_displacement.py output.png results.json[:label] [more.json[:label] ...]

Panel (a): vertical reaction R_y on the loaded (top) and fixed (bottom) faces (the `t_top`, `t_bottom` of the
result file) against the imposed compression;
panel (b): opening of the gap at the symmetry axis against the imposed compression. The imposed
displacement is the ramp of the script, uapplied·(1 − (1 − t)^10), t = (Λ − T₀)/(1 − T₀), with
uapplied = 15.5 mm and T₀ = 0.1. Reactions are those of the half model (symmetry plane x = 0), per unit
thickness (plane problem), in the units of the model (lengths in mm, stiffness in N/mm²).
"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

UAPPLIED, T0, P, GAP0 = 15.5, 0.1, 10, 15.0

def imposed(l):
    t = max(0.0, (l - T0) / (1 - T0))
    return UAPPLIED * (1 - (1 - t) ** P)

out = sys.argv[1]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for k, arg in enumerate(sys.argv[2:]):
    path, label = (arg.rsplit(":", 1) + [None])[:2] if ":" in arg else (arg, None)
    d = json.load(open(path))
    lam = d["step"]
    u = [imposed(l) for l in lam]
    ftop = [t[1] for t in d["t_top"]]            # R_y as stored in the result file
    fbot = [t[1] for t in d["t_bottom"]]
    gap = [GAP0 + a - b for a, b in zip(d["uy_top_left"], d["uy_bottom_left"])]
    lab = label or path
    c = f"C{k}"
    ax1.plot(u, ftop, "-", color=c, lw=1.6, label=f"{lab}: top (loaded) face")
    ax1.plot(u, fbot, "--", color=c, lw=1.2, label=f"{lab}: bottom (fixed) face")
    ax2.plot(u, gap, "-", color=c, lw=1.6, label=lab)
ax1.set_xlabel("imposed compression [mm]"); ax1.set_ylabel("vertical reaction $R_y$ (half model, per unit thickness)")
ax1.set_title("(a) force–displacement"); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
ax2.set_xlabel("imposed compression [mm]"); ax2.set_ylabel("gap opening at the axis [mm]")
ax2.set_title("(b) gap between the grippers"); ax2.grid(alpha=0.3); ax2.set_ylim(bottom=0); ax2.legend(fontsize=8)
fig.savefig(out, dpi=200)
print("written", out)
