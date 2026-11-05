#!/usr/bin/env python3
"""Compare two result files of the sensor (e.g. two solver settings or two runs).

    python3 bench/compare.py a.json b.json [Λmin]

Quantities are compared at the load factors Λ of the first file (linear interpolation of the second one
when its load steps differ), over the common range of Λ, optionally restricted to Λ ≥ Λmin. Reports the
maximum absolute difference and its ratio to the maximum absolute value of the quantity.
"""
import json, sys

def columns(d, k):
    v = d[k]
    if isinstance(v[0], list):
        return {f"{k}[{i}]": [e[i] for e in v] for i in range(len(v[0]))}
    return {k: v}

def interp(x, xs, ys):
    import bisect
    j = bisect.bisect_left(xs, x)
    if j == 0: return ys[0]
    if j >= len(xs): return ys[-1]
    t = (x - xs[j-1]) / (xs[j] - xs[j-1])
    return ys[j-1] + t * (ys[j] - ys[j-1])

a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
lmin = float(sys.argv[3]) if len(sys.argv) > 3 else -1.0
la, lb = a["step"], b["step"]
lmax = min(la[-1], lb[-1])
idx = [i for i, l in enumerate(la) if lmin <= l <= lmax + 1e-12]
print(f"steps: {len(la)} (Λ ≤ {la[-1]:.4f}) vs {len(lb)} (Λ ≤ {lb[-1]:.4f}); compared at {len(idx)} values of Λ in [{la[idx[0]]:.4f}, {la[idx[-1]]:.4f}]")
same_steps = len(la) == len(lb) and all(abs(p - q) < 1e-12 for p, q in zip(la, lb))
print("load steps identical" if same_steps else "load steps differ (interpolating the second file in Λ)")
for k in ["uy_top_right", "uy_bottom_right", "uy_top_left", "uy_bottom_left", "t_top", "t_bottom"]:
    for name, xa in columns(a, k).items():
        xb = columns(b, k)[name]
        vals = [(xa[i], xb[i] if same_steps else interp(la[i], lb, xb)) for i in idx]
        dmax = max(abs(p - q) for p, q in vals)
        scale = max(abs(p) for p, _ in vals) or 1.0
        print(f"{name:18s} max|Δ| = {dmax:.3e}   max|Δ|/max|x| = {dmax/scale:.3e}   (max|x| = {scale:.3e})")
