#!/usr/bin/env python3
"""
si_figures.py

Supplementary Figs. S3-S7 (surrogate) from results/si_nn_data.json, written
by scripts/si_analysis.jl. Plotting only, no computation.

Usage:  python3 plotting/si_figures.py [si_nn_data.json] [outdir]
"""

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
JSON_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "si_nn_data.json")
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "figures")
os.makedirs(OUTDIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.linewidth": 0.7,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
})

BLUE = "#4878a8"
RED = "#c44e52"
GREY = "#4d4d4d"

D = json.load(open(JSON_PATH))


def mat(x):
    """JSON.jl writes a Julia (m x n) matrix as n arrays of length m."""
    return np.asarray(x, dtype=float).T


VARLABEL = {
    "w": r"$w$",
    "e": r"$e$",
    "phi_HM": r"$\phi_{\mathrm{HM}}$",
    "d_BC": r"$d_{\mathrm{BC}}$",
    "W": r"$W$",
    "h": r"$h$",
    "c": r"$c$",
}
OUTLABEL = [r"$u_{\mathrm{On}}$", r"$u_{\mathrm{Union}}$"]


def save(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote", path)


# ----------------------------------------------------------------------
# Fig. S3, data curation
# ----------------------------------------------------------------------
def fig_curation():
    cur = D["curation"]
    y1 = np.asarray(cur["y_admissible"][0])
    y2 = np.asarray(cur["y_admissible"][1])
    keep = np.asarray(cur["keep_mahalanobis"], dtype=bool)
    d2 = np.asarray(cur["d2"])
    thr = cur["chi2_threshold"]

    fig, ax = plt.subplots(2, 2, figsize=(7.0, 5.0))

    # (a) scatter with the chi^2 contour of the fitted Gaussian
    a = ax[0, 0]
    a.plot(y1[keep], y2[keep], ".", ms=3.0, color=BLUE,
           label=f"retained ({keep.sum()})")
    a.plot(y1[~keep], y2[~keep], "o", ms=4.0, mfc="none", mew=0.9, color=RED,
           label=f"rejected ({(~keep).sum()})")
    Y = np.vstack([y1, y2])
    mu = Y.mean(axis=1)
    S = np.cov(Y)
    vals, vecs = np.linalg.eigh(S)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * np.sqrt(vals * thr)
    a.add_patch(Ellipse(mu, width, height, angle=angle, fc="none",
                        ec="k", ls="--", lw=0.9))
    a.set_xlabel(r"$u_{\mathrm{On}}$ (mm)")
    a.set_ylabel(r"$u_{\mathrm{Union}}$ (mm)")
    a.set_title("(a) Mahalanobis screening", loc="left")
    a.legend(loc="upper left", handletextpad=0.4)

    # (b) distribution of squared distances against the cut-off
    b = ax[0, 1]
    b.hist(d2, bins=np.linspace(0, max(d2.max(), thr) * 1.02, 60),
           color=BLUE, ec="white", lw=0.3)
    b.axvline(thr, color=RED, ls="--", lw=1.0)
    b.annotate(r"$\chi^2_{2,\,0.9}$", xy=(thr, b.get_ylim()[1] * 0.88),
               xytext=(4, 0), textcoords="offset points",
               color=RED, fontsize=8)
    b.set_xlabel(r"squared Mahalanobis distance $d^2$")
    b.set_ylabel("count")
    b.set_title("(b) distance distribution", loc="left")

    # (c,d) design-of-experiments coverage per device family
    for j, doe in enumerate(D["doe"]):
        a = ax[1, j]
        w = np.asarray(doe["w"], dtype=float)
        e = np.asarray(doe["e"], dtype=float)
        counts = mat(doe["counts_converged"])                      # (n_e, n_w)
        im = a.imshow(counts, origin="lower", aspect="auto", cmap="YlGnBu",
                      vmin=0, vmax=6)
        for ie in range(counts.shape[0]):
            for iw in range(counts.shape[1]):
                v = counts[ie, iw]
                a.text(iw, ie, f"{int(v)}", ha="center", va="center",
                       fontsize=6.5, color="white" if v >= 4 else "0.15")
        a.set_xticks(range(len(w)))
        a.set_xticklabels([f"{v:g}" for v in w], fontsize=6.5)
        a.set_yticks(range(len(e)))
        a.set_yticklabels([f"{v:g}" for v in e], fontsize=6.5)
        a.set_xlabel(r"$w$ (mm)")
        a.set_ylabel(r"$e$ (mm)")
        a.set_title(rf"({'cd'[j]}) {doe['name']} device: converged of 6 "
                    rf"$(\phi_{{\mathrm{{HM}}}},d_{{\mathrm{{BC}}}})$ runs",
                    loc="left", fontsize=8)
        a.tick_params(top=False, right=False)
        if j == 1:
            cb = fig.colorbar(im, ax=a, fraction=0.035, pad=0.02)
            cb.set_ticks([0, 2, 4, 6])
            cb.ax.tick_params(labelsize=7)

    fig.tight_layout(w_pad=1.6, h_pad=1.6)
    save(fig, "curation.pdf")


# ----------------------------------------------------------------------
# Fig. S4, training history
# ----------------------------------------------------------------------
def fig_loss():
    tr = D["training"]
    L = np.asarray(tr["losses"])
    ep = np.arange(1, L.size + 1)
    best_e = tr["epoch_best"]
    best_L = tr["loss_best"]

    fig, ax = plt.subplots(figsize=(4.2, 2.7))
    ax.plot(ep, L, lw=0.5, color=BLUE, label="per-epoch loss")

    # running minimum: the iterate retained at each epoch
    ax.plot(ep, np.minimum.accumulate(L), lw=0.9, color=RED,
            label="best iterate")
    ax.plot([best_e], [best_L], "o", ms=3.5, color="k")
    ax.annotate(rf"$\mathcal{{L}}^\star = {best_L:.4f}$ (epoch {best_e})",
                xy=(best_e, best_L), xytext=(-150, 14),
                textcoords="offset points", fontsize=7.5,
                arrowprops=dict(arrowstyle="-", lw=0.6, color="k"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"training loss $\mathcal{L}$")
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "loss.pdf")


# ----------------------------------------------------------------------
# Fig. S5, parity and residuals
# ----------------------------------------------------------------------
def fig_parity():
    par = D["parity"]
    acc = D["accuracy"]
    tr = np.asarray(D["split"]["train_idx"]) - 1     # 1-based in Julia
    ho = np.asarray(D["split"]["held_idx"]) - 1

    fig, ax = plt.subplots(2, 2, figsize=(6.6, 5.0))
    for k in range(2):
        yt = np.asarray(par["y_target"][k])
        yp = np.asarray(par["y_pred"][k])
        r = np.asarray(par["residual"][k]) * 1e3     # mm -> um

        a = ax[0, k]
        lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
        pad = 0.04 * (lim[1] - lim[0])
        lim = [lim[0] - pad, lim[1] + pad]
        a.plot(lim, lim, "-", lw=0.6, color="0.4", zorder=0)
        a.plot(yt[tr], yp[tr], ".", ms=2.5, color=BLUE,
               label=f"training ({tr.size})")
        a.plot(yt[ho], yp[ho], "o", ms=3.4, mfc="none", mew=0.8, color=RED,
               label=f"held out ({ho.size})")
        a.set_xlim(lim)
        a.set_ylim(lim)
        a.set_xlabel(rf"FE target {OUTLABEL[k]} (mm)")
        a.set_ylabel(rf"surrogate {OUTLABEL[k]} (mm)")
        a.legend(loc="upper left", handletextpad=0.4)
        a.text(0.97, 0.16, rf"$R^2_{{\mathrm{{train}}}} = {acc['train_out'][k]:.5f}$",
               transform=a.transAxes, ha="right", va="center", fontsize=7.5)
        a.text(0.97, 0.07, rf"$R^2_{{\mathrm{{held}}}} = {acc['held_out'][k]:.5f}$",
               transform=a.transAxes, ha="right", va="center", fontsize=7.5)

        a = ax[1, k]
        a.axhline(0.0, lw=0.6, color="0.4", zorder=0)
        a.plot(yt[tr], r[tr], ".", ms=2.5, color=BLUE)
        a.plot(yt[ho], r[ho], "o", ms=3.4, mfc="none", mew=0.8, color=RED)
        a.set_xlabel(rf"FE target {OUTLABEL[k]} (mm)")
        a.set_ylabel(r"residual ($\mu$m)")

    fig.tight_layout(w_pad=1.8, h_pad=1.4)
    save(fig, "parity.pdf")


# ----------------------------------------------------------------------
# Fig. S6, Sobol' indices over the full design box
# ----------------------------------------------------------------------
def _bars(a, names, s1, st, s1_err=None, st_err=None):
    pos = np.arange(len(names))
    a.bar(pos - 0.19, s1, width=0.36, color=BLUE, ec="k", lw=0.4,
          label=r"$S_1$",
          yerr=s1_err, error_kw=dict(lw=0.7, capsize=1.6, ecolor="0.2"))
    a.bar(pos + 0.19, st, width=0.36, color=RED, ec="k", lw=0.4,
          label=r"$S_T$",
          yerr=st_err, error_kw=dict(lw=0.7, capsize=1.6, ecolor="0.2"))
    a.set_xticks(pos)
    a.set_xticklabels([VARLABEL[n] for n in names])
    a.axhline(0.0, lw=0.6, color="k")
    a.set_ylim(min(0.0, 1.15 * min(s1.min(), 0.0)), 1.0)
    a.tick_params(top=False, right=False)


def fig_sobol():
    s = D["sobol"]
    names = s["varnames"]
    S1, ST = mat(s["S1"]), mat(s["ST"])
    S1lo, S1hi = mat(s["S1_lo"]), mat(s["S1_hi"])
    STlo, SThi = mat(s["ST_lo"]), mat(s["ST_hi"])

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.6), sharey=True)
    for k in range(2):
        order = np.argsort(ST[:, k])[::-1]
        nm = [names[i] for i in order]
        s1, st = S1[order, k], ST[order, k]
        e1 = np.vstack([s1 - S1lo[order, k], S1hi[order, k] - s1])
        et = np.vstack([st - STlo[order, k], SThi[order, k] - st])
        _bars(ax[k], nm, s1, st, np.abs(e1), np.abs(et))
        ax[k].set_title(rf"$y_{k + 1} = {OUTLABEL[k].strip('$')}$")
        if k == 0:
            ax[k].set_ylabel("Sobol' index")
            ax[k].legend(loc="upper right")
    fig.tight_layout(w_pad=1.0)
    save(fig, "sobol.pdf")


# ----------------------------------------------------------------------
# Fig. S7, Sobol' indices per device footprint
# ----------------------------------------------------------------------
def fig_sobol_family():
    fams = D["sobol_family"]
    fig, ax = plt.subplots(2, 2, figsize=(6.8, 4.4), sharey=True)
    for r, fam in enumerate(fams):
        names = fam["varnames"]
        S1, ST = mat(fam["S1"]), mat(fam["ST"])
        W, h, c = fam["triplet"]
        for k in range(2):
            a = ax[r, k]
            order = np.argsort(ST[:, k])[::-1]
            _bars(a, [names[i] for i in order], S1[order, k], ST[order, k])
            a.set_title(rf"{fam['name']}  $(W,h,c) = ({W:g},{h:g},{c:g})$ mm,"
                        rf"  $y_{k + 1} = {OUTLABEL[k].strip('$')}$",
                        fontsize=7.5)
            if k == 0:
                a.set_ylabel("Sobol' index")
            if r == 0 and k == 0:
                a.legend(loc="upper right")
    fig.tight_layout(w_pad=1.0, h_pad=1.4)
    save(fig, "sobol_family.pdf")


if __name__ == "__main__":
    fig_curation()
    fig_loss()
    fig_parity()
    fig_sobol()
    fig_sobol_family()
