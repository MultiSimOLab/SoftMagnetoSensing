#!/usr/bin/env python3
"""
fig4_panels.py

Main-text Fig. 4c-e from results/fig4_sensitivity_data.json, written by
scripts/fig4_sensitivity.jl. Plotting only, no computation.

  c  u_On over the (w, d_BC) plane, large footprint, phi_HM = 30 vol%, e = 2 mm
  d  total-order Sobol' indices of u_On and eps_On, small footprint, split
     into first-order part and interaction remainder
  e  trigger-strain range of the FE designs against reported strain windows

Usage:  python3 plotting/fig4_panels.py [fig4_sensitivity_data.json] [outdir]
"""

import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from style import C_U, C_E, CMAP_U, INK, INK2, GRID, window_fills, darker, mat

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
JSON_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    ROOT, "results", "fig4_sensitivity_data.json")
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "figures")
os.makedirs(OUTDIR, exist_ok=True)

D = json.load(open(JSON_PATH))
FAM = {f["name"]: f for f in D["families"]}
SRF = {s["name"]: s for s in D["surfaces"]}
VARNAMES = D["varnames"]

U_ON = r"$u_{\mathrm{On}}$"
EPS_ON = r"$\varepsilon_{\mathrm{On}}$"
D_BC = r"$d_{\mathrm{BC}}$"
VARLABEL = {"w": r"$w$", "e": r"$e$", "phi_HM": r"$\phi_{\mathrm{HM}}$",
            "d_BC": D_BC}
K_UON, K_EPSON = 0, 2

BASE_RC = {
    "font.family": "serif",
    "font.serif": ["STIXGeneral"],
    "mathtext.fontset": "stix",
    "legend.frameon": False,
    "figure.dpi": 200,
    "pdf.fonttype": 42,
}


def letter(ax, txt, dx, dy, size=11.0):
    ax.text(dx, dy, txt, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="top", ha="left", color=INK)


def save(fig, name):
    for ext in ("pdf", "png"):
        path = os.path.join(OUTDIR, f"{name}.{ext}")
        fig.savefig(path, dpi=600 if ext == "png" else None)
        print("wrote", path)
    plt.close(fig)


# ----------------------------------------------------------------------
# c: u_On over (w, d_BC)
# ----------------------------------------------------------------------
def panel_map(ax, cax, family="large", phi=30.0):
    srf = next(m for m in SRF[family]["maps"] if m["phi_HM"] == phi)["surface"]
    w = np.asarray(srf["w"], float)
    a = np.asarray(srf["d_BC"], float)
    Z = mat(srf["uOn"])

    im = ax.contourf(w, a, Z, levels=np.arange(1.2, 3.61, 0.1), cmap=CMAP_U,
                     extend="both")
    ax.contour(w, a, Z, levels=[1.5, 2.5], colors="white", linewidths=0.5)
    cs = ax.contour(w, a, Z, levels=[2.0, 3.0], colors="white", linewidths=0.55)
    ax.clabel(cs, fmt="%.1f", fontsize=5.6, inline=True, inline_spacing=2)

    f = FAM[family]
    ax.plot(f["fe_w"], f["fe_dBC"], "o", ms=1.1, mfc="none", mec="white",
            mew=0.35, alpha=0.75)

    ax.set_xlim(w.min(), w.max())
    ax.set_ylim(0, 1)
    ax.set_xticks([5, 7.5, 10, 12.5, 15])
    ax.set_xticklabels(["5", "7.5", "10", "12.5", "15"])
    ax.set_yticks([0, 0.5, 1.0])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xlabel(r"$w$  (mm)", labelpad=1.0)
    ax.set_ylabel(D_BC, labelpad=0.5)
    for side in ax.spines.values():
        side.set_linewidth(0.5)
        side.set_color(INK2)
    ax.tick_params(top=False, right=False)
    ax.text(0.022, 0.045, r"$\phi_{\mathrm{HM}} = 30$ vol%,  $e = 2$ mm",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=5.4,
            color=INK, zorder=6,
            bbox=dict(fc="white", ec="none", alpha=0.72, pad=0.9))

    cb = ax.figure.colorbar(im, cax=cax, orientation="vertical",
                            ticks=[1.5, 2.0, 2.5, 3.0, 3.5])
    cb.set_label(U_ON + " (mm)", fontsize=7.0, labelpad=1.0)
    cb.ax.tick_params(labelsize=6.0, width=0.4, length=1.5, pad=1.2)
    cb.outline.set_linewidth(0.4)
    letter(ax, "c", -0.30, 1.13)


# ----------------------------------------------------------------------
# d: Sobol' indices, small footprint
# ----------------------------------------------------------------------
def panel_bars(ax, family="small", order=("w", "d_BC", "phi_HM", "e")):
    idx = [VARNAMES.index(v) for v in order]
    ypos = np.arange(len(order))[::-1]
    h, gap = 0.30, 0.02
    s = FAM[family]["sobol"]
    S1, ST = mat(s["S1"]), mat(s["ST"])

    for k, colour, off in ((K_UON, C_U, +0.17), (K_EPSON, C_E, -0.17)):
        for yi, i in zip(ypos, idx):
            s1 = float(np.clip(S1[i, k], 0.0, None))
            st = float(max(ST[i, k], s1))
            ax.barh(yi + off, s1, height=h, color=colour, edgecolor="none",
                    zorder=3)
            if st - s1 > gap:
                ax.barh(yi + off, st - s1 - gap, height=h, left=s1 + gap,
                        color=colour, alpha=0.34, edgecolor="none", zorder=3)

    ax.legend(handles=[Patch(facecolor=C_U, label=U_ON),
                       Patch(facecolor=C_E, label=EPS_ON)],
              loc="center right", bbox_to_anchor=(1.02, 0.24), ncol=1,
              fontsize=6.6, handlelength=0.85, handleheight=0.85,
              handletextpad=0.35, borderpad=0.0, borderaxespad=0.0)
    ax.text(1.0, 1.04, "small device", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=6.4, color=INK2)

    ax.set_yticks(ypos)
    ax.set_yticklabels([VARLABEL[v] for v in order], fontsize=7.8)
    ax.set_ylim(-0.55, len(order) - 0.45)
    ax.set_xlim(0, 1.0)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xticklabels(["0", "0.5", "1"])
    for x in (0.5, 1.0):
        ax.axvline(x, color=GRID, lw=0.4, zorder=0)
    ax.set_xlabel(r"Sobol' index  $S^{T}$", labelpad=1.0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(axis="y", length=0)
    letter(ax, "d", -0.30, 1.16)


def fig_cd():
    with plt.rc_context({**BASE_RC,
                         "font.size": 7.0, "axes.linewidth": 0.5,
                         "axes.labelsize": 7.2, "xtick.labelsize": 6.6,
                         "ytick.labelsize": 6.6, "xtick.major.width": 0.5,
                         "ytick.major.width": 0.5, "xtick.major.size": 1.8,
                         "ytick.major.size": 1.8, "xtick.major.pad": 1.6,
                         "ytick.major.pad": 1.6}):
        fig = plt.figure(figsize=(161.3 / 72, 145.6 / 72))
        ax_map = fig.add_axes([0.230, 0.650, 0.585, 0.292])
        cax = fig.add_axes([0.833, 0.650, 0.034, 0.292])
        ax_bar = fig.add_axes([0.230, 0.162, 0.585, 0.318])
        panel_map(ax_map, cax)
        panel_bars(ax_bar)
        save(fig, "fig4_cd")


# ----------------------------------------------------------------------
# e: attainable trigger strains against reported strain windows
# ----------------------------------------------------------------------
def fig_e():
    with plt.rc_context({**BASE_RC,
                         "font.size": 10.0, "axes.linewidth": 0.8,
                         "axes.labelsize": 11.0, "xtick.labelsize": 9.5,
                         "ytick.labelsize": 10.5, "xtick.major.width": 0.8,
                         "xtick.major.size": 3.4}):
        windows = D["clinical"]
        n = len(windows)
        ypos = np.arange(n)[::-1]
        fills = window_fills(n)

        fig, ax = plt.subplots(figsize=(7.4, 3.5))

        eps_all = np.concatenate([np.asarray(FAM[k]["fe_epsOn"], float)
                                  for k in ("large", "small")])
        band = (eps_all.min(), eps_all.max())
        ax.axvspan(*band, color=C_E, alpha=0.10, zorder=0, lw=0)
        for x in band:
            ax.axvline(x, color=C_E, lw=0.8, ls=(0, (2.5, 1.6)), zorder=1)

        for yi, cw, fc in zip(ypos, windows, fills):
            a, b = cw["range"]
            ax.add_patch(Rectangle((a, yi - 0.32), b - a, 0.64, facecolor=fc,
                                   edgecolor=darker(fc), lw=0.9, zorder=3))
            txt = f"{a * 100:g}-{b * 100:g}%"
            if b > band[1] - 0.03:
                ax.text(b - 0.004, yi, txt, va="center", ha="right",
                        fontsize=9.0, color=INK, zorder=5)
            else:
                ax.text(b + 0.005, yi, txt, va="center", ha="left",
                        fontsize=9.0, color=INK2, zorder=5,
                        bbox=dict(fc="white", ec="none", alpha=0.85, pad=0.6))

        ax.plot(eps_all, np.full(eps_all.size, ypos[0] + 0.75), "o", ms=8.0,
                mfc=C_E, mec="none", alpha=0.28, zorder=4)
        ax.text(band[0] - 0.005, ypos[0] + 0.75, "FE designs", va="center",
                ha="right", fontsize=10.5, color=C_E)
        ax.text(np.mean(band), ypos[0] + 1.50,
                f"attainable with this design box   {band[0]:.3f}-{band[1]:.3f}",
                ha="center", va="center", fontsize=9.5, color=C_E)

        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{cw['label']} ({cw['site']})" for cw in windows],
                           fontsize=10.5)
        ax.set_ylim(-0.55, n + 0.95)
        ax.set_xlim(0, 0.172)
        ax.set_xticks([0, 0.05, 0.10, 0.15])
        ax.set_xlabel(r"trigger strain  $\varepsilon_{\mathrm{On}}$")
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK2)
        ax.tick_params(axis="x", pad=2.5)
        ax.tick_params(axis="y", length=0)
        fig.tight_layout(pad=0.9)
        save(fig, "fig4_e")


if __name__ == "__main__":
    fig_cd()
    fig_e()
