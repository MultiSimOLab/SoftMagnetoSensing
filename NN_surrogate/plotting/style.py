"""Colours and colour maps shared by the plotting scripts."""

from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb

C_U = "#0072B2"     # u_On
C_E = "#D55E00"     # eps_On
INK = "#1a1a1a"
INK2 = "#5c5c5c"
GRID = "#dcdcdc"

CMAP_U = LinearSegmentedColormap.from_list(
    "uon_seq", ["#eff5fb", "#d2e3f2", "#a8c8e5", "#75a5d1", "#3f7db5",
                "#0f5a90"])

RAMP_WINDOWS = LinearSegmentedColormap.from_list(
    "windows", ["#d9e8fa", "#bfd6f4", "#aec3ed", "#b1b0e4", "#bfa7dc",
                "#caa0d4", "#c898cd"])


def window_fills(n):
    return [to_hex(RAMP_WINDOWS(i / max(n - 1, 1))) for i in range(n)]


def darker(hexcol, f=0.62):
    return to_hex(tuple(f * c for c in to_rgb(hexcol)))


def mat(x):
    """JSON.jl writes a Julia (m x n) matrix as n arrays of length m."""
    import numpy as np
    return np.asarray(x, dtype=float).T
