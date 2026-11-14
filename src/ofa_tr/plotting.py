"""Shared figure style (validated categorical palette, hairline chrome, Turkish labels)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# categorical slots 1-3 (validated all-pairs, light mode); colour follows the configuration everywhere
SERIES = {"ofa-adapt": "#2a78d6", "ofa": "#eb6834", "static": "#1baf7a"}
SERIES_LABEL = {"ofa-adapt": "OFA-Adapt", "ofa": "OFA", "static": "Static"}
ACCENT = "#2a78d6"
INK = {"primary": "#0b0b0b", "secondary": "#52514e", "muted": "#898781"}
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
BLUE_RAMP = ["#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]


def apply_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.75, "axes.labelcolor": INK["secondary"],
        "axes.titlesize": 9, "axes.titlecolor": INK["primary"], "axes.titleweight": "normal",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": INK["muted"], "ytick.color": INK["muted"], "xtick.labelcolor": INK["secondary"],
        "ytick.labelcolor": INK["secondary"], "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.6,
        "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
    })
