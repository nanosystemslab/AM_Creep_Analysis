"""Shared figure styling for thesis plots.

Provides a single-source font hierarchy, Tol qualitative
palette, and a ``save_figure`` helper that defaults to vector PDF output.

Vendored from /Users/ethan/Desktop/figure_style/style.py for repo
self-containment (Zenodo archival).
"""

import itertools
import os

import matplotlib.pyplot as plt

# Tol qualitative palette (16 colours, colorblind-safe)
TOL_COLORS = [
    '#0077BB',  # blue
    '#EE7733',  # orange
    '#009988',  # teal
    '#CC3311',  # red
    '#33BBEE',  # cyan
    '#EE3377',  # magenta
    '#BBBBBB',  # grey
    '#000000',  # black
    '#44BB99',  # mint
    '#AA3377',  # wine
    '#332288',  # indigo
    '#88CCEE',  # light blue
    '#882255',  # dark pink
    '#117733',  # dark green
    '#DDCC77',  # sand
    '#999933',  # olive
]

MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', 'p', '*', '<', '>',
           '8', 'd', 'H']

FONT_TITLE = 28
FONT_SUBTITLE = 24
FONT_LABEL = 22
FONT_TICK = 18
FONT_LEGEND = 18


def apply_style(use_tex=True):
    """Set rcParams for thesis-quality figures.

    Font hierarchy:  18 pt title / 16 pt subtitle / 14 pt label+legend / 12 pt ticks.
    Serif + Computer Modern fonts.  Grid on, top/right spines off.
    """
    plt.rcParams.update({
        "text.usetex": use_tex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.titlesize": FONT_TITLE,
        "axes.titleweight": "bold",
        "figure.titlesize": FONT_TITLE,
        "figure.titleweight": "bold",
        "font.size": FONT_LABEL,
        "axes.labelsize": FONT_LABEL,
        "axes.labelweight": "bold",
        "legend.fontsize": FONT_LEGEND,
        "xtick.labelsize": FONT_TICK,
        "ytick.labelsize": FONT_TICK,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "lines.linewidth": 1.2,
        "lines.markersize": 6,
        "figure.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "legend.handlelength": 2,
    })


def save_figure(fig, path, formats=None, dpi=400, close=True):
    """Save *fig* as PDF (default) and/or PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    path : str
        Output path.  If it already has an extension (e.g. ``"out/plot.png"``),
        only that format is produced (backward-compat).  Otherwise the
        *formats* list is used (default ``["pdf"]``).
    formats : list[str] or None
        E.g. ``["pdf", "png"]``.  Ignored when *path* has an extension.
    dpi : int
        Resolution for raster formats.
    close : bool
        Close the figure after saving (default ``True``).
    """
    root, ext = os.path.splitext(path)
    if ext:
        fig.savefig(path, dpi=dpi)
    else:
        if formats is None:
            formats = ["pdf"]
        for fmt in formats:
            fig.savefig(f"{root}.{fmt}", dpi=dpi)
    if close:
        plt.close(fig)


def get_colors(n=None):
    """Return *n* colours from the Tol qualitative palette, cycling if needed.

    If *n* is ``None``, return the full 16-colour list.
    """
    if n is None:
        return list(TOL_COLORS)
    return [c for c, _ in zip(itertools.cycle(TOL_COLORS), range(n))]
