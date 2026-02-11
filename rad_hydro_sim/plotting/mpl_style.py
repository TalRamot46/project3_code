# plotting/mpl_style.py
"""
Central matplotlib style for the project. Import this module before creating figures
to apply classic style, sans-serif font, and consistent rcParams project-wide.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

plt.style.use("classic")
matplotlib.rcParams.update(
    {
        "font.size": 10,
        "font.family": "sans-serif",
        "font.sans-serif": [
            "DejaVu Sans",
            "Arial",
            "Helvetica",
            "Liberation Sans",
            "sans-serif",
        ],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "savefig.dpi": 150,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
    }
)


def apply_mpl_style() -> None:
    """
    Apply project-wide matplotlib style. Safe to call multiple times.
    Style is applied on module import; this function exists for explicit use.
    """
    plt.style.use("classic")
    matplotlib.rcParams.update(
        {
            "font.size": 10,
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Arial",
                "Helvetica",
                "Liberation Sans",
                "sans-serif",
            ],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 100,
            "savefig.dpi": 150,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )
