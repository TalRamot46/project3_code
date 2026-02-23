# plotting/mpl_style.py
"""
Central matplotlib style for the project. Import this module before creating figures
to apply classic style, serif font (no Sunserif), and consistent rcParams project-wide.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

plt.style.use("classic")
# Serif font for professional look; avoid Sunserif (problematic on some systems)
matplotlib.rcParams.update(
    {
        "font.size": 11,
        "font.family": "serif",
        "font.serif": [
            "DejaVu Serif",
            "Times New Roman",
            "Times",
            "Liberation Serif",
            "serif",
        ],
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "black",
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        # Consistent curve colors (color cycle)
        "axes.prop_cycle": matplotlib.cycler(
            color=[
                "#1f77b4",
                "#d62728",
                "#2ca02c",
                "#ff7f0e",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
            ]
        ),
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
            "font.size": 11,
            "font.family": "serif",
            "font.serif": [
                "DejaVu Serif",
                "Times New Roman",
                "Times",
                "Liberation Serif",
                "serif",
            ],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.dpi": 100,
            "savefig.dpi": 200,
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "axes.prop_cycle": matplotlib.cycler(
                color=[
                    "#1f77b4",
                    "#d62728",
                    "#2ca02c",
                    "#ff7f0e",
                    "#9467bd",
                    "#8c564b",
                    "#e377c2",
                    "#7f7f7f",
                ]
            ),
        }
    )
