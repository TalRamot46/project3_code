# plot_shock_profiles.py
#
# Load shock_profiles.npz and plot a single publication-quality figure
# with multiple time snapshots on the same axes.
#
# Usage examples:
#   python plot_shock_profiles.py shock_profiles.npz
#   python plot_shock_profiles.py shock_profiles.npz --quantity P --xaxis m
#   python plot_shock_profiles.py shock_profiles.npz --quantity T --xaxis xi
#   python plot_shock_profiles.py shock_profiles.npz --save shock_profiles_P_vs_m.png

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _as_list(arr):
    """
    shock_profiles.npz might store profiles as:
      - object arrays of 1D arrays (from the lightweight version), OR
      - 2D numeric arrays (nt x N) from older preallocated code.

    This function returns a list of 1D arrays, one per time.
    """
    if isinstance(arr, list):
        return arr
    arr = np.asarray(arr)
    if arr.dtype == object:
        return [np.asarray(v, float) for v in arr.tolist()]
    if arr.ndim == 2:
        return [arr[i, :].astype(float, copy=False) for i in range(arr.shape[0])]
    if arr.ndim == 1:
        # already one profile (single time)
        return [arr.astype(float, copy=False)]
    raise ValueError(f"Unsupported array shape for profiles: {arr.shape}")


def load_shock_npz(npz_path: str | Path) -> dict:
    npz_path = Path(npz_path)
    data = np.load(npz_path, allow_pickle=True)

    out = {k: data[k] for k in data.files}

    # Normalize key names (support either P0_eff or P0)
    if "P0_eff" not in out and "P0" in out:
        out["P0_eff"] = out["P0"]

    # Required-ish keys
    if "times" not in out:
        raise KeyError("NPZ is missing 'times'.")
    if "t" not in out and "xi" not in out:
        raise KeyError("NPZ is missing 't' (xi grid).")

    return out


def plot_profiles(npz_path: str | Path, *, quantity: str, xaxis: str,
                  save: str | None = None, show: bool = True,
                  title: str | None = None, linewidth: float = 2.0):
    D = load_shock_npz(npz_path)

    times = np.asarray(D["times"], float)
    xi_grid = np.asarray(D.get("t", D.get("xi")), float)

    # Map quantity -> stored key and y-label
    q = quantity.lower()
    if q in ("p", "pressure"):
        key = "P_shock"
        ylabel = r"Pressure $P$"
    elif q in ("u", "velocity"):
        key = "u_shock"
        ylabel = r"Velocity $u$"
    elif q in ("rho", "density"):
        key = "rho_shock"
        ylabel = r"Density $\rho$"
    elif q in ("t", "temp", "temperature"):
        key = "T_shock"
        ylabel = r"Temperature $T$"
    else:
        raise ValueError("quantity must be one of: P, u, rho, T")

    if key not in D:
        raise KeyError(f"NPZ is missing '{key}'.")

    y_list = _as_list(D[key])

    # X axis selection
    xa = xaxis.lower()
    if xa in ("m", "mass"):
        if "m_shock" not in D:
            raise KeyError("Requested xaxis='m' but NPZ is missing 'm_shock'.")
        x_list = _as_list(D["m_shock"])
        xlabel = r"Mass coordinate $m$"
    elif xa in ("xi", "t"):
        # same xi grid for all times
        x_list = [xi_grid for _ in range(len(y_list))]
        xlabel = r"Self-similar coordinate $\xi$"
    elif xa in ("x", "position"):
        if "x_shock" not in D:
            raise KeyError("Requested xaxis='x' but NPZ is missing 'x_shock'.")
        x_list = _as_list(D["x_shock"])
        xlabel = r"Position $x$"
    else:
        raise ValueError("xaxis must be one of: m, xi, x")

    # Basic sanity: align counts
    nt = min(len(times), len(y_list), len(x_list))
    times = times[:nt]
    y_list = y_list[:nt]
    x_list = x_list[:nt]

    # --- Figure style (paper-friendly) ---
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.dpi": 150,
    })

    fig, ax = plt.subplots(figsize=(7.2, 4.6))

    # Use a perceptually uniform colormap across time
    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.1, 0.9, nt))

    for i in range(nt):
        x = np.asarray(x_list[i], float)
        y = np.asarray(y_list[i], float)

        # Some profiles may be stored as row/col; flatten for safety
        x = x.reshape(-1)
        y = y.reshape(-1)

        # Skip if bad
        if x.size < 2 or y.size < 2 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            continue

        # If m-based, it’s common that m is monotonic; if not, sort for nicer lines
        if xa in ("m", "mass") and x.size == y.size:
            order = np.argsort(x)
            x = x[order]
            y = y[order]

        ax.plot(
            x, y,
            linewidth=linewidth,
            color=colors[i],
            label=fr"$t={times[i]:g}$"
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None:
        title = f"{quantity.upper()} profiles at multiple times"
    ax.set_title(title)

    ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.6)

    # Legend: compact, paper-friendly
    ax.legend(
        ncols=2 if nt >= 6 else 1,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        borderpad=0.6,
        handlelength=2.0,
    )

    fig.tight_layout()

    if save is not None:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        print(f"Saved figure to: {save_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    USE_PARSER = False
    if USE_PARSER:  
        p = argparse.ArgumentParser(description="Plot profiles from shock_profiles.npz (paper-quality single figure).")
        p.add_argument("npz", type=str, help="Path to shock_profiles.npz")
        p.add_argument("--quantity", type=str, default="P", help="Which profile to plot: P, u, rho, T (default: P)")
        p.add_argument("--xaxis", type=str, default="m", help="X-axis: m or xi (default: m)")
        p.add_argument("--save", type=str, default=None, help="Save figure to this path (e.g., fig.png or fig.pdf)")
        p.add_argument("--no-show", action="store_true", help="Do not display the plot window")
        p.add_argument("--title", type=str, default=None, help="Custom figure title")
        args = p.parse_args()
    else:
        class Args:
            npz = "project_3/shussman_shock_solver/shock_profiles.npz"
            quantity = "P"
            xaxis = "x"
            save = None
            no_show = False
            title = None
        args = Args()

    plot_profiles(
        args.npz,
        quantity=args.quantity,
        xaxis=args.xaxis,
        save=args.save,
        show=(not args.no_show),
        title=args.title,
    )

    # save a test figure
    plot_profiles(
        args.npz,
        quantity="P",
        xaxis="x",
        save="project_3/shussman_shock_solver/shock_profiles_P_vs_x.png",
        show=False,
        title="Pressure profiles at multiple times",
    )


if __name__ == "__main__":
    main()
