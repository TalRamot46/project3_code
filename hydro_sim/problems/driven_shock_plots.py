# problems/driven_shock_plots.py
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
from driven_shock_sim import ShockHistory
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

def save_driven_shock_gif(history: ShockHistory, case, gif_path="driven_shock.gif", fps=20, stride=1):
    fig, axs = plt.subplots(4, 1, figsize=(7, 10), sharex=True)

    k0 = 0
    lines = []
    lines.append(axs[0].plot(history.x[k0], history.rho[k0], lw=2)[0])
    lines.append(axs[1].plot(history.x[k0], history.p[k0], lw=2)[0])
    lines.append(axs[2].plot(history.x[k0], history.u[k0], lw=2)[0])
    lines.append(axs[3].plot(history.x[k0], history.e[k0], lw=2)[0])

    axs[0].set_ylabel(r"$\rho$")
    axs[1].set_ylabel(r"$p$")
    axs[2].set_ylabel(r"$u$")
    axs[3].set_ylabel(r"$e$")
    axs[3].set_xlabel(r"$x$")

    for ax in axs:
        ax.grid(True)

    title = fig.suptitle("", fontsize=12)
    frame_ids = np.arange(0, len(history.t), stride)

    def init():
        return lines

    def update(frame_idx):
        k = int(frame_ids[frame_idx])
        lines[0].set_data(history.x[k], history.rho[k])
        lines[1].set_data(history.x[k], history.p[k])
        lines[2].set_data(history.x[k], history.u[k])
        lines[3].set_data(history.x[k], history.e[k])
        t = history.t[k]
        title.set_text(
            f"{case.name}\n"
            f"$p(0,t)=P_0 t^{{\\tau}},\\; P_0={case.P0},\\; \\tau={case.tau},\\; t={t:.3e}$"
        )

        for ax in axs:
            ax.relim()
            ax.autoscale_view()

        return lines
    anim = FuncAnimation(fig, update, frames=len(frame_ids), init_func=init, blit=False)
    anim.save(gif_path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved GIF to {gif_path}")

def plot_driven_shock_slider(history: ShockHistory, case, savepath=None, show=True):
    # initial frame index
    k0 = 0

    fig, axs = plt.subplots(4, 1, figsize=(7, 10), sharex=True)
    plt.subplots_adjust(bottom=0.12)  # room for slider

    # initial lines
    l_rho, = axs[0].plot(history.x[k0], history.rho[k0], lw=2)
    l_p,   = axs[1].plot(history.x[k0], history.p[k0], lw=2)
    l_u,   = axs[2].plot(history.x[k0], history.u[k0], lw=2)
    l_e,   = axs[3].plot(history.x[k0], history.e[k0], lw=2)

    axs[0].set_ylabel(r"$\rho$")
    axs[1].set_ylabel(r"$p$")
    axs[2].set_ylabel(r"$u$")
    axs[3].set_ylabel(r"$e$")
    axs[3].set_xlabel(r"$x$")

    for ax in axs:
        ax.grid(True)

    title = fig.suptitle("", fontsize=12)

    def set_title(k):
        t = history.t[k]
        title.set_text(
            f"{case.name}\n"
            f"$p(0,t)=P_0 t^{{\\tau}},\\; P_0={case.P0},\\; \\tau={case.tau},\\; t={t:.3e}$"
        )

    set_title(k0)

    # slider: use frame index for simplicity
    ax_slider = fig.add_axes([0.15, 0.04, 0.7, 0.03])
    slider = Slider(ax_slider, "frame", 0, len(history.t) - 1, valinit=k0, valstep=1)

    def update(val):
        k = int(slider.val)
        l_rho.set_data(history.x[k], history.rho[k])
        l_p.set_data(history.x[k], history.p[k])
        l_u.set_data(history.x[k], history.u[k])
        l_e.set_data(history.x[k], history.e[k])
        set_title(k)

        # autoscale y (optional). Comment out if you want fixed axes.
        for ax in axs:
            ax.relim()
            ax.autoscale_view()

        fig.canvas.draw_idle()

    slider.on_changed(update)

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])

    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"Saved slider figure (static) to {savepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, axs

def plot_driven_shock_profiles(x_cells, rho, p, u, e, case, t, savepath=None, show=True):
    fig, axs = plt.subplots(4, 1, figsize=(7, 10), sharex=True)

    m = x_cells * 1.932e1

    axs[0].plot(m, rho, lw=2)
    axs[0].set_ylabel(r"$\rho$")

    axs[1].plot(m, p, lw=2)
    axs[1].set_ylabel(r"$p$")

    axs[2].plot(m, u, lw=2)
    axs[2].set_ylabel(r"$u$")

    axs[3].plot(m, e, lw=2)
    axs[3].set_ylabel(r"$e$")
    axs[3].set_xlabel(r"$x$")

    for ax in axs:
        ax.grid(True)

    fig.suptitle(
        f"{case.name}\n"
        f"$p(0,t)=P_0 t^{{\\tau}},\\; P_0={case.P0},\\; \\tau={case.tau},\\; t={t:.3e}$",
        fontsize=12
    )

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    if savepath:
        fig.savefig(savepath, dpi=200)
        print(f"Saved figure to {savepath}")

    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, axs