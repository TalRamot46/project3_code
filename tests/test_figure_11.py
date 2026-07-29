"""
Reproduction of Figure 11 of

    Menahem Krief, "Analytic solutions of the nonlinear radiation diffusion
    equation with an instantaneous point source in non-homogeneous media",
    Physics of Fluids 33, 057105 (2021); doi: 10.1063/5.0050422.

Figure 11 compares the analytic solution [Eq. (42)] against numerical
simulations of the radiation diffusion equation with an instantaneous point
source, for six values of the spatial density exponent omega.

Setup, taken verbatim from the paper (Sec. VI and the Fig. 11 caption):

- Spherical symmetry, d = 3.
- alpha = 2, beta = 1.6, lambda = 1, mu = 0  =>  n = 2.75 (nonlinear conduction).
- Power law opacity [Eq. (4)] with g = 1 and EOS [Eq. (5)] with f = 1.
- Spatial power law density [Eq. (6)] with rho_0 = 1.
- Total energy Q = 1 deposited at the origin at t = 0.
- Temperature profiles *in Kelvin*, shown at the times at which the heat wave
  [Eq. (43)] reaches radii r = 0.225, 0.45, 0.675 and 0.9.
- All quantities in c.g.s. units.

The four times per panel are not free parameters: each is obtained by
inverting the heat front trajectory r_h(t) = xi_0 (Q^n A t)^(1/p) for t, which
is what ``heat_front_time`` does.

Note on omega = 2/3: for these parameters omega_c = 2/3 exactly, so this panel
sits on the marginal case 2 - k - m = 0 where Eqs. (42), (44) and (45)
degenerate and the logarithmic form of Eqs. (49)-(50) applies instead.

Run ``python tests/test_figure_11.py`` to write the figure to
``results/figure_11_reproduction.png``; run under pytest for the quantitative
checks alone.
"""

from pathlib import Path

import numpy as np
import pytest
import scipy.integrate

from menahem_new.supersonic_instantaneous_analytic import (
    SupersonicInstantaneousAnalytic,
)

# ---------------------------------------------------------------------------
# Paper parameters (Sec. VI / Fig. 11 caption)
# ---------------------------------------------------------------------------

FIG11_MATERIAL = dict(
    g=1.0,
    alpha=2.0,
    lambdap=1.0,
    f=1.0,
    beta=1.6,
    mu=0.0,
    rho0=1.0,
    Q=1.0,
    d=3,
)

# Panel order matches the figure: left-to-right, top-to-bottom.
FIG11_OMEGAS = [1.5, 2.0 / 3.0, 0.3, 0.0]

# Front radii the snapshots are taken at.
FIG11_RADII = [0.225, 0.45, 0.675, 0.9]

# y-axis limits of each panel in the published figure, keyed by omega.
FIG11_YLIM = {
    1.5: 6.0,
    2.0 / 3.0: 6.0,
    0.3: 7.0,
    0.0: 9.0,
    -1.0: 25.0,
    -3.0: 200.0,
}


def make_solver(omega: float) -> SupersonicInstantaneousAnalytic:
    """Solver for one Fig. 11 panel."""
    return SupersonicInstantaneousAnalytic(omega=omega, **FIG11_MATERIAL)


# ---------------------------------------------------------------------------
# Simulation side of the comparison
# ---------------------------------------------------------------------------
#
# The paper deposits Q in the first cell at t = 0. That exact initial state is
# a delta function, which no grid can represent, so the simulation is instead
# started from the exact self-similar profile at a time when the front is
# already resolved (r_h = FIG11_SEED_RADIUS) and left to evolve on its own.
#
# The inner boundary is the symmetry condition of Eq. (21) -- zero energy flux
# at the origin -- which is what a point source actually obeys for t > 0, and
# the only closure available for omega >= omega_c where T(0,t) diverges.

# The seed is placed at a fixed *fraction* of each target radius rather than at
# a fixed radius. The front advances as r ~ t^(1/p), so a fixed seed radius
# makes the simulated time span (r_target/r_seed)^p, and p grows quickly as
# omega goes negative (p = 16 at omega = -1, p = 27.5 at omega = -3). Seeding at
# r = 0.05 would then ask for ~1e20 and ~1e34 in elapsed time respectively,
# which the timestep controller can only cross in an impractical number of
# steps. Seeding proportionally caps the span at (1/FIG11_SEED_FRACTION)^p for
# every panel while still letting the front travel a factor of 1/0.4 = 2.5x.
FIG11_SEED_FRACTION = 0.4
FIG11_SIM_N = 80
FIG11_X_MIN = 1e-5
FIG11_X_MAX = 1.0


def make_fig11_case(omega: float, r_target: float):
    """Build the spherical (d=3) RadHydroCase for one curve of one panel."""
    from hydro_sim.core.geometry import spherical
    from rad_hydro_sim.problems.RadHydroCase import RadHydroCase

    solver = make_solver(omega)
    return RadHydroCase(
        g_Kelvin=FIG11_MATERIAL["g"],
        alpha=FIG11_MATERIAL["alpha"],
        lambda_=FIG11_MATERIAL["lambdap"],
        f_Kelvin=FIG11_MATERIAL["f"],
        beta_Rosen=FIG11_MATERIAL["beta"],
        mu=FIG11_MATERIAL["mu"],
        chi=1.0,
        T0_Kelvin=None,            # -> zero-flux (symmetry) inner boundary
        P0_Barye=None,
        tau=solver.tau,
        rho0=FIG11_MATERIAL["rho0"],
        p0=None,
        u0=None,
        T_initial_Kelvin=1e-6,     # numerically cold ambient
        r=0.25,
        x_min=FIG11_X_MIN,
        x_max=FIG11_X_MAX,
        t_sec_start=float(solver.heat_front_time(FIG11_SEED_FRACTION * r_target)),
        t_sec_end=float(solver.heat_front_time(r_target)),
        initial_condition="analytic_supersonic_instantaneous",
        scenario="radiation_only",
        geom=spherical(),          # d = 3, angular symmetry
        omega=omega,
        force_black="conduction",  # Krief Eq. (1)
        Q_point_source=FIG11_MATERIAL["Q"],
    )


def run_fig11_simulation(omega: float, r_target: float, N: int = FIG11_SIM_N):
    """Evolve the wave to the time its front should reach ``r_target``.

    Returns ``(r_centers, T, energy_ratio)``, the last being the simulated
    total energy divided by Q (1.0 if the scheme conserved energy exactly).
    """
    from hydro_sim.problems.simulation_config import SimulationConfig
    from rad_hydro_sim.simulation.iterator import simulate_rad_hydro

    case = make_fig11_case(omega, r_target)
    config = SimulationConfig(N=N, CFL=1.0 / 3.0, sigma_visc=1.0, store_every=10)
    _, state, _, _ = simulate_rad_hydro(rad_hydro_case=case, simulation_config=config)

    r_nodes = np.asarray(state.x, dtype=float)
    r_cent = 0.5 * (r_nodes[:-1] + r_nodes[1:])
    T = np.asarray(state.T_material, dtype=float)

    # Total energy int u dV with the spherical volume element.
    volume = (4.0 * np.pi / 3.0) * (r_nodes[1:] ** 3 - r_nodes[:-1] ** 3)
    u = (
        FIG11_MATERIAL["f"]
        * T ** FIG11_MATERIAL["beta"]
        * np.asarray(state.rho, dtype=float) ** (1.0 - FIG11_MATERIAL["mu"])
    )
    energy_ratio = float(np.sum(u * volume) / FIG11_MATERIAL["Q"])

    return r_cent, T, energy_ratio


# ---------------------------------------------------------------------------
# Quantitative checks
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("omega", FIG11_OMEGAS)
def test_derived_exponents_match_paper(omega):
    """n, omega_max, omega_c and omega_0 must match the values quoted in the text."""
    solver = make_solver(omega)

    assert solver.n == pytest.approx(2.75)

    n, lam, mu_, d = solver.n, solver.lambdap, solver.mu, solver.d
    omega_max = (2.0 + d * n) / (1.0 + lam + (1.0 - mu_) * (1.0 + n))   # Eq. (35)
    omega_0 = 1.0 / (2.0 + lam - mu_)                                    # Eq. (46)
    omega_c = 2.0 * omega_0                                              # Eq. (48)

    assert omega_max == pytest.approx(1.7826, abs=1e-4)
    assert omega_c == pytest.approx(2.0 / 3.0)
    assert omega_0 == pytest.approx(1.0 / 3.0)


@pytest.mark.parametrize("omega", FIG11_OMEGAS)
def test_front_times_reproduce_requested_radii(omega):
    """heat_front_time must invert the front trajectory of Eq. (43) exactly."""
    solver = make_solver(omega)
    for r in FIG11_RADII:
        t = solver.heat_front_time(r)
        assert t > 0.0
        assert solver.heat_front_radius(t) == pytest.approx(r, rel=1e-10)


@pytest.mark.parametrize("omega", FIG11_OMEGAS)
def test_energy_conservation(omega):
    """The deposited energy must integrate back to Q at every snapshot time.

    This is Eq. (9): Q = A_d * int u(r,t) r^(d-1) dr, and it is the constraint
    that fixes xi_0, so it exercises Eqs. (44)-(45) (or (50) when marginal).
    """
    solver = make_solver(omega)
    for r_front in FIG11_RADII:
        t = solver.heat_front_time(r_front)
        # Integrate up to the front; the solution is identically zero beyond it.
        # A geometric grid is needed because for omega >= omega_c the integrand
        # carries an integrable singularity at the origin (~r^-0.41 at
        # omega = 1.5), which a uniform grid resolves poorly.
        r = np.geomspace(1e-14, r_front * (1.0 - 1e-12), 200_000)
        u = solver.energy_density(r, t)
        integrand = solver.Ad * u * r ** (solver.d - 1.0)
        Q_num = scipy.integrate.simpson(integrand, r)
        assert Q_num == pytest.approx(solver.Q, rel=2e-3)


@pytest.mark.parametrize("omega", FIG11_OMEGAS)
def test_temperature_vanishes_beyond_front(omega):
    """T must be exactly zero ahead of the heat front (compact support)."""
    solver = make_solver(omega)
    t = solver.heat_front_time(FIG11_RADII[0])
    r_front = solver.heat_front_radius(t)
    r_outside = np.linspace(r_front * 1.0001, r_front * 3.0, 500)
    assert np.all(solver.temperature_profile(r_outside, t) == 0.0)


def test_marginal_case_is_detected_and_logarithmic():
    """omega = omega_c = 2/3 must use the Eq. (49)-(50) branch, not Eq. (42)."""
    marginal = make_solver(2.0 / 3.0)
    assert marginal.marginal
    assert np.isclose(2.0 - marginal.k - marginal.m, 0.0, atol=1e-12)
    # Eq. (49) diverges logarithmically at the origin (paper Table I).
    assert marginal.f_0 == np.inf
    assert marginal.self_similar_profile(marginal.xi_0 * 1e-6) > marginal.self_similar_profile(
        marginal.xi_0 * 1e-1
    )
    # Neighbouring omegas must not take that branch.
    assert not make_solver(0.3).marginal
    assert not make_solver(1.5).marginal


def test_profile_shape_is_monotonic_decreasing():
    """Inside the front the temperature must fall monotonically outward."""
    for omega in FIG11_OMEGAS:
        solver = make_solver(omega)
        t = solver.heat_front_time(0.45)
        r = np.linspace(1e-6, solver.heat_front_radius(t) * 0.999, 3000)
        T = np.asarray(solver.temperature_profile(r, t), dtype=float)
        assert np.all(np.diff(T) <= 1e-9 * np.max(T))


# ---------------------------------------------------------------------------
# Simulation vs analytic
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("omega", FIG11_OMEGAS)
def test_simulation_conserves_energy(omega):
    """The d=3 solve must hold the deposited energy Q to a fraction of a percent.

    This exercises the spherical area-weighted divergence: with a planar
    operator the enclosed volume is wrong by a factor ~r^2 and the energy
    drifts immediately.
    """
    _, _, energy_ratio = run_fig11_simulation(omega, FIG11_RADII[1])
    assert energy_ratio == pytest.approx(1.0, abs=0.02)


@pytest.mark.slow
@pytest.mark.parametrize("omega", FIG11_OMEGAS)
def test_simulation_tracks_analytic_profile(omega):
    """The simulated profile must follow the analytic one across the wave.

    Compared on the analytic solution's own support, away from the front where
    the profile turns over vertically and a small offset in front position
    would dominate a pointwise comparison.
    """
    solver = make_solver(omega)
    r_target = FIG11_RADII[1]
    t = solver.heat_front_time(r_target)

    r_sim, T_sim, _ = run_fig11_simulation(omega, r_target)
    T_ana = np.asarray(solver.temperature_profile(r_sim, t), dtype=float)

    inside = (r_sim > 1.2 * FIG11_SEED_FRACTION * r_target) & (r_sim < 0.8 * r_target)
    assert np.count_nonzero(inside) > 10

    rel = np.abs(T_sim[inside] - T_ana[inside]) / np.max(T_ana[inside])
    assert np.median(rel) < 0.05


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_figure_11(
    savepath: str | Path | None = None,
    show: bool = False,
    include_simulation: bool = True,
    N: int = FIG11_SIM_N,
):
    """Render the six-panel Fig. 11 reproduction.

    With ``include_simulation`` the numerical solution is overlaid on the
    analytic one, in the paper's convention (simulation solid red, analytic
    dashed blue).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(3, 2, figsize=(11, 13))

    for ax, omega in zip(axes.ravel(), FIG11_OMEGAS):
        solver = make_solver(omega)

        for r_front in FIG11_RADII:
            t = solver.heat_front_time(r_front)

            if include_simulation:
                r_sim, T_sim, e_ratio = run_fig11_simulation(omega, r_front, N=N)
                ax.plot(r_sim, T_sim, color="red", linewidth=1.4)
                print(f"  omega={omega:<9.4g} r={r_front:<6} E/Q={e_ratio:.4f}")

            # Start just off the origin: for omega >= omega_c the analytic
            # profile genuinely diverges there (paper Table I).
            r = np.linspace(1e-6, 1.0, 4000)
            T = solver.temperature_profile(r, t)
            ax.plot(r, T, color="blue", linestyle="--", linewidth=1.4)

        ax.set_title(rf"$\omega = {omega:g}$" if omega != FIG11_OMEGAS[1] else r"$\omega = 0.666667$")
        ax.set_xlabel(r"$r$")
        ax.set_ylabel(r"$T(r,t)$")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, FIG11_YLIM[omega])
        ax.grid(True, linestyle="--", alpha=0.4)

    handles = [
        Line2D([], [], color="red", linewidth=1.4, label="Simulation"),
        Line2D([], [], color="blue", linestyle="--", linewidth=1.4, label="Analytic"),
    ]
    axes.ravel()[0].legend(handles=handles, loc="upper right", fontsize=9)

    fig.suptitle(
        "Reproduction of Krief (2021) Fig. 11 — simulation vs analytic [Eq. (42)]\n"
        r"$d=3$ (spherical), $\alpha=2$, $\beta=1.6$, $\lambda=1$, $\mu=0$, $n=2.75$, "
        r"$\rho_0=1$, $Q=1$; fronts at $r=0.225,\,0.45,\,0.675,\,0.9$",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))

    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=140)
        print(f"Saved figure to {savepath}")
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent
    plot_figure_11(savepath=repo_root / "results" / "figure_11_reproduction.png", show=True)