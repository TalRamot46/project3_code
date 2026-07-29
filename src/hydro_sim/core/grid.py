from dataclasses import dataclass
import numpy as np
from .geometry import Geometry

@dataclass
class Grid1D:
    x_nodes: np.ndarray   # nodes positions (Nnodes,)
    m_cells: np.ndarray   # cell masses (Ncells,)
    geom: Geometry

def cell_volumes(x_nodes: np.ndarray, geom: Geometry) -> np.ndarray:
    # PDF Eq.(13): V = zeta * (r_{i+1}^{alpha+1} - r_i^{alpha+1})
    a = geom.alpha
    return geom.zeta * (x_nodes[1:]**(a+1) - x_nodes[:-1]**(a+1))

def make_uniform_nodes(x_min: float, x_max: float, Ncells: int) -> np.ndarray:
    # Ncells = 1000 => Nnodes=1001
    return np.linspace(x_min, x_max, Ncells + 1)

def _limit_width_ratio(dx: np.ndarray, max_ratio: float) -> np.ndarray:
    """Largest width profile <= ``dx`` with neighbouring widths within ``max_ratio``.

    Two monotone sweeps, each of which only lowers widths: forward enforces
    dx[k] <= dx[k-1]*R, backward enforces dx[k] <= dx[k+1]*R. Running the
    backward sweep second preserves the forward property, so the result obeys
    the bound on both sides. Done in log space, where the closed form
    dx[k] = min_j dx[j] * R**|k-j| turns each sweep into a running minimum.
    """
    log_r = np.log(max_ratio)
    k = np.arange(dx.size, dtype=float)

    w = np.log(dx)
    w = np.minimum.accumulate(w - k * log_r) + k * log_r                # forward
    w = np.minimum.accumulate((w + k * log_r)[::-1])[::-1] - k * log_r  # backward
    return np.exp(w)


def make_nonuniform_nodes(
    x_min: float,
    x_max: float,
    Ncells: int,
    omega: float = 0.0,
    max_cell_ratio: float = 1.05,
    refine_by_mass: bool = False,
    d: float = 1.0,
) -> np.ndarray:
    """Build a node grid refined towards ``x_min``, in space or in enclosed mass.

    The zone layout follows lines 261-267 of ablation_solver_og.py: four nested
    zones ([0, L/1000], [L/1000, L/20], [L/20, L/3], [L/3, L]) carrying
    6N-1, 6N-1, N-1 and N cells, so the mesh is orders of magnitude finer at the ablation front than in the far field.

    Gluing those zones together directly leaves the cell width jumping by up to a
    factor of 49 at a seam, which corrupts the solution there. A shock captured by
    artificial viscosity is smeared over a fixed number of *cells*, so at such a
    seam it has to re-form its viscous structure over cells tens of times wider and
    briefly falls out of steady viscous balance. The wrong entropy jump gets
    deposited in the seam cells and then rides along as a frozen contact
    discontinuity: rho dips, e and T spike, while p stays clean because a pure
    entropy mode carries no pressure perturbation.

    So we keep the zone layout and the cell count, but grade the widths across the
    seams so neighbouring cells never differ by more than ``max_cell_ratio``.
    Grading only shrinks cells, so the widths are rescaled afterwards to span the
    full range exactly; at the default ratio that costs a uniform coarsening of
    ~24% while preserving the cell count and the ~4000x dynamic range.

    ``refine_by_mass`` chooses the coordinate the layout is laid down in. With
    ``False`` it is ``x``, as written above. With ``True`` it is
    ``s = x**(d-omega)`` for a problem in ``d`` dimensions (1 planar, 2
    cylindrical, 3 spherical): for the power-law medium rho(r) = rho0 r^-omega
    the enclosed mass is m(r) = beta rho0 r^(d-omega) / (d-omega), so s is
    proportional to mass and a grid uniform in s is uniform in mass. Finite
    enclosed mass requires omega < d, which is what the caller already asserts,
    so the exponent is positive whenever the problem is well posed.

    A radiative ablation front needs the mass grading. The ablated region is
    thin in *mass* (~1e-3 g/cm^2 for gold at 2 ns) and it is the mass it
    contains that fixes the ablation pressure driving the shock. Refining in x
    leaves that whole region inside the first cell or two whenever omega > 0,
    because rho ~ r^-omega piles the mass up at small r, and the piston pressure
    then comes out roughly an order of magnitude too high.

    A piston-driven shock does not: its structure is spread over the whole mass
    range, so x grading already resolves it, and mass grading only buys cells so
    narrow (dx ~ 1e-15 cm at omega = 0.5) that the CFL limit collapses.

    At omega = 0 in planar geometry the two coincide and both reduce to the
    plain x-space layout.
    """
    q = (float(d) - float(omega)) if refine_by_mass else 1.0
    if q <= 0.0:
        raise ValueError(
            f"omega must be < d to grade by mass (got omega={omega}, d={d}); "
            "the enclosed mass diverges otherwise"
        )

    # Grading coordinate: s ∝ m when refine_by_mass, else s == x.
    s_min, s_max = float(x_min) ** q, float(x_max) ** q
    S = s_max - s_min
    n = Ncells

    edges = np.array([0.0, S / 1000.0, S / 20.0, S / 3.0, S])
    counts = np.array([6 * n - 1, 6 * n - 1, n - 1, n])
    ds = np.repeat(np.diff(edges) / counts, counts)

    ds = _limit_width_ratio(ds, max_cell_ratio)
    ds *= S / ds.sum()

    s_nodes = np.empty(ds.size + 1)
    s_nodes[0] = s_min
    np.cumsum(ds, out=s_nodes[1:])
    s_nodes[1:] += s_min
    s_nodes[-1] = s_max
    return s_nodes ** (1.0 / q)

def make_nodes(
    x_min: float,
    x_max: float,
    Ncells: int,
    omega: float = 0.0,
    max_cell_ratio: float = 1.05,
    refine_by_mass: bool = False,
    d: float = 1.0,
) -> np.ndarray:
    """Generate 1D spatial node coordinates. Uses a gradually refined grid if omega != 0.

    Set ``refine_by_mass`` when the problem carries a radiative ablation front,
    so the refinement lands where the ablated mass is, and pass the problem's
    dimensionality ``d`` (1 planar, 2 cylindrical, 3 spherical) with it; see
    ``make_nonuniform_nodes``.
    """
    if omega != 0.0:
        return make_nonuniform_nodes(
            x_min,
            x_max,
            Ncells,
            omega=omega,
            max_cell_ratio=max_cell_ratio,
            refine_by_mass=refine_by_mass,
            d=d,
        )
    return make_uniform_nodes(x_min, x_max, Ncells)

def masses_from_initial_rho(x_nodes: np.ndarray, rho_cells0: np.ndarray, geom: Geometry) -> np.ndarray:
    V0 = cell_volumes(x_nodes, geom)
    return rho_cells0 * V0
