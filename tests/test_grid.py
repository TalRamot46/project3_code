"""Smoke tests for the grid module."""

import numpy as np
import pytest

from project_3.hydro_sim.core.geometry import planar, spherical, cylindrical
from project_3.hydro_sim.core.grid import (
    Grid1D,
    cell_volumes,
    make_uniform_nodes,
    masses_from_initial_rho,
)


# ---------------------------------------------------------------------------
# make_uniform_nodes
# ---------------------------------------------------------------------------

def test_uniform_nodes_count():
    """Ncells cells should produce Ncells+1 nodes."""
    for n in (10, 100, 500):
        nodes = make_uniform_nodes(0.0, 1.0, n)
        assert len(nodes) == n + 1


def test_uniform_nodes_endpoints():
    """First and last node should match x_min and x_max."""
    nodes = make_uniform_nodes(0.5, 2.5, 200)
    np.testing.assert_allclose(nodes[0], 0.5)
    np.testing.assert_allclose(nodes[-1], 2.5)


def test_uniform_nodes_monotonic():
    """Node positions should be strictly increasing."""
    nodes = make_uniform_nodes(0.0, 1.0, 50)
    assert np.all(np.diff(nodes) > 0)


# ---------------------------------------------------------------------------
# cell_volumes
# ---------------------------------------------------------------------------

def test_cell_volumes_planar():
    """In planar geometry V_i = x_{i+1} - x_i for uniform spacing."""
    nodes = make_uniform_nodes(0.0, 1.0, 100)
    vol = cell_volumes(nodes, planar())
    expected_dx = 1.0 / 100
    np.testing.assert_allclose(vol, expected_dx, rtol=1e-12)


def test_cell_volumes_positive():
    """Cell volumes must be positive for all geometries."""
    # Use x_min > 0 to avoid r=0 issues in spherical/cylindrical
    nodes = make_uniform_nodes(0.1, 1.0, 50)
    for geom in (planar(), cylindrical(), spherical()):
        vol = cell_volumes(nodes, geom)
        assert np.all(vol > 0), f"Non-positive volume in {geom}"


def test_cell_volumes_count():
    """Number of cell volumes should equal number of cells (Nnodes - 1)."""
    nodes = make_uniform_nodes(0.0, 1.0, 40)
    vol = cell_volumes(nodes, planar())
    assert len(vol) == 40


# ---------------------------------------------------------------------------
# masses_from_initial_rho
# ---------------------------------------------------------------------------

def test_masses_positive():
    """Cell masses should be positive for positive density."""
    nodes = make_uniform_nodes(0.1, 1.0, 30)
    rho = np.ones(30) * 2.0
    m = masses_from_initial_rho(nodes, rho, planar())
    assert np.all(m > 0)


def test_total_mass_planar():
    """Total mass = rho * L for uniform density in planar geometry."""
    L = 2.0
    rho0 = 3.0
    N = 100
    nodes = make_uniform_nodes(0.0, L, N)
    rho = np.full(N, rho0)
    m = masses_from_initial_rho(nodes, rho, planar())
    np.testing.assert_allclose(m.sum(), rho0 * L, rtol=1e-12)


# ---------------------------------------------------------------------------
# Grid1D dataclass
# ---------------------------------------------------------------------------

def test_grid1d_construction():
    """Grid1D should store nodes, masses, and geometry."""
    nodes = make_uniform_nodes(0.0, 1.0, 10)
    rho = np.ones(10)
    m = masses_from_initial_rho(nodes, rho, planar())
    grid = Grid1D(x_nodes=nodes, m_cells=m, geom=planar())
    assert grid.x_nodes is nodes
    assert grid.m_cells is m
    assert grid.geom == planar()
