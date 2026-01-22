import numpy as np

def total_mass(m_cells):
    return np.sum(m_cells)

def total_energy(state, grid, geom):
    # internal: sum m e
    Ein = np.sum(grid.m_cells * state.e)
    # kinetic: need node masses; simplest approx: average neighbor half-masses
    m_node = np.zeros_like(state.r)
    m_node[1:-1] = 0.5*(grid.m_cells[:-1] + grid.m_cells[1:])
    Ek = 0.5*np.sum(m_node * state.u**2)
    return Ein + Ek

def shock_position_from_q(state):
    j = np.argmax(state.q)
    # return cell center location estimate
    return 0.5*(state.r[j] + state.r[j+1])
