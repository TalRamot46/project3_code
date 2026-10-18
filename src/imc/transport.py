"""Numba random-walk kernel for one IMC time step (gray, 1D slab).

Solves, in expectation, the Fleck-Cummings equation with physical scattering

    (1/c) dI/dt + mu dI/dx + (k_a + k_s) I
        = (1/2) [(1 - f) k_a + k_s] int I dmu' + (1/2) f k_a c u_r^n

for one step (md/imc_mc_derivation.md, Theorem 2).  Each packet flies
straight; the distance to its next collision is exponential with the total
opacity k_t = k_a + k_s.  At a collision it is absorbed with probability
f k_a / k_t (its weight is deposited in the cell) and otherwise scatters
isotropically.  It stops at the census time t1, or leaves through a vacuum
boundary; a reflecting boundary flips mu.

Particles carry their cell index, so a face crossing moves the packet exactly
onto the face and changes the index, with no position nudge.

Tallies, all energies per unit area:
  deposited[j]      weight absorbed in cell j
  face_plus[k]      weight crossing face k towards +x (k = 0..N)
  face_minus[k]     weight crossing face k towards -x
  track[j]          integral of weight * time spent in cell j (time-integrated
                    radiation energy; divided by dt and dx it is the
                    step-averaged energy density)
  escaped_left/right, n_collisions, n_capped (histories stopped by max_events)
"""
from __future__ import annotations

import numpy as np
from numba import njit

VACUUM = 0
REFLECTING = 1


@njit(cache=True)
def seed(value):
    """Numba keeps its own random state; it must be seeded in compiled code."""
    np.random.seed(value)


@njit(cache=True)
def transport_step(
    x, mu, w, t, cell,               # particles (cell = containing cell index)
    edges, k_a, k_s, fleck,          # mesh and per-cell coefficients
    c, t1,                           # speed of light, census time
    left_bc, right_bc,               # VACUUM or REFLECTING
    max_events,                      # safety cap on events per history
):
    n = x.size
    n_cells = edges.size - 1

    cx = np.empty(n)
    cmu = np.empty(n)
    cw = np.empty(n)
    ccell = np.empty(n, dtype=np.int64)
    n_census = 0

    deposited = np.zeros(n_cells)
    face_plus = np.zeros(n_cells + 1)
    face_minus = np.zeros(n_cells + 1)
    track = np.zeros(n_cells)
    escaped_left = 0.0
    escaped_right = 0.0
    n_collisions = 0
    n_capped = 0

    for p in range(n):
        xp = x[p]
        mup = mu[p]
        wp = w[p]
        tp = t[p]
        j = cell[p]
        if wp <= 0.0:
            continue

        events = 0
        while True:
            events += 1
            if events > max_events:
                # Runaway history: park it in the census where it is.
                cx[n_census] = xp
                cmu[n_census] = mup
                cw[n_census] = wp
                ccell[n_census] = j
                n_census += 1
                n_capped += 1
                break

            k_t = k_a[j] + k_s[j]
            if k_t > 0.0:
                d_collision = -np.log(1.0 - np.random.random()) / k_t
            else:
                d_collision = np.inf

            if mup > 0.0:
                d_face = (edges[j + 1] - xp) / mup
            elif mup < 0.0:
                d_face = (edges[j] - xp) / mup
            else:
                d_face = np.inf
            if d_face < 0.0:
                d_face = 0.0

            d_census = c * (t1 - tp)
            if d_census < 0.0:
                d_census = 0.0

            # Census wins ties, then the face.
            if d_census <= d_face and d_census <= d_collision:
                track[j] += wp * d_census / c
                xp += mup * d_census
                # Keep the census position inside its cell despite round-off.
                if xp < edges[j]:
                    xp = edges[j]
                elif xp > edges[j + 1]:
                    xp = edges[j + 1]
                cx[n_census] = xp
                cmu[n_census] = mup
                cw[n_census] = wp
                ccell[n_census] = j
                n_census += 1
                break

            if d_face <= d_collision:
                track[j] += wp * d_face / c
                tp += d_face / c
                if mup > 0.0:
                    face_plus[j + 1] += wp
                    if j == n_cells - 1:
                        if right_bc == REFLECTING:
                            xp = edges[n_cells]
                            mup = -mup
                            face_minus[n_cells] += wp   # re-enters through the same face
                            continue
                        escaped_right += wp
                        break
                    j += 1
                    xp = edges[j]
                else:
                    face_minus[j] += wp
                    if j == 0:
                        if left_bc == REFLECTING:
                            xp = edges[0]
                            mup = -mup
                            face_plus[0] += wp
                            continue
                        escaped_left += wp
                        break
                    j -= 1
                    xp = edges[j + 1]
                continue

            # Collision.
            track[j] += wp * d_collision / c
            tp += d_collision / c
            xp += mup * d_collision
            n_collisions += 1
            if np.random.random() * k_t < fleck[j] * k_a[j]:
                deposited[j] += wp               # effective absorption: history ends
                break
            mup = 2.0 * np.random.random() - 1.0  # effective or physical scattering

    return (
        cx[:n_census], cmu[:n_census], cw[:n_census], ccell[:n_census],
        deposited, face_plus, face_minus, track,
        escaped_left, escaped_right, n_collisions, n_capped,
    )
