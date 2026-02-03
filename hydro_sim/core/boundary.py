import numpy as np
def apply_velocity_bc(u, bc_left, bc_right, t):
    u = u.copy()

    # ---- Left boundary ----
    if isinstance(bc_left, dict):
        if bc_left["type"] == "pressure":
            u[0] = bc_left.get("u", lambda t: 0.0)

    elif bc_left == "outflow":
        u[0] = u[1]

    elif bc_left == "none":
        pass

    else:
        raise ValueError(f"Unknown bc_left: {bc_left}")

    # ---- Right boundary ----
    if isinstance(bc_right, dict):
        raise NotImplementedError("Driven BC not supported on right boundary")

    elif bc_right == "outflow":
        u[-1] = u[-2]

    elif bc_right == "none":
        pass

    else:
        raise ValueError(f"Unknown bc_right: {bc_right}")

    return u

def apply_pressure_bc(p, bc_left, bc_right, t):
    p = p.copy()

    if isinstance(bc_left, dict):
        if bc_left["type"] == "pressure":
            p[0] = bc_left["p"]

    elif bc_left == "outflow":
        p[0] = p[1]

    elif bc_left == "none":
        pass

    else:
        raise ValueError(f"Unknown bc_left: {bc_left}")

    if bc_right == "outflow":
        p[-1] = p[-2]

    elif bc_right == "none":
        pass

    else:
        raise ValueError(f"Unknown bc_right: {bc_right}")

    return p
