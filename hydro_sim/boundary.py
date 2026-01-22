def apply_velocity_bc(u, bc_left, bc_right):
    if bc_left == "reflecting":
        u[0] = 0.0
    elif bc_left == "outflow":
        pass  # do nothing

    if bc_right == "reflecting":
        u[-1] = 0.0
    elif bc_right == "outflow":
        pass  # do nothing
    return u
