# Shock solver (legacy)

This folder holds the **pre-rewrite** shock solver implementation for reference only.
The codebase uses `shussman_solvers.shock_solver` (parent directory), which has been
reimplemented to follow the original MATLAB units and scaling exactly (P0 in Barye,
time in seconds, consistent cgs output).

Do not import from this package in production code.
