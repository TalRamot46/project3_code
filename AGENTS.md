## Cursor Cloud specific instructions

### Project overview

Scientific computing Python project for Lagrangian hydrodynamics and radiation-hydrodynamics simulations. No external services (databases, Docker, APIs) are required. See `PROJECT_STRUCTURE.md` for full directory layout and `Makefile` for available commands.

### Package import path

The workspace `/workspace` IS the `project_3/` package directory. A symlink `/project_3 -> /workspace` must exist so that `python -m project_3.*` resolves correctly when run from `/`. The update script creates this symlink automatically.

All commands (simulations, tests, lint) must be run from `/` (the parent of the symlink) so that `project_3` is on `sys.path`. Alternatively, use `make` targets from within `/workspace` — the Makefile handles `cd $(REPO_ROOT)` automatically.

### Running simulations

Use `MPLBACKEND=Agg` since there is no X display. GIF generation can be very slow (minutes) with high cell counts; simulations themselves complete much faster.

```
cd / && MPLBACKEND=Agg python3 -m project_3.hydro_sim.run_hydro
cd / && MPLBACKEND=Agg python3 -m project_3.rad_hydro_sim.run_rad_hydro
```

Or from `/workspace`:
```
MPLBACKEND=Agg make run-hydro
MPLBACKEND=Agg make run-rad-hydro
```

### Tests

```
cd / && python3 -m pytest project_3/tests -v
```
Or: `make test` from `/workspace`.

### Lint

```
cd / && python3 -m ruff check project_3/
```
Or: `make lint` from `/workspace`. The codebase has pre-existing lint warnings (import sorting, unused imports, whitespace). `make fmt` auto-fixes.
