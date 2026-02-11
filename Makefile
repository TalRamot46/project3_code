# project_3/Makefile
# ============================================================================
# Targets for running simulations, tests, and code quality tools.
# Run from inside project_3/: make <target>
# REPO_ROOT = parent of project_3 so that "python -m project_3...." works.
# ============================================================================

PYTHON ?= python
REPO_ROOT := $(abspath ..)

# ---- Run simulations -------------------------------------------------------

.PHONY: run-hydro run-rad-hydro run-verification

run-hydro:
	cd $(REPO_ROOT) && $(PYTHON) -m project_3.hydro_sim.run_hydro

run-rad-hydro:
	cd $(REPO_ROOT) && $(PYTHON) -m project_3.rad_hydro_sim.run_rad_hydro

run-verification:
	cd $(REPO_ROOT) && $(PYTHON) -m project_3.rad_hydro_sim.verification.run_comparison

# ---- Testing ---------------------------------------------------------------

.PHONY: test

test:
	cd $(REPO_ROOT) && $(PYTHON) -m pytest project_3/tests -v

# ---- Code quality ----------------------------------------------------------

.PHONY: fmt lint

fmt:
	cd $(REPO_ROOT) && $(PYTHON) -m ruff format project_3/
	cd $(REPO_ROOT) && $(PYTHON) -m ruff check --fix project_3/

lint:
	cd $(REPO_ROOT) && $(PYTHON) -m ruff check project_3/

# ---- Documentation (LaTeX submodule) ----------------------------------------
# "Numerical problems" is a separate Git repo added as a submodule under docs/.
# One-time add (from project_3): git submodule add "../Numerical problems" "docs/Numerical problems"
# Then: make submodule-init to fetch/checkout the submodule after clone.

.PHONY: submodule-init submodule-add

submodule-init:
	git submodule update --init --recursive

# Add the LaTeX repo as submodule (run once). Set REPO to local path or URL, e.g.:
#   make submodule-add REPO="../Numerical problems"
#   make submodule-add REPO="https://github.com/you/Numerical-problems.git"
submodule-add:
	git submodule add "$(REPO)" "docs/Numerical problems"
