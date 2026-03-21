# project3_code/Makefile
# ============================================================================
# Targets for running simulations, tests, and code quality tools.
# Run from inside project3_code/: make <target>
# REPO_ROOT = parent of project3_code so that "python -m project3_code...." works.
# ============================================================================

PYTHON ?= python
REPO_ROOT := $(abspath ..)

# ---- Run simulations -------------------------------------------------------

.PHONY: run-hydro run-rad-hydro run-verification

run-hydro:
	cd $(REPO_ROOT) && $(PYTHON) -m project3_code.hydro_sim.run_hydro

run-rad-hydro:
	cd $(REPO_ROOT) && $(PYTHON) -m project3_code.rad_hydro_sim.run_rad_hydro

run-verification:
	cd $(REPO_ROOT) && $(PYTHON) -m project3_code.rad_hydro_sim.verification.run_comparison

# ---- Testing ---------------------------------------------------------------

.PHONY: test

test:
	cd $(REPO_ROOT) && $(PYTHON) -m pytest project3_code/tests -v

# ---- Code quality ----------------------------------------------------------

.PHONY: fmt lint

fmt:
	cd $(REPO_ROOT) && $(PYTHON) -m ruff format project3_code/
	cd $(REPO_ROOT) && $(PYTHON) -m ruff check --fix project3_code/

lint:
	cd $(REPO_ROOT) && $(PYTHON) -m ruff check project3_code/
