"""Smoke tests for presets and configurations.

Verifies that all entries in PRESETS, SIMULATION_CONFIGS, and problem-specific
test-case dictionaries can be instantiated without error.
"""

import pytest

from project_3.hydro_sim.problems.simulation_config import (
    SIMULATION_CONFIGS,
    SimulationConfig,
    get_config,
)
from project_3.hydro_sim.problems.presets import (
    PRESETS as HYDRO_PRESETS,
    get_preset as hydro_get_preset,
    get_preset_names as hydro_get_preset_names,
)
from project_3.hydro_sim.problems.Hydro_case import HydroCase


# ---------------------------------------------------------------------------
# SIMULATION_CONFIGS
# ---------------------------------------------------------------------------

def test_simulation_configs_not_empty():
    """There should be at least one predefined simulation config."""
    assert len(SIMULATION_CONFIGS) > 0


@pytest.mark.parametrize("name", list(SIMULATION_CONFIGS.keys()))
def test_simulation_config_valid(name):
    """Each config should be a SimulationConfig with sensible fields."""
    cfg = get_config(name)
    assert isinstance(cfg, SimulationConfig)
    assert cfg.N > 0
    assert cfg.CFL > 0
    assert cfg.store_every > 0


def test_get_config_unknown_raises():
    """Requesting an unknown config name should raise ValueError."""
    with pytest.raises(ValueError):
        get_config("nonexistent_config_xyz")


# ---------------------------------------------------------------------------
# Hydro PRESETS
# ---------------------------------------------------------------------------

def test_hydro_presets_not_empty():
    """There should be at least one hydro preset."""
    assert len(HYDRO_PRESETS) > 0


@pytest.mark.parametrize("name", list(HYDRO_PRESETS.keys()))
def test_hydro_preset_valid(name):
    """Each hydro preset should return a (HydroCase, SimulationConfig) pair."""
    case, config = hydro_get_preset(name)
    assert isinstance(config, SimulationConfig)
    # HydroCase is abstract; subclasses are used, so check duck-typing attrs
    assert hasattr(case, "gamma")
    assert hasattr(case, "x_min")
    assert hasattr(case, "x_max")
    assert hasattr(case, "t_end")
    assert case.x_max > case.x_min
    assert case.t_end > 0


def test_hydro_preset_names_consistent():
    """get_preset_names() should match PRESETS keys."""
    names = hydro_get_preset_names()
    assert set(names) == set(HYDRO_PRESETS.keys())


def test_hydro_get_preset_unknown_raises():
    """Requesting an unknown preset should raise ValueError."""
    with pytest.raises(ValueError):
        hydro_get_preset("nonexistent_preset_xyz")


# ---------------------------------------------------------------------------
# Rad-Hydro PRESETS (import only if available)
# ---------------------------------------------------------------------------

def test_rad_hydro_presets_importable():
    """rad_hydro_sim presets should import without error."""
    from project_3.rad_hydro_sim.problems.presets_config import (
        PRESETS as RAD_PRESETS,
        PRESET_TEST_CASES,
    )
    assert len(RAD_PRESETS) > 0
    assert len(PRESET_TEST_CASES) > 0


def test_rad_hydro_presets_valid():
    """Each rad-hydro preset should be a (case, config) pair with expected attrs."""
    from project_3.rad_hydro_sim.problems.presets_config import PRESETS as RAD_PRESETS

    for name, (case, config) in RAD_PRESETS.items():
        assert isinstance(config, SimulationConfig), f"Bad config in rad preset {name}"
        assert hasattr(case, "x_min")
        assert hasattr(case, "x_max")
        assert hasattr(case, "t_end")
        assert case.x_max > case.x_min
        assert case.t_end > 0
