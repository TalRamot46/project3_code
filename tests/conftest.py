"""Pytest configuration for project3_code tests."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (run full simulations); use -m 'not slow' to skip",
    )
    config.addinivalue_line(
        "markers",
        "matlab: compares against MATLAB-generated .mat fixtures under matlab_shussman_solvers/test_exports/",
    )
