"""Pytest configuration for project_3 tests."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "slow: marks tests as slow (run full simulations); use -m 'not slow' to skip",
    )
