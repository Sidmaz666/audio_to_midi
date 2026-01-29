"""Pytest configuration and fixtures."""

import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir():
    """Fixture providing path to test data directory (src/tests/test_data)."""
    root = Path(__file__).resolve().parents[1]  # src/
    test_data = root / "tests" / "test_data"
    assert test_data.exists(), f"Test data directory not found: {test_data}"
    return test_data


@pytest.fixture(scope="session")
def models_dir():
    """Fixture providing path to models directory."""
    root = Path(__file__).resolve().parents[1]  # src/
    models = root / "models"
    assert models.exists(), f"Models directory not found: {models}"
    return models
