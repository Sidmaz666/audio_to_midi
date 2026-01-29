"""Tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path

from app import app

# Path to sample files
ROOT = Path(__file__).resolve().parents[1]  # src/
SAMPLE_DIR = ROOT / "sample"


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_transcribe_ogg_file(client):
    """Test transcription with OGG file."""
    ogg_file = SAMPLE_DIR / "Chuck_Berry_intro.ogg"
    if not ogg_file.exists():
        pytest.skip(f"Sample file not found: {ogg_file}")
    
    with open(ogg_file, "rb") as f:
        response = client.post(
            "/transcribe",
            files={"file": ("Chuck_Berry_intro.ogg", f, "audio/ogg")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "notes" in data
    assert "num_notes" in data
    assert isinstance(data["notes"], list)
    assert data["num_notes"] == len(data["notes"])
    
    # Verify note structure
    if len(data["notes"]) > 0:
        note = data["notes"][0]
        assert "pitch" in note
        assert "start" in note
        assert "end" in note
        assert "amplitude" in note
        assert "bends" in note


def test_transcribe_weba_file(client):
    """Test transcription with WebA file."""
    weba_file = SAMPLE_DIR / "scale.weba"
    if not weba_file.exists():
        pytest.skip(f"Sample file not found: {weba_file}")
    
    with open(weba_file, "rb") as f:
        response = client.post(
            "/transcribe",
            files={"file": ("scale.weba", f, "audio/webm")}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "notes" in data
    assert "num_notes" in data


def test_transcribe_with_config(client):
    """Test transcription with custom configuration."""
    import json
    
    ogg_file = SAMPLE_DIR / "Chuck_Berry_intro.ogg"
    if not ogg_file.exists():
        pytest.skip(f"Sample file not found: {ogg_file}")
    
    config = {
        "note_sensitivity": 0.8,
        "split_sensitivity": 0.6,
        "enable_note_quantization": True,
        "key_type": 1
    }
    
    with open(ogg_file, "rb") as f:
        response = client.post(
            "/transcribe",
            files={"file": ("Chuck_Berry_intro.ogg", f, "audio/ogg")},
            data={"config": json.dumps(config)}
        )
    
    assert response.status_code == 200
    data = response.json()
    assert "notes" in data


def test_transcribe_unsupported_format(client):
    """Test that unsupported formats are rejected."""
    # Create a dummy file with unsupported extension
    response = client.post(
        "/transcribe",
        files={"file": ("test.txt", b"dummy content", "text/plain")}
    )
    
    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_transcribe_invalid_config(client):
    """Test that invalid config JSON is rejected."""
    ogg_file = SAMPLE_DIR / "Chuck_Berry_intro.ogg"
    if not ogg_file.exists():
        pytest.skip(f"Sample file not found: {ogg_file}")
    
    with open(ogg_file, "rb") as f:
        response = client.post(
            "/transcribe",
            files={"file": ("Chuck_Berry_intro.ogg", f, "audio/ogg")},
            data={"config": "invalid json {{{{}}"}
        )
    
    assert response.status_code == 400
    assert "Invalid config JSON" in response.json()["detail"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
