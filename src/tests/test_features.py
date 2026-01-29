"""Test feature extraction against reference data - matches features_test.h."""

import numpy as np
import pytest
from pathlib import Path

from engine.features import FeatureExtractor
from engine.constants import NUM_HARMONICS, NUM_FREQ_IN

# Path to test data (src/tests/test_data)
ROOT = Path(__file__).resolve().parents[1]  # src/
TEST_DATA = ROOT / "tests" / "test_data"


def test_features_against_reference():
    """Test feature extraction matches reference output exactly."""
    # Load input audio (float CSV)
    audio = np.loadtxt(TEST_DATA / "input_audio.csv", dtype=np.float32, delimiter=",")
    
    # Load reference features
    ref_features = np.loadtxt(TEST_DATA / "features_onnx.csv", dtype=np.float32, delimiter=",")
    
    # Calculate expected number of frames
    num_frames_ref = ref_features.size // (NUM_HARMONICS * NUM_FREQ_IN)
    
    # Compute features
    fe = FeatureExtractor(model_path=str(ROOT / "models" / "features_model.onnx"))
    features, num_frames = fe.compute_features(audio, sample_rate=22050)
    
    # Verify frame count matches
    assert num_frames == num_frames_ref, \
        f"Frame count mismatch: got {num_frames}, expected {num_frames_ref}"
    
    # Flatten both for comparison
    flat_new = features.reshape(-1)
    flat_ref = ref_features.reshape(-1)
    
    assert flat_new.shape == flat_ref.shape, \
        f"Shape mismatch: got {flat_new.shape}, expected {flat_ref.shape}"
    
    # Compare with tolerance (matching C++ test: threshold = 1e-3)
    diff = np.abs(flat_new - flat_ref)
    max_err = float(np.max(diff))
    num_err = int(np.count_nonzero(diff > 1e-3))
    
    print(f"Features test: num errors = {num_err} over {len(flat_ref)} values")
    print(f"Max error is {max_err}")
    
    # Should have zero errors (exact match within tolerance)
    assert num_err == 0, \
        f"Too many feature mismatches: {num_err} errors, max_err={max_err}"
    
    # Also verify max error is reasonable
    assert max_err < 1e-3, \
        f"Max error {max_err} exceeds threshold 1e-3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
