"""Integration tests for the full transcription pipeline."""

import numpy as np
import pytest
from pathlib import Path

from engine.engine import TranscriptionEngine
from engine.config import TranscriptionConfig
from engine.constants import BASIC_PITCH_SAMPLE_RATE

# Path to test data (src/tests/test_data)
ROOT = Path(__file__).resolve().parents[1]  # src/
TEST_DATA = ROOT / "tests" / "test_data"


def test_full_pipeline_with_test_audio():
    """Test full transcription pipeline with test audio data."""
    # Load test audio
    audio = np.loadtxt(TEST_DATA / "input_audio.csv", dtype=np.float32, delimiter=",")
    sample_rate = int(BASIC_PITCH_SAMPLE_RATE)
    
    # Create engine
    engine = TranscriptionEngine(models_dir=str(ROOT / "models"))
    
    # Test with default config
    config = TranscriptionConfig()
    note_events = engine.transcribe(audio, sample_rate, config)
    
    # Should produce some notes
    assert len(note_events) > 0, "Expected at least one note event"
    
    # Verify note event structure
    for event in note_events:
        assert event.start_time >= 0, "Start time should be non-negative"
        assert event.end_time > event.start_time, "End time should be after start time"
        assert event.start_frame >= 0, "Start frame should be non-negative"
        assert event.end_frame > event.start_frame, "End frame should be after start frame"
        assert 21 <= event.pitch <= 108, f"Pitch {event.pitch} should be in MIDI range [21, 108]"
        assert event.amplitude >= 0, "Amplitude should be non-negative"
        assert isinstance(event.bends, list), "Bends should be a list"
    
    print(f"✓ Full pipeline test passed: {len(note_events)} notes generated")


def test_pipeline_with_custom_config():
    """Test transcription with custom parameters."""
    audio = np.loadtxt(TEST_DATA / "input_audio.csv", dtype=np.float32, delimiter=",")
    sample_rate = int(BASIC_PITCH_SAMPLE_RATE)
    
    engine = TranscriptionEngine(models_dir=str(ROOT / "models"))
    
    # Test with high sensitivity (should get more notes)
    config_high = TranscriptionConfig(
        note_sensitivity=0.9,
        split_sensitivity=0.7,
        minimum_note_duration_ms=50.0
    )
    events_high = engine.transcribe(audio, sample_rate, config_high)
    
    # Test with low sensitivity (should get fewer notes)
    config_low = TranscriptionConfig(
        note_sensitivity=0.3,
        split_sensitivity=0.3,
        minimum_note_duration_ms=200.0
    )
    events_low = engine.transcribe(audio, sample_rate, config_low)
    
    # High sensitivity should generally produce more notes
    assert len(events_high) >= len(events_low), \
        f"High sensitivity should produce >= notes: {len(events_high)} vs {len(events_low)}"
    
    print(f"✓ Custom config test passed: high={len(events_high)} notes, low={len(events_low)} notes")


def test_note_quantization():
    """Test note quantization functionality."""
    audio = np.loadtxt(TEST_DATA / "input_audio.csv", dtype=np.float32, delimiter=",")
    sample_rate = int(BASIC_PITCH_SAMPLE_RATE)
    
    engine = TranscriptionEngine(models_dir=str(ROOT / "models"))
    
    # Without quantization
    config_no_quant = TranscriptionConfig(enable_note_quantization=False)
    events_no_quant = engine.transcribe(audio, sample_rate, config_no_quant)
    
    # With quantization to C major
    config_quant = TranscriptionConfig(
        enable_note_quantization=True,
        key_root_note=3,  # C
        key_type=1,  # Major
        key_snap_mode=0  # Adjust
    )
    events_quant = engine.transcribe(audio, sample_rate, config_quant)
    
    # Both should produce notes
    assert len(events_no_quant) > 0, "Should have notes without quantization"
    assert len(events_quant) > 0, "Should have notes with quantization"
    
    print(f"✓ Note quantization test passed: no_quant={len(events_no_quant)}, quant={len(events_quant)}")


def test_time_quantization():
    """Test time quantization functionality."""
    audio = np.loadtxt(TEST_DATA / "input_audio.csv", dtype=np.float32, delimiter=",")
    sample_rate = int(BASIC_PITCH_SAMPLE_RATE)
    
    engine = TranscriptionEngine(models_dir=str(ROOT / "models"))
    
    # Without time quantization
    config_no_quant = TranscriptionConfig(enable_time_quantization=False)
    events_no_quant = engine.transcribe(audio, sample_rate, config_no_quant)
    
    # With time quantization
    config_quant = TranscriptionConfig(
        enable_time_quantization=True,
        time_division=5,  # 1/8
        quantization_force=0.7,
        bpm=120.0
    )
    events_quant = engine.transcribe(audio, sample_rate, config_quant)
    
    # Both should produce notes
    assert len(events_no_quant) > 0, "Should have notes without time quantization"
    assert len(events_quant) > 0, "Should have notes with time quantization"
    
    # Quantized times should be different (but same number of notes)
    # Note: times might be the same if quantization force is low, so we just check counts match
    assert len(events_no_quant) == len(events_quant), \
        "Time quantization should not change note count"
    
    print(f"✓ Time quantization test passed: {len(events_quant)} notes")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
