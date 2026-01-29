"""Test note event creation against reference data - matches notes_test.h."""

import json
import numpy as np
import pytest
from pathlib import Path

from engine.notes import NotesConverter, NoteEvent
from engine.constants import NUM_FREQ_OUT, NUM_FREQ_IN, MIDI_OFFSET
from engine.config import PitchBendMode

# Path to test data (src/tests/test_data)
ROOT = Path(__file__).resolve().parents[1]  # src/
TEST_DATA = ROOT / "tests" / "test_data"


def _load_pg_csv(name: str, num_cols: int) -> np.ndarray:
    """Load posteriorgram CSV and reshape to 2D."""
    data_1d = np.loadtxt(TEST_DATA / name, dtype=np.float32, delimiter=",")
    assert data_1d.size % num_cols == 0, \
        f"Data size {data_1d.size} not divisible by {num_cols}"
    num_frames = data_1d.size // num_cols
    return data_1d.reshape(num_frames, num_cols)


def _hz_to_midi(hz: float) -> int:
    """Convert frequency to MIDI note."""
    return int(round(12.0 * np.log2(hz / 440.0) + 69.0))


def test_notes_against_reference():
    """Test note conversion matches reference output exactly."""
    # Load posteriorgrams
    notes_pg = _load_pg_csv("notes.csv", NUM_FREQ_OUT)
    onsets_pg = _load_pg_csv("onsets.csv", NUM_FREQ_OUT)
    contours_pg = _load_pg_csv("contours.csv", NUM_FREQ_IN)
    
    # Load test cases and expected outputs
    with open(TEST_DATA / "note_events.input.json", "r") as f:
        all_cases = json.load(f)
    with open(TEST_DATA / "note_events.output.json", "r") as f:
        all_expected = json.load(f)
    
    assert len(all_cases) == len(all_expected), \
        f"Test case count mismatch: {len(all_cases)} cases vs {len(all_expected)} expected outputs"
    
    converter = NotesConverter()
    all_passed = True
    
    for idx, (case, expected_events) in enumerate(zip(all_cases, all_expected)):
        print(f"\nCase {idx}: frameThreshold={case.get('frameThreshold')}, "
              f"onsetThreshold={case.get('onsetThreshold')}, "
              f"minNoteLength={case.get('minNoteLength')}")
        
        # Map JSON params to converter arguments
        frame_threshold = case["frameThreshold"]
        onset_threshold = case["onsetThreshold"]
        min_note_length = case["minNoteLength"]
        infer_onsets = case.get("inferOnsets", True)
        max_freq = case.get("maxFrequency", -1.0)
        min_freq = case.get("minFrequency", -1.0)
        melodia_trick = case.get("melodiaTrick", True)
        energy_threshold = case.get("energyThreshold", 11)
        
        # Map pitch bend mode
        pitch_bend_json = case.get("pitchBend")
        if pitch_bend_json == "single":
            pitch_bend_mode = PitchBendMode.SINGLE_PITCH_BEND
        elif pitch_bend_json == "multi":
            pitch_bend_mode = PitchBendMode.MULTI_PITCH_BEND
        else:
            pitch_bend_mode = PitchBendMode.NO_PITCH_BEND
        
        # Convert note events
        events = converter.convert(
            notes_pg=notes_pg,
            onsets_pg=onsets_pg,
            contours_pg=contours_pg,
            frame_threshold=frame_threshold,
            onset_threshold=onset_threshold,
            min_note_length=min_note_length,
            pitch_bend_mode=pitch_bend_mode,
            infer_onsets=infer_onsets,
            melodia_trick=melodia_trick,
            energy_threshold=energy_threshold,
            max_frequency=max_freq,
            min_frequency=min_freq,
        )
        
        # Verify count matches
        assert len(events) == len(expected_events), \
            f"Case {idx}: got {len(events)} events, expected {len(expected_events)}"
        
        # Compare each event
        for j, (ev, exp) in enumerate(zip(events, expected_events)):
            # Compare discrete fields exactly
            assert ev.pitch == exp["pitch"], \
                f"Case {idx}, event {j}: pitch {ev.pitch} != {exp['pitch']}"
            assert ev.start_frame == exp["startFrame"], \
                f"Case {idx}, event {j}: startFrame {ev.start_frame} != {exp['startFrame']}"
            assert ev.end_frame == exp["endFrame"], \
                f"Case {idx}, event {j}: endFrame {ev.end_frame} != {exp['endFrame']}"
            
            # Compare float fields with tolerance
            assert abs(ev.start_time - exp["startTime"]) < 1e-5, \
                f"Case {idx}, event {j}: startTime {ev.start_time} != {exp['startTime']}"
            assert abs(ev.end_time - exp["endTime"]) < 1e-5, \
                f"Case {idx}, event {j}: endTime {ev.end_time} != {exp['endTime']}"
            assert abs(ev.amplitude - exp["amplitude"]) < 1e-4, \
                f"Case {idx}, event {j}: amplitude {ev.amplitude} != {exp['amplitude']}"
            
            # Compare bends (if present)
            expected_bends = exp.get("bends", [])
            assert ev.bends == expected_bends, \
                f"Case {idx}, event {j}: bends {ev.bends} != {expected_bends}"
        
        print(f"  ✓ Case {idx} passed: {len(events)} events")
    
    print("\n✅ All note conversion tests passed!")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
