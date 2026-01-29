# NeuralNote Backend Tests

Comprehensive test suite validating the Python implementation against the original C++ NeuralNote engine.

## Test Structure

### `test_features.py`
Tests feature extraction (ONNX model) against reference data from `tests/test_data/features_onnx.csv`.
- Validates that our ONNX feature extraction produces identical output to the C++ implementation
- Tolerance: 1e-3 (matching C++ test)

### `test_notes.py`
Tests note event creation from posteriorgrams against reference data.
- Uses `tests/test_data/notes.csv`, `onsets.csv`, `contours.csv`
- Tests multiple parameter combinations from `note_events.input.json`
- Validates against expected outputs in `note_events.output.json`
- Compares all fields: pitch, frames, times, amplitude, bends

### `test_integration.py`
Tests the full transcription pipeline end-to-end.
- Tests with default and custom configurations
- Validates note quantization functionality
- Validates time quantization functionality
- Ensures all note events have valid structure

### `test_api.py`
Tests FastAPI endpoints.
- Health check endpoint
- Transcription with OGG files
- Transcription with WebA files
- Custom configuration handling
- Error handling for invalid inputs

## Running Tests

### Run all tests:
```bash
cd src
source .env/Scripts/activate  # or .env\Scripts\activate.bat on Windows
pytest tests/ -v
```

### Run specific test suite:
```bash
# Feature extraction tests
pytest tests/test_features.py -v

# Note conversion tests (most comprehensive)
pytest tests/test_notes.py -v -s

# Integration tests
pytest tests/test_integration.py -v -s

# API tests
pytest tests/test_api.py -v
```

### Using the test runner script:
```bash
cd src
./run_tests.sh
```

## Test Data

Tests use reference data from `tests/test_data/`:
- `input_audio.csv` - Test audio input (22050 Hz, mono)
- `features_onnx.csv` - Expected ONNX feature output
- `notes.csv` - Note posteriorgrams
- `onsets.csv` - Onset posteriorgrams
- `contours.csv` - Contour posteriorgrams
- `note_events.input.json` - Test parameter configurations
- `note_events.output.json` - Expected note event outputs

## Expected Results

All tests should pass with **zero errors** when the implementation matches the C++ engine exactly. Any failures indicate discrepancies that need to be fixed.

## Notes

- Feature extraction test requires exact match (tolerance 1e-3)
- Note conversion tests compare all fields with appropriate tolerances:
  - Times: 1e-5 tolerance
  - Amplitude: 1e-4 tolerance
  - Discrete fields (pitch, frames): exact match
- Integration tests verify functionality but don't require exact numerical matches
