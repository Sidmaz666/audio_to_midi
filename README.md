# Audio to MIDI FastAPI Backend

Complete Python FastAPI implementation of Audio to MIDI's transcription engine using Basic Pitch and DL.

## Features

- **Full Audio to MIDI compatibility**: Uses the exact same models and algorithms as the original C++ plugin
- **Comprehensive parameter control**: All transcription parameters from the UI are available via API
- **MIDI export**: Returns note events as JSON and optionally as base64-encoded MIDI files
- **CORS enabled**: Ready for web frontend integration
- **Hugging Face Spaces ready**: Includes Dockerfile for easy deployment

## API Endpoints

### `GET /health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

### `POST /transcribe`

Transcribe audio file to MIDI note events.

**Request:**
- `file`: Audio file (multipart/form-data)
  - Supported formats: `.wav`, `.aiff`, `.aif`, `.flac`, `.mp3`, `.ogg`
- `config`: Optional JSON string with transcription parameters (multipart/form-data)

**Response:**
```json
{
  "notes": [
    {
      "pitch": 60,
      "start": 0.5,
      "end": 1.0,
      "start_frame": 44,
      "end_frame": 88,
      "amplitude": 0.8,
      "bends": [0, 1, -1, 0]
    }
  ],
  "num_notes": 1,
  "midi_base64": "..." // Optional, if MIDI generation succeeds
}
```

## Configuration Parameters

All parameters match Audio to MIDI's UI defaults exactly:

### Transcription Parameters

- `note_sensitivity` (float, 0.05-0.95, default: 0.7): Note sensitivity threshold. Higher gives more notes.
- `split_sensitivity` (float, 0.05-0.95, default: 0.5): Split sensitivity threshold. Higher will split notes more.
- `minimum_note_duration_ms` (float, 35.0-580.0, default: 125.0): Minimum note duration in milliseconds.
- `pitch_bend_mode` (string, "no" or "single", default: "no"): Pitch bend mode.

### Note Options

- `enable_note_quantization` (bool, default: false): Enable note quantization to scale.
- `min_midi_note` (int, 21-108, default: 21): Minimum MIDI note number.
- `max_midi_note` (int, 21-108, default: 108): Maximum MIDI note number.
- `key_root_note` (int, 0-11, default: 3): Root note index (0=A, 1=A#, 2=B, 3=C, etc.).
- `key_type` (int, 0-13, default: 0): Scale type index (0=Chromatic, 1=Major, 2=Minor, etc.).
- `key_snap_mode` (int, 0-1, default: 0): Snap mode (0=Adjust, 1=Remove).

### Time Quantization

- `enable_time_quantization` (bool, default: false): Enable time quantization.
- `time_division` (int, 0-11, default: 5): Time division index (0=1/1, 1=1/2, ..., 5=1/8, etc.).
- `quantization_force` (float, 0.0-1.0, default: 0.0): Quantization force.

### Transport/Time Info

- `bpm` (float, default: 120.0): Tempo in BPM (for time quantization).
- `time_signature_num` (int, default: 4): Time signature numerator.
- `time_signature_denom` (int, default: 4): Time signature denominator.

## Example Usage

### Using curl

```bash
# Basic transcription with defaults
curl -X POST "http://localhost:7860/transcribe" \
  -F "file=@audio.wav"

# With custom parameters
curl -X POST "http://localhost:7860/transcribe" \
  -F "file=@audio.wav" \
  -F 'config={"note_sensitivity": 0.8, "enable_note_quantization": true, "key_type": 1}'
```

### Using Python

```python
import requests

# Basic transcription
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:7860/transcribe",
        files={"file": f}
    )
    result = response.json()
    print(f"Found {result['num_notes']} notes")

# With custom config
config = {
    "note_sensitivity": 0.8,
    "split_sensitivity": 0.6,
    "enable_note_quantization": True,
    "key_root_note": 3,  # C
    "key_type": 1,  # Major
    "enable_time_quantization": True,
    "time_division": 5,  # 1/8
    "quantization_force": 0.5,
    "bpm": 120.0
}

with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:7860/transcribe",
        files={"file": f},
        data={"config": json.dumps(config)}
    )
    result = response.json()
    
    # Access notes
    for note in result["notes"]:
        print(f"Note {note['pitch']} from {note['start']:.2f}s to {note['end']:.2f}s")
    
    # Decode MIDI if available
    if "midi_base64" in result:
        import base64
        midi_bytes = base64.b64decode(result["midi_base64"])
        with open("output.mid", "wb") as midi_file:
            midi_file.write(midi_bytes)
```

### Using JavaScript/Fetch

```javascript
const formData = new FormData();
formData.append('file', audioFile);

const config = {
  note_sensitivity: 0.8,
  enable_note_quantization: true,
  key_type: 1
};
formData.append('config', JSON.stringify(config));

const response = await fetch('http://localhost:7860/transcribe', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(`Found ${result.num_notes} notes`);
```

## Local Development

1. Install dependencies:
```bash
cd src
pip install -r ../requirements.txt
# Or from root: pip install -r requirements.txt
```

2. Ensure model files are in `src/models/`:
   - `features_model.onnx`
   - `cnn_contour_model.json`
   - `cnn_note_model.json`
   - `cnn_onset_1_model.json`
   - `cnn_onset_2_model.json`

3. Run the server:
```bash
# From repo root (recommended):
python start.py
# If using a venv in src/.env:  src/.env/Scripts/python start.py  (Windows)
#                              source src/.env/bin/activate && python start.py  (Unix)

# Or from src/:
cd src && python app.py

# Or with uvicorn directly (from src/):
cd src && uvicorn app:app --host 0.0.0.0 --port 7860
```

4. Run tests:
```bash
# From src/ (with venv activated):
cd src && pytest tests/ -v

# From root (with venv in src/.env):
src/.env/Scripts/python -m pytest src/tests/ -v   # Windows
```

## Deployment to Hugging Face Spaces

1. Ensure the `Dockerfile` is in the repository root.

2. The Dockerfile will:
   - Install system dependencies (libsndfile1)
   - Copy `requirements.txt` and `src/`
   - Install Python dependencies
   - Start the FastAPI server on port 7860 (or $PORT if set)

3. Hugging Face Spaces will automatically:
   - Build the Docker image
   - Expose the API on their domain
   - Handle CORS and routing

## Architecture

```
Audio to MIDI/
├── start.py          # Run API from root: python start.py
├── requirements.txt
├── Dockerfile
├── README.md
└── src/
    ├── app.py        # FastAPI application
    ├── engine/       # Transcription engine
    │   ├── constants.py, config.py, features.py, cnn.py
    │   ├── notes.py, note_options.py, time_quantize.py
    │   ├── engine.py, midi.py
    │   └── ...
    ├── models/       # ONNX + JSON model files
    ├── sample/       # Sample audio files
    └── tests/        # Tests + test_data
```

## Notes

- The implementation faithfully replicates the C++ Audio to MIDI engine
- All parameter defaults match the original plugin exactly
- The CNN implementation uses manual convolution to match RTNeural behavior
- Pitch bends are included in note events when enabled
- MIDI export requires the `mido` library (included in requirements.txt)

## License

Same as Audio to MIDI (Apache-2.0).
