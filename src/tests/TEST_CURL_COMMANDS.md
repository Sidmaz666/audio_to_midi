# Curl Commands for Testing NeuralNote API

## Prerequisites

Make sure the API server is running. From repo root: `python start.py`. Or from `src/`:
```bash
cd src
source .env/Scripts/activate   # if using a venv in src/
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Basic Commands

### Health Check
```bash
curl http://localhost:7860/health
```

### Transcribe OGG File (Chuck Berry)
```bash
# Basic transcription with defaults
curl -X POST "http://localhost:7860/transcribe" \
  -F "file=@sample/Chuck_Berry_intro.ogg" \
  -o response_berry.json

# View results
cat response_berry.json | python -m json.tool
```

### Transcribe WebA File (Scale)
```bash
# Basic transcription
curl -X POST "http://localhost:7860/transcribe" \
  -F "file=@sample/scale.weba" \
  -o response_scale.json

# View results
cat response_scale.json | python -m json.tool
```

## Advanced Commands with Custom Configuration

### High Sensitivity (More Notes)
```bash
curl -X POST "http://localhost:7860/transcribe" \
  -F "file=@sample/Chuck_Berry_intro.ogg" \
  -F 'config={"note_sensitivity": 0.9, "split_sensitivity": 0.8}' \
  -o response_high_sensitivity.json
```

### With Note Quantization (C Major)
```bash
curl -X POST "http://localhost:7860/transcribe" \
  -F "file=@sample/Chuck_Berry_intro.ogg" \
  -F 'config={
    "note_sensitivity": 0.7,
    "enable_note_quantization": true,
    "key_root_note": 3,
    "key_type": 1,
    "key_snap_mode": 0
  }' \
  -o response_quantized.json
```

### With Time Quantization
```bash
curl -X POST "http://localhost:7860/transcribe" \
  -F "file=@sample/scale.weba" \
  -F 'config={
    "enable_time_quantization": true,
    "time_division": 5,
    "quantization_force": 0.7,
    "bpm": 120.0
  }' \
  -o response_time_quantized.json
```

### Full Configuration Example
```bash
curl -X POST "http://localhost:7860/transcribe" \
  -F "file=@sample/Chuck_Berry_intro.ogg" \
  -F 'config={
    "note_sensitivity": 0.75,
    "split_sensitivity": 0.55,
    "minimum_note_duration_ms": 100,
    "pitch_bend_mode": "single",
    "enable_note_quantization": true,
    "min_midi_note": 40,
    "max_midi_note": 90,
    "key_root_note": 0,
    "key_type": 1,
    "key_snap_mode": 0,
    "enable_time_quantization": true,
    "time_division": 5,
    "quantization_force": 0.5,
    "bpm": 120,
    "time_signature_num": 4,
    "time_signature_denom": 4
  }' \
  -o response_full.json
```

### Streaming Transcription (NDJSON)

To stream progress and the final result as NDJSON events, use the `stream=true` query parameter and `curl -N`:

```bash
curl -N -X POST "http://localhost:7860/transcribe?stream=true" \
  -F "file=@sample/Chuck_Berry_intro.ogg"
```

You will see one NDJSON line per step (in order):

- `{"event": "transcription_started", "filename": "..."}`
- `{"event": "features_computed", "num_frames": ..., "elapsed_sec": ..., "estimated_total_sec": ...}`
- `{"event": "cnn_run_complete", "elapsed_sec": ...}`
- `{"event": "notes_converted", "num_notes": ..., "elapsed_sec": ...}`
- `{"event": "transcription_complete", "num_notes": ..., "elapsed_sec": ...}`
- `{"event": "result", "elapsed_total_sec": ..., "data": {...}}`

Use `estimated_total_sec` from the first step to show an ETA or progress bar.

## Viewing Results

### Pretty Print JSON
```bash
cat response_berry.json | python -m json.tool
```

### Count Notes
```bash
cat response_berry.json | python -c "import json, sys; d=json.load(sys.stdin); print(f'Found {d[\"num_notes\"]} notes')"
```

### Extract First Note
```bash
cat response_berry.json | python -c "import json, sys; d=json.load(sys.stdin); print(json.dumps(d['notes'][0], indent=2))"
```

### Save MIDI File (if included)
```bash
cat response_berry.json | python -c "import json, sys, base64; d=json.load(sys.stdin); open('output.mid', 'wb').write(base64.b64decode(d['midi_base64']))" 2>/dev/null && echo "MIDI saved to output.mid" || echo "No MIDI in response"
```

## Windows PowerShell Alternative

If using PowerShell instead of Git Bash:

```powershell
# Health check
Invoke-RestMethod -Uri "http://localhost:7860/health"

# Transcribe file
$file = Get-Item "sample\Chuck_Berry_intro.ogg"
$form = @{
    file = $file
    config = '{"note_sensitivity": 0.8}'
}
Invoke-RestMethod -Uri "http://localhost:7860/transcribe" -Method Post -Form $form | ConvertTo-Json -Depth 10
```
