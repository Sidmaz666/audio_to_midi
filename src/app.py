"""FastAPI application for NeuralNote transcription API."""

import asyncio
import os
import uuid
import json
import shutil
import queue
import threading
from pathlib import Path
from typing import Optional
import time
import logging

import numpy as np
import soundfile as sf
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import ValidationError

from engine.engine import TranscriptionEngine
from engine.config import TranscriptionConfig
from engine.midi import notes_to_json, notes_to_midi_base64

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("neuralnote.api")


app = FastAPI(
    title="NeuralNote Transcription API",
    description="Audio → MIDI transcription API backed by NeuralNote engine",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can be restricted to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize engine (lazy loading could be added if needed)
_engine: Optional[TranscriptionEngine] = None


def get_engine() -> TranscriptionEngine:
    """Get or create transcription engine instance."""
    global _engine
    if _engine is None:
        _engine = TranscriptionEngine()
    return _engine


# Temporary directory for uploads (cross-platform)
import tempfile
TMP_DIR = Path(tempfile.gettempdir()) / "neuralnote_uploads"
TMP_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    config: Optional[str] = Form(None),
    stream: bool = Query(
        False,
        description="If true, stream progress and final result as NDJSON.",
    ),
):
    """
    Transcribe audio file to MIDI note events.
    
    Args:
        file: Audio file (wav, aiff, flac, mp3, ogg, weba)
        config: Optional JSON string with transcription parameters
    
    Returns:
        JSON response with note events and optional MIDI file
    """
    logger.info(
        "request_received",
        extra={"upload_filename": file.filename, "config_provided": bool(config), "stream": stream},
    )

    # 1. Validate file extension
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    if not file.filename.lower().endswith((".wav", ".aiff", ".aif", ".flac", ".mp3", ".ogg", ".weba")):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.filename}. Supported: wav, aiff, flac, mp3, ogg, weba"
        )
    
    # 2. Save upload to temporary file
    tmp_id = uuid.uuid4().hex
    input_path = TMP_DIR / f"{tmp_id}_{file.filename}"
    
    try:
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        await file.close()
    try:
        # 3. Load audio file - handle .weba with librosa, others with soundfile
        try:
            file_ext = Path(file.filename).suffix.lower()
            if file_ext == ".weba":
                if not LIBROSA_AVAILABLE:
                    raise HTTPException(
                        status_code=400,
                        detail="WebA format requires librosa library. Install with: pip install librosa"
                    )
                # librosa loads as mono by default
                audio_data, sample_rate = librosa.load(str(input_path), sr=None, mono=True, dtype='float32')
            else:
                audio_data, sample_rate = sf.read(str(input_path), dtype='float32')
                # Convert to mono if needed
                if audio_data.ndim > 1:
                    audio_data = np.mean(audio_data, axis=-1)
        except Exception as e:
            logger.exception("audio_load_failed")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to load audio file: {str(e)}"
            )

        logger.info(
            "audio_loaded",
            extra={
                "audio_filename": file.filename,
                "sample_rate": int(sample_rate),
                "num_samples": int(len(audio_data)),
            },
        )
        
        # Normalize to reasonable range (avoid clipping)
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.95
            logger.info("audio_normalized", extra={"max_before": float(max_val)})
        
        # 4. Parse config
        try:
            if config:
                config_dict = json.loads(config)
                transcription_config = TranscriptionConfig(**config_dict)
            else:
                transcription_config = TranscriptionConfig()  # Use defaults
        except (json.JSONDecodeError, ValidationError) as e:
            logger.exception("config_parse_failed")
            raise HTTPException(
                status_code=400,
                detail=f"Invalid config JSON: {str(e)}"
            )

        logger.info("config_parsed")

        # Non-streaming mode: keep existing behavior
        if not stream:
            # 5. Run transcription
            try:
                engine = get_engine()
                t0 = time.perf_counter()
                note_events = engine.transcribe(audio_data, sample_rate, transcription_config)
                t1 = time.perf_counter()
            except Exception as e:
                logger.exception("transcription_failed")
                raise HTTPException(
                    status_code=500,
                    detail=f"Transcription failed: {str(e)}"
                )
            
            logger.info(
                "transcription_complete",
                extra={"num_notes": int(len(note_events)), "elapsed_sec": t1 - t0},
            )
            
            # 6. Convert to JSON
            notes_json = notes_to_json(note_events)
            
            # 7. Optionally generate MIDI
            midi_base64 = None
            try:
                midi_base64 = notes_to_midi_base64(note_events, transcription_config.bpm)
            except Exception:
                # MIDI generation is optional, don't fail if it doesn't work
                logger.exception("midi_generation_failed")
                pass
            
            # 8. Return response
            response = {
                "notes": notes_json,
                "num_notes": len(notes_json)
            }
            
            if midi_base64:
                response["midi_base64"] = midi_base64
            
            return JSONResponse(content=response)

        # Streaming mode: run pipeline step-by-step, yield each event + estimated_total_sec
        async def event_stream():
            start_time = time.perf_counter()
            try:
                yield json.dumps(
                    {
                        "event": "transcription_started",
                        "filename": file.filename,
                    }
                ) + "\n"

                engine = get_engine()
                q = queue.Queue()

                def run_steps():
                    try:
                        gen = engine.transcribe_steps(
                            audio_data, sample_rate, transcription_config
                        )
                        try:
                            while True:
                                event_name, payload = next(gen)
                                q.put(("event", event_name, payload))
                        except StopIteration as e:
                            note_events = e.value
                        q.put(("done", note_events))
                    except Exception as e:
                        logger.exception("transcription_failed_stream")
                        q.put(("error", e))

                thread = threading.Thread(target=run_steps, daemon=True)
                thread.start()

                note_events = None
                while True:
                    item = await asyncio.to_thread(q.get)
                    if item[0] == "event":
                        event_name, payload = item[1], item[2]
                        yield json.dumps({"event": event_name, **payload}) + "\n"
                    elif item[0] == "done":
                        note_events = item[1]
                        break
                    else:
                        yield json.dumps(
                            {
                                "event": "error",
                                "status": 500,
                                "detail": str(item[1]),
                            }
                        ) + "\n"
                        return

                if note_events is None:
                    yield json.dumps(
                        {
                            "event": "error",
                            "status": 500,
                            "detail": "Transcription did not return notes",
                        }
                    ) + "\n"
                    return

                notes_json = notes_to_json(note_events)
                midi_base64 = None
                try:
                    midi_base64 = notes_to_midi_base64(
                        note_events, transcription_config.bpm
                    )
                except Exception:
                    logger.exception("midi_generation_failed_stream")
                    pass

                final_response = {
                    "notes": notes_json,
                    "num_notes": len(notes_json),
                }
                if midi_base64:
                    final_response["midi_base64"] = midi_base64

                total_elapsed = time.perf_counter() - start_time
                yield json.dumps(
                    {
                        "event": "result",
                        "elapsed_total_sec": round(total_elapsed, 4),
                        "data": final_response,
                    }
                ) + "\n"

            except HTTPException as e:
                logger.exception("stream_http_exception")
                yield json.dumps(
                    {
                        "event": "error",
                        "status": e.status_code,
                        "detail": e.detail,
                    }
                ) + "\n"
            except Exception as e:
                logger.exception("stream_unexpected_exception")
                yield json.dumps(
                    {
                        "event": "error",
                        "status": 500,
                        "detail": str(e),
                    }
                ) + "\n"

        return StreamingResponse(event_stream(), media_type="application/x-ndjson")
    
    finally:
        # Cleanup
        try:
            if input_path.exists():
                input_path.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn  # pyright: ignore[reportMissingImports]
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
