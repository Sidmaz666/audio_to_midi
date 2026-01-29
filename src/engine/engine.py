"""High-level transcription engine - orchestrates all components."""

import logging
import time

import numpy as np
from typing import List, Generator, Tuple
from pathlib import Path

from .features import FeatureExtractor
from .cnn import CnnModel
from .notes import NotesConverter, NoteEvent, drop_overlapping_pitch_bends, merge_overlapping_notes_with_same_pitch
from .note_options import NoteOptions
from .time_quantize import TimeQuantizeOptions
from .config import TranscriptionConfig, PitchBendMode, RootNote, ScaleType, SnapMode, TimeDivision
from .constants import FFT_HOP, BASIC_PITCH_SAMPLE_RATE

# Heuristic: total time ~= features_elapsed * this ratio (from typical run: features ~8s, total ~14.5s)
ESTIMATED_TOTAL_RATIO = 1.85

logger = logging.getLogger("neuralnote.engine")


class TranscriptionEngine:
    """Main transcription engine - matches BasicPitch + TranscriptionManager logic."""
    
    def __init__(self, models_dir: str = None):
        """
        Initialize transcription engine.
        
        Args:
            models_dir: Directory containing model files. If None, uses src/models/
        """
        if models_dir is None:
            src_dir = Path(__file__).parent.parent
            models_dir = str(src_dir / "models")
        
        self.feature_extractor = FeatureExtractor(model_path=str(Path(models_dir) / "features_model.onnx"))
        self.cnn_model = CnnModel(models_dir=models_dir)
        self.notes_converter = NotesConverter()
        self.note_options = NoteOptions()
        self.time_quantize_options = TimeQuantizeOptions()
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int,
        config: TranscriptionConfig
    ) -> List[NoteEvent]:
        """
        Transcribe audio to MIDI note events.
        
        Args:
            audio: Input audio as numpy array (mono, float32)
            sample_rate: Sample rate of input audio
            config: Transcription configuration
        
        Returns:
            List of NoteEvent objects
        """
        t_start = time.perf_counter()

        # 1. Compute features
        t0 = time.perf_counter()
        features, num_frames = self.feature_extractor.compute_features(audio, sample_rate)
        t1 = time.perf_counter()
        logger.info(
            "features_computed",
            extra={"num_frames": int(num_frames), "elapsed_sec": t1 - t0},
        )
        
        # 2. Run CNN
        t2 = time.perf_counter()
        contours_pg, notes_pg, onsets_pg = self.cnn_model.run(features)
        t3 = time.perf_counter()
        logger.info(
            "cnn_run_complete",
            extra={"elapsed_sec": t3 - t2},
        )
        
        # 3. Convert posteriorgrams to note events
        # Build ConvertParams from config
        frame_threshold = 1.0 - config.note_sensitivity
        onset_threshold = 1.0 - config.split_sensitivity
        min_note_length = int(round(
            config.minimum_note_duration_ms / 1000.0 / (FFT_HOP / BASIC_PITCH_SAMPLE_RATE)
        ))
        
        # Convert pitch_bend_mode string to enum
        pitch_bend_mode = PitchBendMode.NO_PITCH_BEND
        if config.pitch_bend_mode == "single":
            pitch_bend_mode = PitchBendMode.SINGLE_PITCH_BEND
        
        t4 = time.perf_counter()
        note_events = self.notes_converter.convert(
            notes_pg=notes_pg,
            onsets_pg=onsets_pg,
            contours_pg=contours_pg,
            frame_threshold=frame_threshold,
            onset_threshold=onset_threshold,
            min_note_length=min_note_length,
            pitch_bend_mode=pitch_bend_mode,
            infer_onsets=True,
            melodia_trick=True,
            energy_threshold=11,
            max_frequency=-1.0,
            min_frequency=-1.0
        )
        t5 = time.perf_counter()
        logger.info(
            "notes_converted",
            extra={"num_notes": int(len(note_events)), "elapsed_sec": t5 - t4},
        )
        
        # 4. Apply note options
        self.note_options.set_parameters(
            enable=config.enable_note_quantization,
            root_note=RootNote(config.key_root_note),
            scale_type=ScaleType(config.key_type),
            snap_mode=SnapMode(config.key_snap_mode),
            min_midi_note=config.min_midi_note,
            max_midi_note=config.max_midi_note
        )
        note_events = self.note_options.process(note_events)
        
        # 5. Apply time quantization
        self.time_quantize_options.set_parameters(
            enable=config.enable_time_quantization,
            division=TimeDivision(config.time_division),
            quantization_force=config.quantization_force,
            bpm=config.bpm,
            time_signature_num=config.time_signature_num,
            time_signature_denom=config.time_signature_denom
        )
        note_events = self.time_quantize_options.quantize(note_events)
        
        # 6. Clean up overlapping notes and pitch bends
        drop_overlapping_pitch_bends(note_events)
        merge_overlapping_notes_with_same_pitch(note_events)
        
        t_end = time.perf_counter()
        logger.info(
            "transcription_engine_complete",
            extra={
                "num_notes": int(len(note_events)),
                "elapsed_total_sec": t_end - t_start,
            },
        )
        
        return note_events

    def transcribe_steps(
        self,
        audio: np.ndarray,
        sample_rate: int,
        config: TranscriptionConfig,
    ) -> Generator[Tuple[str, dict], None, List[NoteEvent]]:
        """
        Run the same pipeline as transcribe(), but yield (event_name, payload) after each step.
        Caller can stream these events to the client. Returns note_events when exhausted.
        """
        t_start = time.perf_counter()

        # 1. Compute features
        t0 = time.perf_counter()
        features, num_frames = self.feature_extractor.compute_features(audio, sample_rate)
        t1 = time.perf_counter()
        features_elapsed = t1 - t0
        estimated_total_sec = round(features_elapsed * ESTIMATED_TOTAL_RATIO, 2)
        logger.info(
            "features_computed",
            extra={"num_frames": int(num_frames), "elapsed_sec": features_elapsed},
        )
        yield ("features_computed", {
            "num_frames": int(num_frames),
            "elapsed_sec": round(features_elapsed, 4),
            "estimated_total_sec": estimated_total_sec,
        })

        # 2. Run CNN
        t2 = time.perf_counter()
        contours_pg, notes_pg, onsets_pg = self.cnn_model.run(features)
        t3 = time.perf_counter()
        cnn_elapsed = t3 - t2
        logger.info(
            "cnn_run_complete",
            extra={"elapsed_sec": cnn_elapsed},
        )
        yield ("cnn_run_complete", {
            "elapsed_sec": round(cnn_elapsed, 4),
        })

        # 3. Convert posteriorgrams to note events
        frame_threshold = 1.0 - config.note_sensitivity
        onset_threshold = 1.0 - config.split_sensitivity
        min_note_length = int(round(
            config.minimum_note_duration_ms / 1000.0 / (FFT_HOP / BASIC_PITCH_SAMPLE_RATE)
        ))
        pitch_bend_mode = PitchBendMode.NO_PITCH_BEND
        if config.pitch_bend_mode == "single":
            pitch_bend_mode = PitchBendMode.SINGLE_PITCH_BEND

        t4 = time.perf_counter()
        note_events = self.notes_converter.convert(
            notes_pg=notes_pg,
            onsets_pg=onsets_pg,
            contours_pg=contours_pg,
            frame_threshold=frame_threshold,
            onset_threshold=onset_threshold,
            min_note_length=min_note_length,
            pitch_bend_mode=pitch_bend_mode,
            infer_onsets=True,
            melodia_trick=True,
            energy_threshold=11,
            max_frequency=-1.0,
            min_frequency=-1.0,
        )
        t5 = time.perf_counter()
        notes_elapsed = t5 - t4
        logger.info(
            "notes_converted",
            extra={"num_notes": int(len(note_events)), "elapsed_sec": notes_elapsed},
        )
        yield ("notes_converted", {
            "num_notes": int(len(note_events)),
            "elapsed_sec": round(notes_elapsed, 4),
        })

        # 4. Apply note options
        self.note_options.set_parameters(
            enable=config.enable_note_quantization,
            root_note=RootNote(config.key_root_note),
            scale_type=ScaleType(config.key_type),
            snap_mode=SnapMode(config.key_snap_mode),
            min_midi_note=config.min_midi_note,
            max_midi_note=config.max_midi_note,
        )
        note_events = self.note_options.process(note_events)

        # 5. Apply time quantization
        self.time_quantize_options.set_parameters(
            enable=config.enable_time_quantization,
            division=TimeDivision(config.time_division),
            quantization_force=config.quantization_force,
            bpm=config.bpm,
            time_signature_num=config.time_signature_num,
            time_signature_denom=config.time_signature_denom,
        )
        note_events = self.time_quantize_options.quantize(note_events)

        # 6. Clean up overlapping notes and pitch bends
        drop_overlapping_pitch_bends(note_events)
        merge_overlapping_notes_with_same_pitch(note_events)

        t_end = time.perf_counter()
        total_elapsed = t_end - t_start
        logger.info(
            "transcription_engine_complete",
            extra={"num_notes": int(len(note_events)), "elapsed_total_sec": total_elapsed},
        )
        yield ("transcription_complete", {
            "num_notes": int(len(note_events)),
            "elapsed_sec": round(total_elapsed, 4),
        })
        return note_events
