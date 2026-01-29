"""Configuration models matching ParameterHelpers defaults exactly."""

from enum import IntEnum
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator


class PitchBendMode(IntEnum):
    """Pitch bend mode enum matching PitchBendModes."""
    NO_PITCH_BEND = 0
    SINGLE_PITCH_BEND = 1
    MULTI_PITCH_BEND = 2


class RootNote(IntEnum):
    """Root note enum matching NoteUtils::RootNote."""
    A = 0
    A_SHARP = 1
    B = 2
    C = 3
    C_SHARP = 4
    D = 5
    D_SHARP = 6
    E = 7
    F_SHARP = 8
    F = 9
    G_SHARP = 10
    G = 11


class ScaleType(IntEnum):
    """Scale type enum matching NoteUtils::ScaleType."""
    CHROMATIC = 0
    MAJOR = 1
    MINOR = 2
    DORIAN = 3
    MIXOLYDIAN = 4
    LYDIAN = 5
    PHRYGIAN = 6
    LOCRIAN = 7
    MINOR_BLUES = 8
    MINOR_PENTATONIC = 9
    MAJOR_PENTATONIC = 10
    MELODIC_MINOR = 11
    HARMONIC_MINOR = 12
    HARMONIC_MAJOR = 13


class SnapMode(IntEnum):
    """Snap mode enum matching NoteUtils::SnapMode."""
    ADJUST = 0
    REMOVE = 1


class TimeDivision(IntEnum):
    """Time division enum matching TimeQuantizeUtils::TimeDivisions."""
    _1_1 = 0
    _1_2 = 1
    _1_3 = 2
    _1_4 = 3
    _1_6 = 4
    _1_8 = 5
    _1_12 = 6
    _1_16 = 7
    _1_24 = 8
    _1_32 = 9
    _1_48 = 10
    _1_64 = 11


# Time division values in quarter notes
TIME_DIVISIONS_DOUBLE = [
    1.0 / 1.0, 1.0 / 2.0, 1.0 / 3.0, 1.0 / 4.0,
    1.0 / 6.0, 1.0 / 8.0, 1.0 / 12.0, 1.0 / 16.0,
    1.0 / 24.0, 1.0 / 32.0, 1.0 / 48.0, 1.0 / 64.0
]


class TranscriptionConfig(BaseModel):
    """Complete transcription configuration matching ParameterHelpers defaults."""
    
    # Transcription parameters (defaults from ParameterHelpers::getRangedAudioParamForID)
    note_sensitivity: float = Field(
        default=0.7,
        ge=0.05,
        le=0.95,
        description="Note sensitivity threshold (0.05-0.95). Higher gives more notes."
    )
    split_sensitivity: float = Field(
        default=0.5,
        ge=0.05,
        le=0.95,
        description="Split sensitivity threshold (0.05-0.95). Higher will split notes more."
    )
    minimum_note_duration_ms: float = Field(
        default=125.0,
        ge=35.0,
        le=580.0,
        description="Minimum note duration in milliseconds."
    )
    pitch_bend_mode: Literal["no", "single"] = Field(
        default="no",
        description="Pitch bend mode: 'no' or 'single'."
    )
    
    # Note options
    enable_note_quantization: bool = Field(
        default=False,
        description="Enable note quantization to scale."
    )
    min_midi_note: int = Field(
        default=21,  # MIN_MIDI_NOTE
        ge=21,
        le=108,
        description="Minimum MIDI note number."
    )
    max_midi_note: int = Field(
        default=108,  # MAX_MIDI_NOTE
        ge=21,
        le=108,
        description="Maximum MIDI note number."
    )
    key_root_note: int = Field(
        default=3,  # D#/Eb (matching ParameterHelpers default)
        ge=0,
        le=11,
        description="Root note index (0-11: A, A#, B, C, C#, D, D#, E, F, F#, G, G#)."
    )
    key_type: int = Field(
        default=0,  # Chromatic
        ge=0,
        le=13,
        description="Scale type index (0=Chromatic, 1=Major, 2=Minor, etc.)."
    )
    key_snap_mode: int = Field(
        default=0,  # Adjust
        ge=0,
        le=1,
        description="Snap mode (0=Adjust, 1=Remove)."
    )
    
    # Time quantization
    enable_time_quantization: bool = Field(
        default=False,
        description="Enable time quantization."
    )
    time_division: int = Field(
        default=5,  # 1/8 (matching ParameterHelpers default)
        ge=0,
        le=11,
        description="Time division index (0=1/1, 1=1/2, ..., 5=1/8, etc.)."
    )
    quantization_force: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Quantization force (0.0-1.0)."
    )
    
    # Transport/time info (for time quantization)
    bpm: float = Field(
        default=120.0,
        gt=0.0,
        description="Tempo in BPM."
    )
    time_signature_num: int = Field(
        default=4,
        gt=0,
        description="Time signature numerator."
    )
    time_signature_denom: int = Field(
        default=4,
        gt=0,
        description="Time signature denominator."
    )
    
    class Config:
        """Pydantic config."""
        use_enum_values = False
