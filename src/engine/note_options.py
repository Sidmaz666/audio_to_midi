"""Note options processing - matches NoteOptions exactly."""

from typing import List
from .notes import NoteEvent
from .constants import MIN_MIDI_NOTE, MAX_MIDI_NOTE
from .config import RootNote, ScaleType, SnapMode


# Scale intervals matching NoteOptions.h
MAJOR_SCALE_INTERVALS = [0, 2, 4, 5, 7, 9, 11]
MINOR_SCALE_INTERVALS = [0, 2, 3, 5, 7, 8, 10]
DORIAN_SCALE_INTERVALS = [0, 2, 3, 5, 7, 9, 10]
MIXOLYDIAN_SCALE_INTERVALS = [0, 2, 4, 5, 7, 9, 10]
LYDIAN_SCALE_INTERVALS = [0, 2, 4, 6, 7, 9, 11]
PHRYGIAN_SCALE_INTERVALS = [0, 1, 3, 5, 7, 8, 10]
LOCRIAN_SCALE_INTERVALS = [0, 1, 3, 5, 6, 8, 10]
MINOR_BLUES_SCALE_INTERVALS = [0, 3, 5, 6, 7, 10]
MINOR_PENTATONIC_SCALE_INTERVALS = [0, 3, 5, 7, 10]
MAJOR_PENTATONIC_SCALE_INTERVALS = [0, 2, 4, 7, 9]
MELODIC_MINOR_SCALE_INTERVALS = [0, 2, 3, 5, 7, 8, 10]
HARMONIC_MINOR_SCALE_INTERVALS = [0, 2, 3, 5, 7, 8, 11]
HARMONIC_MAJOR_SCALE_INTERVALS = [0, 2, 4, 5, 7, 8, 11]


def _root_note_to_note_idx(root_note: RootNote) -> int:
    """Convert root note to note index - matches NoteOptions::_rootNoteToNoteIdx."""
    return (int(root_note) + 12 - 3) % 12


def _midi_to_note_index(midi_note: int) -> int:
    """Get note index (0-11) from MIDI note - matches NoteOptions::_midiToNoteIndex."""
    return midi_note % 12


def _create_key_vector_for_scale(root_note_idx: int, intervals: List[int]) -> List[int]:
    """Create key vector for a scale - matches NoteOptions::_createKeyVectorForScale."""
    return [root_note_idx + interval for interval in intervals]


def _create_key_vector(root_note: RootNote, scale_type: ScaleType) -> List[int]:
    """Create key vector - matches NoteOptions::_createKeyVector."""
    root_note_idx = _root_note_to_note_idx(root_note)
    
    if scale_type == ScaleType.MAJOR:
        return _create_key_vector_for_scale(root_note_idx, MAJOR_SCALE_INTERVALS)
    elif scale_type == ScaleType.MINOR:
        return _create_key_vector_for_scale(root_note_idx, MINOR_SCALE_INTERVALS)
    elif scale_type == ScaleType.DORIAN:
        return _create_key_vector_for_scale(root_note_idx, DORIAN_SCALE_INTERVALS)
    elif scale_type == ScaleType.MIXOLYDIAN:
        return _create_key_vector_for_scale(root_note_idx, MIXOLYDIAN_SCALE_INTERVALS)
    elif scale_type == ScaleType.LYDIAN:
        return _create_key_vector_for_scale(root_note_idx, LYDIAN_SCALE_INTERVALS)
    elif scale_type == ScaleType.PHRYGIAN:
        return _create_key_vector_for_scale(root_note_idx, PHRYGIAN_SCALE_INTERVALS)
    elif scale_type == ScaleType.LOCRIAN:
        return _create_key_vector_for_scale(root_note_idx, LOCRIAN_SCALE_INTERVALS)
    elif scale_type == ScaleType.MINOR_BLUES:
        return _create_key_vector_for_scale(root_note_idx, MINOR_BLUES_SCALE_INTERVALS)
    elif scale_type == ScaleType.MINOR_PENTATONIC:
        return _create_key_vector_for_scale(root_note_idx, MINOR_PENTATONIC_SCALE_INTERVALS)
    elif scale_type == ScaleType.MAJOR_PENTATONIC:
        return _create_key_vector_for_scale(root_note_idx, MAJOR_PENTATONIC_SCALE_INTERVALS)
    elif scale_type == ScaleType.MELODIC_MINOR:
        return _create_key_vector_for_scale(root_note_idx, MELODIC_MINOR_SCALE_INTERVALS)
    elif scale_type == ScaleType.HARMONIC_MINOR:
        return _create_key_vector_for_scale(root_note_idx, HARMONIC_MINOR_SCALE_INTERVALS)
    elif scale_type == ScaleType.HARMONIC_MAJOR:
        return _create_key_vector_for_scale(root_note_idx, HARMONIC_MAJOR_SCALE_INTERVALS)
    else:
        return []  # Chromatic


def _is_in_key(midi_note: int, key_array: List[int]) -> bool:
    """Check if MIDI note is in key - matches NoteOptions::_isInKey."""
    note_index = _midi_to_note_index(midi_note)
    return note_index in key_array


def _get_closest_midi_note_in_key(midi_note: int, key_array: List[int], adjust_up: bool) -> int:
    """Get closest MIDI note in key - matches NoteOptions::_getClosestMidiNoteInKey."""
    if _is_in_key(midi_note, key_array):
        return midi_note
    
    if adjust_up:
        if midi_note < MAX_MIDI_NOTE - 1:
            return midi_note + 1
        else:
            return midi_note - 1
    else:
        if midi_note > MIN_MIDI_NOTE:
            return midi_note - 1
        else:
            return midi_note + 1


class NoteOptions:
    """Note options processing - matches NoteOptions class."""
    
    def __init__(self):
        """Initialize with defaults."""
        self.enable = False
        self.root_note = RootNote.C
        self.scale_type = ScaleType.CHROMATIC
        self.snap_mode = SnapMode.REMOVE
        self.min_midi_note = MIN_MIDI_NOTE
        self.max_midi_note = MAX_MIDI_NOTE
    
    def set_parameters(
        self,
        enable: bool,
        root_note: RootNote,
        scale_type: ScaleType,
        snap_mode: SnapMode,
        min_midi_note: int,
        max_midi_note: int
    ):
        """Set parameters - matches NoteOptions::setParameters."""
        self.enable = enable
        self.root_note = root_note
        self.scale_type = scale_type
        self.snap_mode = snap_mode
        self.min_midi_note = min_midi_note
        self.max_midi_note = max_midi_note
    
    def process(self, note_events: List[NoteEvent]) -> List[NoteEvent]:
        """Process note events - matches NoteOptions::process."""
        if not self.enable:
            return note_events
        
        processed_events = []
        key_vector = _create_key_vector(self.root_note, self.scale_type)
        
        for event in note_events:
            # Filter by MIDI note range
            if event.pitch < self.min_midi_note or event.pitch > self.max_midi_note:
                continue
            
            if self.scale_type == ScaleType.CHROMATIC:
                processed_events.append(event)
            else:
                if self.snap_mode == SnapMode.REMOVE:
                    if _is_in_key(event.pitch, key_vector):
                        processed_events.append(event)
                else:  # ADJUST
                    processed_event = NoteEvent(
                        start_time=event.start_time,
                        end_time=event.end_time,
                        start_frame=event.start_frame,
                        end_frame=event.end_frame,
                        pitch=event.pitch,
                        amplitude=event.amplitude,
                        bends=event.bends.copy()
                    )
                    
                    # If pitch bends are more positive: adjust up, otherwise adjust down
                    adjust_up = sum(event.bends) >= 0 if event.bends else True
                    processed_event.pitch = _get_closest_midi_note_in_key(
                        event.pitch, key_vector, adjust_up
                    )
                    
                    processed_events.append(processed_event)
        
        return processed_events
