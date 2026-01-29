"""MIDI conversion utilities."""

import base64
from typing import List, Dict, Any, Optional
from .notes import NoteEvent

try:
    import mido
    MIDO_AVAILABLE = True
except ImportError:
    MIDO_AVAILABLE = False


def notes_to_json(notes: List[NoteEvent]) -> List[Dict[str, Any]]:
    """
    Convert note events to JSON-serializable format.
    
    Args:
        notes: List of NoteEvent objects
    
    Returns:
        List of dictionaries with note data
    """
    return [
        {
            "pitch": note.pitch,
            "start": note.start_time,
            "end": note.end_time,
            "start_frame": note.start_frame,
            "end_frame": note.end_frame,
            "amplitude": note.amplitude,
            "bends": note.bends
        }
        for note in notes
    ]


def notes_to_midi(notes: List[NoteEvent], bpm: float = 120.0) -> Optional[bytes]:
    """
    Convert note events to MIDI file bytes.
    
    Args:
        notes: List of NoteEvent objects
        bpm: Tempo in BPM
    
    Returns:
        MIDI file as bytes, or None if mido is not available
    """
    if not MIDO_AVAILABLE:
        return None
    
    # Create MIDI file
    mid = mido.MidiFile()
    track = mido.MidiTrack()
    mid.tracks.append(track)
    
    # Set tempo
    tempo = mido.bpm2tempo(bpm)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    
    # Set time signature (default 4/4)
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4))
    
    # Convert time to ticks (assuming 480 ticks per quarter note)
    ticks_per_quarter = 480
    ticks_per_second = ticks_per_quarter * bpm / 60.0
    
    # Sort notes by start time
    sorted_notes = sorted(notes, key=lambda n: n.start_time)
    
    # Track active notes and their note-on times
    active_notes = {}  # pitch -> (start_tick, velocity)
    current_tick = 0
    
    for note in sorted_notes:
        note_start_tick = int(note.start_time * ticks_per_second)
        note_end_tick = int(note.end_time * ticks_per_second)
        velocity = int(min(127, max(1, note.amplitude * 127)))
        
        # Add delay if needed
        if note_start_tick > current_tick:
            track.append(mido.Message('note_on', note=note.pitch, velocity=velocity, time=note_start_tick - current_tick))
            current_tick = note_start_tick
        else:
            track.append(mido.Message('note_on', note=note.pitch, velocity=velocity, time=0))
        
        active_notes[note.pitch] = (current_tick, velocity)
        
        # Handle note off
        note_off_tick = note_end_tick
        if note_off_tick > current_tick:
            track.append(mido.Message('note_off', note=note.pitch, velocity=0, time=note_off_tick - current_tick))
            current_tick = note_off_tick
        else:
            track.append(mido.Message('note_off', note=note.pitch, velocity=0, time=0))
        
        # Handle pitch bends if present
        if note.bends:
            # Pitch bend messages (8192 is center, range is -8192 to 8191)
            # Note: bends are in units of 1/3 semitone
            for i, bend in enumerate(note.bends):
                bend_time = note.start_time + (i * (note.end_time - note.start_time) / len(note.bends))
                bend_tick = int(bend_time * ticks_per_second)
                
                # Convert bend to MIDI pitch bend value
                # 1/3 semitone = 8192 / 3 ≈ 2731
                bend_value = int(bend * 2731) + 8192
                bend_value = max(0, min(16383, bend_value))  # Clamp to valid range
                
                if bend_tick > current_tick:
                    track.append(mido.Message('pitchwheel', pitch=bend_value, time=bend_tick - current_tick))
                    current_tick = bend_tick
                else:
                    track.append(mido.Message('pitchwheel', pitch=bend_value, time=0))
    
    # Close any remaining notes
    for pitch in list(active_notes.keys()):
        track.append(mido.Message('note_off', note=pitch, velocity=0, time=0))
    
    # Save to bytes
    import io
    buffer = io.BytesIO()
    mid.save(file=buffer)
    return buffer.getvalue()


def notes_to_midi_base64(notes: List[NoteEvent], bpm: float = 120.0) -> Optional[str]:
    """
    Convert note events to base64-encoded MIDI file.
    
    Args:
        notes: List of NoteEvent objects
        bpm: Tempo in BPM
    
    Returns:
        Base64-encoded MIDI file as string, or None if mido is not available
    """
    midi_bytes = notes_to_midi(notes, bpm)
    if midi_bytes is None:
        return None
    return base64.b64encode(midi_bytes).decode('utf-8')
