"""Note event creation from posteriorgrams - matches Notes.cpp exactly."""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

from .constants import (
    FFT_HOP, AUDIO_SAMPLE_RATE, AUDIO_WINDOW_LENGTH, MIDI_OFFSET, MAX_NOTE_IDX,
    ANNOTATIONS_BASE_FREQUENCY, CONTOURS_BINS_PER_SEMITONE,
    NUM_FREQ_OUT, NUM_BINS_TOLERANCE, PITCH_BEND_STD, ENERGY_THRESHOLD
)
from .config import PitchBendMode


@dataclass
class NoteEvent:
    """Note event matching Notes::Event."""
    start_time: float
    end_time: float
    start_frame: int
    end_frame: int
    pitch: int  # MIDI note number
    amplitude: float
    bends: List[int]  # Pitch bend per frame (units: 1/3 semitone)


def _model_frame_to_time(frame: int) -> float:
    """
    Convert frame index to time in seconds - matches Notes::_modelFrameToTime.
    Uses the test version to match reference outputs exactly.
    """
    # Constants matching Notes.h
    ANNOTATIONS_FPS = AUDIO_SAMPLE_RATE // FFT_HOP  # Integer division
    ANNOT_N_FRAMES = ANNOTATIONS_FPS * AUDIO_WINDOW_LENGTH
    AUDIO_N_SAMPLES = AUDIO_SAMPLE_RATE * AUDIO_WINDOW_LENGTH - FFT_HOP
    WINDOW_OFFSET = (FFT_HOP / float(AUDIO_SAMPLE_RATE)) * (ANNOT_N_FRAMES - AUDIO_N_SAMPLES / float(FFT_HOP)) + 0.0018
    
    # Test version formula (USE_TEST_NOTE_FRAME_TO_TIME)
    # Note: frame / ANNOT_N_FRAMES uses integer division in C++ (both are int)
    # So we use // for integer division here
    return (frame * FFT_HOP) / float(AUDIO_SAMPLE_RATE) - WINDOW_OFFSET * (frame // ANNOT_N_FRAMES)


def _inferred_onsets(
    onsets_pg: np.ndarray,
    notes_pg: np.ndarray,
    num_diffs: int = 2
) -> np.ndarray:
    """
    Infer onsets from note posteriorgrams - matches Notes::_inferredOnsets.
    
    Args:
        onsets_pg: Onset posteriorgrams [num_frames, num_notes]
        notes_pg: Note posteriorgrams [num_frames, num_notes]
        num_diffs: Maximum offset for difference calculation
    
    Returns:
        Inferred onsets [num_frames, num_notes]
    """
    n_frames, n_notes = notes_pg.shape
    
    # Initialize notes_diff to all 1s
    notes_diff = np.ones((n_frames, n_notes), dtype=np.float32)
    
    max_min_notes_diff = 0.0
    max_onset = 0.0
    
    # For each frame offset
    for n in range(num_diffs):
        offset = n + 1
        # For each frame
        for i in range(n_frames):
            i_behind = i - offset
            # For each note
            for j in range(n_notes):
                # Calculate difference
                diff = notes_pg[i, j] - (notes_pg[i_behind, j] if i_behind >= 0 else 0.0)
                
                # Update minimum (matching C++ logic)
                if diff < notes_diff[i, j]:
                    diff = max(0.0, diff)  # Zero negative diffs
                    notes_diff[i, j] = diff if i >= num_diffs else 0.0
                
                # If last diff, compute max values
                if offset == num_diffs:
                    onset = onsets_pg[i, j]
                    if onset > max_onset:
                        max_onset = onset
                    if notes_diff[i, j] > max_min_notes_diff:
                        max_min_notes_diff = notes_diff[i, j]
    
    # Rescale notes_diff to match scale of original onsets
    if max_min_notes_diff > 0:
        notes_diff = max_onset * notes_diff / max_min_notes_diff
    
    # Element-wise max with original onsets
    inferred = np.maximum(notes_diff, onsets_pg)
    
    return inferred


def _hz_to_midi(hz: float) -> int:
    """Convert frequency to MIDI note - matches NoteUtils::hzToMidi."""
    return int(round(12.0 * np.log2(hz / 440.0) + 69.0))


class NotesConverter:
    """Converts posteriorgrams to note events - matches Notes class."""
    
    def __init__(self):
        """Initialize converter."""
        self.remaining_energy = None
        self.remaining_energy_index = None
    
    def convert(
        self,
        notes_pg: np.ndarray,
        onsets_pg: np.ndarray,
        contours_pg: np.ndarray,
        frame_threshold: float,
        onset_threshold: float,
        min_note_length: int,
        pitch_bend_mode: PitchBendMode,
        infer_onsets: bool = True,
        melodia_trick: bool = True,
        energy_threshold: int = ENERGY_THRESHOLD,
        max_frequency: float = -1.0,
        min_frequency: float = -1.0
    ) -> List[NoteEvent]:
        """
        Convert posteriorgrams to note events - matches Notes::convert.
        
        Args:
            notes_pg: Note posteriorgrams [num_frames, NUM_FREQ_OUT]
            onsets_pg: Onset posteriorgrams [num_frames, NUM_FREQ_OUT]
            contours_pg: Contour posteriorgrams [num_frames, NUM_FREQ_IN]
            frame_threshold: Frame threshold (1 - note_sensitivity)
            onset_threshold: Onset threshold (1 - split_sensitivity)
            min_note_length: Minimum note length in frames
            pitch_bend_mode: Pitch bend mode
            infer_onsets: Whether to infer onsets
            melodia_trick: Whether to use melodia trick
            energy_threshold: Energy threshold for note detection
            max_frequency: Maximum frequency in Hz (-1 for unset)
            min_frequency: Minimum frequency in Hz (-1 for unset)
        
        Returns:
            List of NoteEvent objects
        """
        n_frames = notes_pg.shape[0]
        if n_frames == 0:
            return []
        
        n_notes = notes_pg.shape[1]
        assert n_notes == NUM_FREQ_OUT
        assert onsets_pg.shape == (n_frames, n_notes)
        assert contours_pg.shape[0] == n_frames
        
        # Infer onsets if needed
        if infer_onsets:
            onsets = _inferred_onsets(onsets_pg, notes_pg)
        else:
            onsets = onsets_pg
        
        # Initialize remaining energy
        self.remaining_energy = notes_pg.copy()
        
        # Initialize remaining energy index for melodia trick
        if melodia_trick:
            self.remaining_energy_index = []
            for frame_idx in range(n_frames):
                for freq_idx in range(NUM_FREQ_OUT):
                    self.remaining_energy_index.append((
                        frame_idx, freq_idx, self.remaining_energy[frame_idx, freq_idx]
                    ))
            # Sort by energy (descending) - will be done later
        
        events = []
        
        # Constrain frequencies
        max_note_idx = NUM_FREQ_OUT - 1
        min_note_idx = 0
        if max_frequency >= 0:
            max_note_idx = _hz_to_midi(max_frequency) - MIDI_OFFSET
        if min_frequency >= 0:
            min_note_idx = max(0, _hz_to_midi(min_frequency) - MIDI_OFFSET)
        
        # Stop 1 frame early
        last_frame = n_frames - 1
        
        # Go backwards in time (matching C++ logic)
        for frame_idx in range(last_frame - 1, -1, -1):
            for note_idx in range(max_note_idx, min_note_idx - 1, -1):
                onset = onsets[frame_idx, note_idx]
                
                # Equivalent to argrelmax logic
                prev = onsets[frame_idx - 1, note_idx] if frame_idx > 0 else onset
                next_val = onsets[frame_idx + 1, note_idx] if frame_idx < last_frame else onset
                
                if onset < onset_threshold or onset < prev or onset < next_val:
                    continue
                
                # Find time index where frames drop below threshold
                i = frame_idx + 1
                k = 0
                while i < last_frame and k < energy_threshold:
                    if self.remaining_energy[i, note_idx] < frame_threshold:
                        k += 1
                    else:
                        k = 0
                    i += 1
                
                i -= k  # Go back to frame above threshold
                
                # Skip if too short
                if i - frame_idx <= min_note_length:
                    continue
                
                # Calculate amplitude and zero out energy
                amplitude = 0.0
                for f in range(frame_idx, i):
                    amplitude += self.remaining_energy[f, note_idx]
                    self.remaining_energy[f, note_idx] = 0.0
                    
                    # Zero neighbors
                    if note_idx < MAX_NOTE_IDX:
                        self.remaining_energy[f, note_idx + 1] = 0.0
                    if note_idx > 0:
                        self.remaining_energy[f, note_idx - 1] = 0.0
                
                amplitude /= (i - frame_idx)
                
                events.append(NoteEvent(
                    start_time=_model_frame_to_time(frame_idx),
                    end_time=_model_frame_to_time(i),
                    start_frame=frame_idx,
                    end_frame=i,
                    pitch=note_idx + MIDI_OFFSET,
                    amplitude=float(amplitude),
                    bends=[]
                ))
        
        # Melodia trick: process remaining energy
        if melodia_trick:
            # Sort by CURRENT energy value (descending) - matching C++ pointer behavior
            # C++ uses pointers so *a.value gets current value, not stale copy
            self.remaining_energy_index.sort(
                key=lambda x: self.remaining_energy[x[0], x[1]], 
                reverse=True
            )
            
            for frame_idx, note_idx, _ in self.remaining_energy_index:
                # Apply frequency filtering (matching C++ behavior - constraints apply to both passes)
                if note_idx < min_note_idx or note_idx > max_note_idx:
                    continue
                
                # Get current energy value (may have been zeroed in first pass)
                energy = self.remaining_energy[frame_idx, note_idx]
                
                # Skip if already zeroed
                if energy == 0.0:
                    continue
                
                if energy <= frame_threshold:
                    break
                
                self.remaining_energy[frame_idx, note_idx] = 0.0
                
                # Inhibit function
                def inhibit(pg, frame_i, note_i, k_val):
                    if pg[frame_i, note_i] < frame_threshold:
                        k_val += 1
                    else:
                        k_val = 0
                    pg[frame_i, note_i] = 0.0
                    if note_i < MAX_NOTE_IDX:
                        pg[frame_i, note_i + 1] = 0.0
                    if note_i > 0:
                        pg[frame_i, note_i - 1] = 0.0
                    return k_val
                
                # Forward pass
                i = frame_idx + 1
                k = 0
                while i < last_frame and k < energy_threshold:
                    k = inhibit(self.remaining_energy, i, note_idx, k)
                    i += 1
                
                i_end = i - 1 - k
                
                # Backward pass
                i = frame_idx - 1
                k = 0
                while i > 0 and k < energy_threshold:
                    k = inhibit(self.remaining_energy, i, note_idx, k)
                    i -= 1
                
                i_start = i + 1 + k
                
                # Skip if too short
                if i_end - i_start <= min_note_length:
                    continue
                
                # Calculate amplitude
                amplitude = np.mean(notes_pg[i_start:i_end, note_idx])
                
                events.append(NoteEvent(
                    start_time=_model_frame_to_time(i_start),
                    end_time=_model_frame_to_time(i_end),
                    start_frame=i_start,
                    end_frame=i_end,
                    pitch=note_idx + MIDI_OFFSET,
                    amplitude=float(amplitude),
                    bends=[]
                ))
        
        # Sort events
        events.sort(key=lambda e: (e.start_frame, e.end_frame))
        
        # Add pitch bends if needed
        if pitch_bend_mode != PitchBendMode.NO_PITCH_BEND:
            _add_pitch_bends(events, contours_pg)
            # Single pitch bend mode drops overlapping bends, multi keeps all
            if pitch_bend_mode == PitchBendMode.SINGLE_PITCH_BEND:
                drop_overlapping_pitch_bends(events)
        
        return events
    
    def clear(self):
        """Clear internal state - matches Notes::clear."""
        self.remaining_energy = None
        self.remaining_energy_index = None


def _add_pitch_bends(events: List[NoteEvent], contours_pg: np.ndarray, num_bins_tolerance: int = NUM_BINS_TOLERANCE):
    """
    Add pitch bends to events - matches Notes::_addPitchBends.
    
    Args:
        events: List of note events (modified in place)
        contours_pg: Contour posteriorgrams [num_frames, NUM_FREQ_IN]
        num_bins_tolerance: Number of bins tolerance
    """
    for event in events:
        # midi_pitch_to_contour_bin
        note_idx = CONTOURS_BINS_PER_SEMITONE * (
            event.pitch - 69 + 12 * int(round(np.log2(440.0 / ANNOTATIONS_BASE_FREQUENCY)))
        )
        
        n_freq_bins_contours = NUM_FREQ_OUT * CONTOURS_BINS_PER_SEMITONE
        note_start_idx = max(note_idx - num_bins_tolerance, 0)
        note_end_idx = min(n_freq_bins_contours, note_idx + num_bins_tolerance + 1)
        
        gauss_start = float(max(0, num_bins_tolerance - note_idx))
        pb_shift = num_bins_tolerance - max(0, num_bins_tolerance - note_idx)
        
        event.bends = []
        for i in range(event.start_frame, event.end_frame):
            bend = 0
            max_val = 0.0
            
            for j in range(note_start_idx, note_end_idx):
                k = j - note_start_idx
                x = gauss_start + float(k)
                n = x - float(num_bins_tolerance)
                
                # Gaussian weight
                w = np.exp(-(n * n) / (2.0 * PITCH_BEND_STD * PITCH_BEND_STD)) * contours_pg[i, j]
                
                if w > max_val:
                    bend = k
                    max_val = w
            
            event.bends.append(bend - pb_shift)


def drop_overlapping_pitch_bends(events: List[NoteEvent]):
    """
    Drop pitch bends for overlapping notes - matches Notes::dropOverlappingPitchBends.
    
    Args:
        events: List of note events (modified in place), must be sorted
    """
    for i in range(len(events) - 1):
        event = events[i]
        for j in range(i + 1, len(events)):
            event2 = events[j]
            if event2.start_frame >= event.end_frame:
                break
            event.bends = []
            event2.bends = []


def merge_overlapping_notes_with_same_pitch(events: List[NoteEvent]):
    """
    Merge overlapping notes with same pitch - matches Notes::mergeOverlappingNotesWithSamePitch.
    
    Args:
        events: List of note events (modified in place), must be sorted
    """
    events.sort(key=lambda e: (e.start_frame, e.end_frame))
    
    i = 0
    while i < len(events) - 1:
        event = events[i]
        j = i + 1
        while j < len(events):
            event2 = events[j]
            
            # If notes don't overlap, break
            if event2.start_frame >= event.end_frame:
                break
            
            # If notes overlap and have same pitch: merge them
            if event.pitch == event2.pitch:
                event.end_time = event2.end_time
                event.end_frame = event2.end_frame
                events.pop(j)
            else:
                j += 1
        i += 1
