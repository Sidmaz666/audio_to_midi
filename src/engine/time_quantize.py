"""Time quantization - matches TimeQuantizeOptions exactly."""

import numpy as np
from typing import List
from dataclasses import dataclass
from .notes import NoteEvent
from .config import TimeDivision, TIME_DIVISIONS_DOUBLE


@dataclass
class TimeQuantizeInfo:
    """Time quantization info - matches TimeQuantizeOptions::TimeQuantizeInfo."""
    bpm: float = 120.0
    time_signature_num: int = 4
    time_signature_denom: int = 4
    ref_last_bar_qn: float = 0.0
    ref_position_qn: float = 0.0
    ref_position_seconds: float = 0.0
    
    @staticmethod
    def qn_to_sec(duration_qn: float, bpm: float) -> float:
        """Convert quarter notes to seconds."""
        return duration_qn * 60.0 / bpm
    
    @staticmethod
    def sec_to_qn(duration_seconds: float, bpm: float) -> float:
        """Convert seconds to quarter notes."""
        return duration_seconds * bpm / 60.0
    
    def get_start_qn(self) -> float:
        """Get start position in quarter notes."""
        return self.ref_position_qn - self.sec_to_qn(self.ref_position_seconds, self.bpm)
    
    def get_start_last_bar_qn(self) -> float:
        """Get last bar position before recording started in quarter notes."""
        bar_duration_qn = self.time_signature_num * 4.0 / self.time_signature_denom
        start_qn = self.get_start_qn()
        num_bars = int(np.ceil((self.ref_last_bar_qn - start_qn) / bar_duration_qn))
        return self.ref_last_bar_qn - num_bars * bar_duration_qn


def _quantize_time(
    event_time: float,
    bpm: float,
    time_division: float,
    start_time_qn: float,
    quantization_force: float
) -> float:
    """
    Quantize a time value - matches TimeQuantizeOptions::_quantizeTime.
    
    Args:
        event_time: Event time in seconds
        bpm: Tempo in BPM
        time_division: Time division in quarter notes (e.g., 0.25 for 1/4)
        start_time_qn: Start time in quarter notes
        quantization_force: Quantization force (0.0-1.0)
    
    Returns:
        Quantized time in seconds
    """
    import numpy as np
    
    seconds_per_qn = 60.0 / bpm
    division_duration = time_division * 4.0 * seconds_per_qn
    
    # Set previous bar start as new time origin
    new_time_origin = start_time_qn * seconds_per_qn
    shifted_time = event_time + new_time_origin
    
    time_since_previous_division = np.fmod(shifted_time, division_duration)
    
    # Get time of first division tick before note start
    previous_division_time = shifted_time - time_since_previous_division
    
    target_time = (
        previous_division_time if time_since_previous_division < division_duration / 2.0
        else previous_division_time + division_duration
    )
    
    # Linear interpolation between original and target
    quantized_shifted_time = (
        (1.0 - quantization_force) * shifted_time + quantization_force * target_time
    )
    
    # Re-shift
    quantized_time = quantized_shifted_time - new_time_origin
    
    return quantized_time


class TimeQuantizeOptions:
    """Time quantization options - matches TimeQuantizeOptions class."""
    
    def __init__(self):
        """Initialize with defaults."""
        self.enable = False
        self.division = TimeDivision._1_8  # Default index 5
        self.quantization_force = 0.0
        self.info = TimeQuantizeInfo()
    
    def set_parameters(
        self,
        enable: bool,
        division: TimeDivision,
        quantization_force: float,
        bpm: float = 120.0,
        time_signature_num: int = 4,
        time_signature_denom: int = 4
    ):
        """Set parameters - matches TimeQuantizeOptions::setParameters."""
        self.enable = enable
        self.division = division
        self.quantization_force = quantization_force
        self.info.bpm = bpm
        self.info.time_signature_num = time_signature_num
        self.info.time_signature_denom = time_signature_denom
    
    def quantize(self, note_events: List[NoteEvent]) -> List[NoteEvent]:
        """Quantize note events - matches TimeQuantizeOptions::quantize."""
        if not self.enable:
            return note_events
        
        out_events = []
        bpm = self.info.bpm
        
        # Offset from previous bar start
        start_pos_qn = self.info.get_start_qn() - self.info.get_start_last_bar_qn()
        
        time_division = TIME_DIVISIONS_DOUBLE[int(self.division)]
        
        for event in note_events:
            duration = event.end_time - event.start_time
            assert duration > 0
            
            new_start_time = _quantize_time(
                event.start_time,
                bpm,
                time_division,
                start_pos_qn,
                self.quantization_force
            )
            new_end_time = new_start_time + duration
            
            quantized_event = NoteEvent(
                start_time=new_start_time,
                end_time=new_end_time,
                start_frame=event.start_frame,
                end_frame=event.end_frame,
                pitch=event.pitch,
                amplitude=event.amplitude,
                bends=event.bends.copy()
            )
            out_events.append(quantized_event)
        
        return out_events
    
    def clear(self):
        """Clear state - matches TimeQuantizeOptions::clear."""
        self.info.ref_position_qn = 0.0
        self.info.ref_last_bar_qn = 0.0
        self.info.ref_position_seconds = 0.0
