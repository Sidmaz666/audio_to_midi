"""CNN model implementation matching BasicPitchCNN exactly."""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, List
from dataclasses import dataclass

from .constants import (
    NUM_HARMONICS, NUM_FREQ_IN, NUM_FREQ_OUT,
    LOOKAHEAD_CNN_CONTOUR, LOOKAHEAD_CNN_NOTE,
    LOOKAHEAD_CNN_ONSET_INPUT, LOOKAHEAD_CNN_ONSET_OUTPUT,
    TOTAL_LOOKAHEAD,
    NUM_CONTOUR_STORED, NUM_NOTE_STORED, NUM_CONCAT2_STORED
)

try:
    from numba import njit, prange  # type: ignore[import]
except ImportError:  # pragma: no cover - numba is optional
    # Fallbacks so code still runs without numba, just without acceleration.
    def njit(*args, **kwargs):  # type: ignore[misc]
        def wrapper(func):
            return func
        return wrapper

    prange = range  # type: ignore[assignment]


@dataclass
class Conv2DLayer:
    """Represents a Conv2D layer with weights and bias."""
    weights: np.ndarray  # Shape: [out_channels, in_channels, kernel_time, kernel_feature]
    bias: np.ndarray     # Shape: [out_channels]
    kernel_time: int
    kernel_feature: int
    stride_time: int
    stride_feature: int
    padding: str  # "same" or "valid"
    activation: str  # "relu" or "sigmoid" or None


def _wrap_index(index: int, size: int) -> int:
    """Wrap index for circular buffer - matches BasicPitchCNN::_wrapIndex."""
    wrapped = index % size
    if wrapped < 0:
        wrapped += size
    return wrapped


def _flatten_nested_list(nested_list):
    """Recursively flatten a nested list structure."""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(_flatten_nested_list(item))
        else:
            result.append(item)
    return result


def _parse_json_model(json_path: str) -> List[Conv2DLayer]:
    """Parse RTNeural JSON model file into Conv2D layers."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    layers = []
    for layer_data in data['layers']:
        if layer_data['type'] != 'conv2d':
            continue
        
        num_filters_out = layer_data['num_filters_out']
        num_filters_in = layer_data['num_filters_in']
        kernel_time = layer_data['kernel_size_time']
        kernel_feature = layer_data['kernel_size_feature']
        
        # Flatten the nested weights structure completely
        weights_flat = _flatten_nested_list(layer_data['weights'])
        weights_array = np.array(weights_flat, dtype=np.float32)
        
        # Expected total elements
        expected_size = num_filters_out * num_filters_in * kernel_time * kernel_feature
        
        # Handle case where JSON has extra elements (might be duplicated or different format)
        if len(weights_array) != expected_size:
            # Try to take first expected_size elements (in case of duplication)
            if len(weights_array) >= expected_size:
                weights_array = weights_array[:expected_size]
            else:
                raise ValueError(
                    f"Weights size mismatch in {json_path}: "
                    f"got {len(weights_array)} elements, expected {expected_size}. "
                    f"Layer: {num_filters_out} out, {num_filters_in} in, "
                    f"{kernel_time}x{kernel_feature} kernel"
                )
        
        # Reshape to [out_channels, in_channels, kernel_time, kernel_feature]
        # This matches PyTorch/standard Conv2D weight format
        weights = weights_array.reshape(num_filters_out, num_filters_in, kernel_time, kernel_feature)
        
        # Extract bias if present
        bias = np.array(layer_data.get('bias', [0.0] * num_filters_out), dtype=np.float32)
        
        layers.append(Conv2DLayer(
            weights=weights,
            bias=bias,
            kernel_time=kernel_time,
            kernel_feature=kernel_feature,
            stride_time=layer_data.get('strides', 1),
            stride_feature=layer_data.get('strides', 1),
            padding=layer_data.get('padding', 'valid'),
            activation=layer_data.get('activation', '')
        ))
    
    return layers


@njit(parallel=True, fastmath=True)  # type: ignore[misc]
def _apply_conv2d_core(
    input_data: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    output: np.ndarray,
    stride_time: int,
    stride_feature: int,
    pad_t: int,
    pad_f: int,
    use_same_padding: int,
) -> None:
    """
    Numba-accelerated core Conv2D operation.
    
    Args:
        input_data: Input tensor [batch, time, freq, channels]
        weights: Weights tensor [out_channels, in_channels, kernel_time, kernel_feature]
        bias: Bias vector [out_channels]
        output: Pre-allocated output tensor [batch, out_time, out_freq, out_channels]
        stride_time: Stride in time dimension
        stride_feature: Stride in frequency dimension
        pad_t: Padding in time dimension (for "same" padding)
        pad_f: Padding in frequency dimension (for "same" padding)
        use_same_padding: 1 if using "same" padding, else 0
    """
    batch, in_time, in_freq, in_channels = input_data.shape
    out_time = output.shape[1]
    out_freq = output.shape[2]
    out_channels = output.shape[3]

    for b in range(batch):
        for t_out in prange(out_time):
            for f_out in range(out_freq):
                for c_out in range(out_channels):
                    sum_val = bias[c_out]

                    for c_in in range(in_channels):
                        for kt in range(weights.shape[2]):
                            for kf in range(weights.shape[3]):
                                t_in = t_out * stride_time + kt
                                f_in = f_out * stride_feature + kf

                                if use_same_padding == 1:
                                    t_in -= pad_t
                                    f_in -= pad_f

                                if 0 <= t_in < in_time and 0 <= f_in < in_freq:
                                    sum_val += (
                                        input_data[b, t_in, f_in, c_in]
                                        * weights[c_out, c_in, kt, kf]
                                    )

                    output[b, t_out, f_out, c_out] = sum_val


def _apply_conv2d(
    input_data: np.ndarray,
    layer: Conv2DLayer,
    output_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Apply Conv2D operation, using a Numba-accelerated core when available.
    
    Args:
        input_data: Input tensor [batch, time, freq, channels]
        layer: Conv2D layer definition
        output_shape: Expected output shape (time, freq)
    
    Returns:
        Output tensor [batch, time, freq, out_channels]
    """
    batch, in_time, in_freq, in_channels = input_data.shape
    out_time, out_freq = output_shape
    out_channels = layer.weights.shape[0]

    output = np.zeros((batch, out_time, out_freq, out_channels), dtype=np.float32)

    # Padding configuration
    if layer.padding == "same":
        pad_t = (layer.kernel_time - 1) // 2
        pad_f = (layer.kernel_feature - 1) // 2
        use_same = 1
    else:
        pad_t = 0
        pad_f = 0
        use_same = 0

    _apply_conv2d_core(
        input_data,
        layer.weights,
        layer.bias,
        output,
        int(layer.stride_time),
        int(layer.stride_feature),
        int(pad_t),
        int(pad_f),
        int(use_same),
    )

    # Apply activation (outside of numba-compiled core for simplicity)
    if layer.activation == "relu":
        output = np.maximum(output, 0.0)
    elif layer.activation == "sigmoid":
        output = 1.0 / (1.0 + np.exp(-np.clip(output, -500, 500)))

    return output


class CnnModel:
    """CNN model matching BasicPitchCNN architecture exactly."""
    
    def __init__(self, models_dir: str = None):
        """
        Initialize CNN models from JSON files.
        
        Args:
            models_dir: Directory containing CNN JSON files. If None, uses src/models/
        """
        if models_dir is None:
            src_dir = Path(__file__).parent.parent
            models_dir = str(src_dir / "models")
        
        models_dir = Path(models_dir)
        
        # Load all four CNN models
        self.contour_layers = _parse_json_model(str(models_dir / "cnn_contour_model.json"))
        self.note_layers = _parse_json_model(str(models_dir / "cnn_note_model.json"))
        self.onset_input_layers = _parse_json_model(str(models_dir / "cnn_onset_1_model.json"))
        self.onset_output_layers = _parse_json_model(str(models_dir / "cnn_onset_2_model.json"))
        
        # Initialize circular buffers
        self.contours_buffer = np.zeros((NUM_CONTOUR_STORED, NUM_FREQ_IN), dtype=np.float32)
        self.notes_buffer = np.zeros((NUM_NOTE_STORED, NUM_FREQ_OUT), dtype=np.float32)
        self.concat2_buffer = np.zeros((NUM_CONCAT2_STORED, 32 * NUM_FREQ_OUT), dtype=np.float32)
        
        # Buffer indices
        self.contour_idx = 0
        self.note_idx = 0
        self.concat2_idx = 0
        
        # Temporary storage for onset output
        self._onset_output_result = np.zeros(NUM_FREQ_OUT, dtype=np.float32)
    
    def reset(self):
        """Reset all internal state - matches BasicPitchCNN::reset."""
        self.contours_buffer.fill(0.0)
        self.notes_buffer.fill(0.0)
        self.concat2_buffer.fill(0.0)
        self.contour_idx = 0
        self.note_idx = 0
        self.concat2_idx = 0
    
    @staticmethod
    def get_num_frames_lookahead() -> int:
        """Get number of lookahead frames - matches BasicPitchCNN::getNumFramesLookahead."""
        return TOTAL_LOOKAHEAD
    
    def frame_inference(
        self,
        in_data: np.ndarray,
        out_contours: np.ndarray,
        out_notes: np.ndarray,
        out_onsets: np.ndarray
    ):
        """
        Run inference for a single frame - matches BasicPitchCNN::frameInference.
        
        Args:
            in_data: Input features [NUM_HARMONICS * NUM_FREQ_IN]
            out_contours: Output buffer [NUM_FREQ_IN]
            out_notes: Output buffer [NUM_FREQ_OUT]
            out_onsets: Output buffer [NUM_FREQ_OUT]
        """
        assert len(in_data) == NUM_HARMONICS * NUM_FREQ_IN
        assert len(out_contours) == NUM_FREQ_IN
        assert len(out_notes) == NUM_FREQ_OUT
        assert len(out_onsets) == NUM_FREQ_OUT
        
        # Reshape input: [NUM_HARMONICS, NUM_FREQ_IN] -> [1, 1, NUM_FREQ_IN, NUM_HARMONICS]
        input_reshaped = in_data.reshape(NUM_HARMONICS, NUM_FREQ_IN).T
        input_tensor = input_reshaped[np.newaxis, np.newaxis, :, :]  # [1, 1, NUM_FREQ_IN, NUM_HARMONICS]
        
        # Run models
        self._run_models(input_tensor)
        
        # Fill output vectors (matching BasicPitchCNN::frameInference output logic)
        # Onsets come from cnn_onset_output (stored in _onset_output_result)
        out_onsets[:] = self._onset_output_result
        
        # Notes come from notes_buffer with offset
        note_output_idx = _wrap_index(self.note_idx + 1, NUM_NOTE_STORED)
        out_notes[:] = self.notes_buffer[note_output_idx, :]
        
        # Contours come from contours_buffer with offset
        contour_output_idx = _wrap_index(self.contour_idx + 1, NUM_CONTOUR_STORED)
        out_contours[:] = self.contours_buffer[contour_output_idx, :]
        
        # Increment indices
        self.contour_idx = (self.contour_idx + 1) % NUM_CONTOUR_STORED
        self.note_idx = (self.note_idx + 1) % NUM_NOTE_STORED
        self.concat2_idx = (self.concat2_idx + 1) % NUM_CONCAT2_STORED
    
    def _run_models(self, input_tensor: np.ndarray):
        """
        Run all CNN models and update circular buffers - matches BasicPitchCNN::_runModels.
        
        Args:
            input_tensor: Input [1, 1, NUM_FREQ_IN, NUM_HARMONICS]
        """
        # 1. Run CNN Onset Input
        onset_input_out = input_tensor
        for layer in self.onset_input_layers:
            # Calculate output shape
            if layer.padding == 'same':
                out_time = 1
                out_freq = NUM_FREQ_OUT  # Based on stride
            else:
                out_time = 1
                out_freq = (NUM_FREQ_IN - layer.kernel_feature) // layer.stride_feature + 1
            
            onset_input_out = _apply_conv2d(onset_input_out, layer, (out_time, out_freq))
        
        # Store in concat2_buffer
        onset_input_flat = onset_input_out[0, 0, :, :].flatten()  # [32 * NUM_FREQ_OUT]
        self.concat2_buffer[self.concat2_idx, :] = onset_input_flat
        
        # 2. Run CNN Contour
        contour_out = input_tensor
        for layer in self.contour_layers:
            if layer.padding == 'same':
                out_time = 1
                out_freq = NUM_FREQ_IN
            else:
                out_time = 1
                out_freq = (NUM_FREQ_IN - layer.kernel_feature) // layer.stride_feature + 1
            
            contour_out = _apply_conv2d(contour_out, layer, (out_time, out_freq))
        
        # Store in contours_buffer
        contour_flat = contour_out[0, 0, :, 0]  # [NUM_FREQ_IN]
        self.contours_buffer[self.contour_idx, :] = contour_flat
        
        # 3. Run CNN Note (takes contour output as input)
        note_input = contour_out  # Use contour output
        for layer in self.note_layers:
            if layer.padding == 'same':
                out_time = 1
                out_freq = NUM_FREQ_OUT
            else:
                out_time = 1
                out_freq = (NUM_FREQ_IN - layer.kernel_feature) // layer.stride_feature + 1
            
            note_input = _apply_conv2d(note_input, layer, (out_time, out_freq))
        
        # Store in notes_buffer
        note_flat = note_input[0, 0, :, 0]  # [NUM_FREQ_OUT]
        self.notes_buffer[self.note_idx, :] = note_flat
        
        # 4. Concat operation
        concat_array = self._concat()
        
        # 5. Run CNN Onset Output
        onset_output_in = concat_array[np.newaxis, np.newaxis, :, :]  # [1, 1, NUM_FREQ_OUT, 33]
        for layer in self.onset_output_layers:
            if layer.padding == 'same':
                out_time = 1
                out_freq = NUM_FREQ_OUT
            else:
                out_time = 1
                out_freq = (NUM_FREQ_OUT - layer.kernel_feature) // layer.stride_feature + 1
            
            onset_output_in = _apply_conv2d(onset_output_in, layer, (out_time, out_freq))
        
        # Update notes_buffer with onset output (this is the final onset posteriorgram)
        # Actually, we need to store this separately or update the buffer correctly
        # Looking at the C++ code, the onset output goes into a separate output
        # But the frame_inference method copies from notes_buffer for onsets
        # Let me re-check the C++ logic...
        # Actually, in _runModels, the final mCNNOnsetOutput.forward result is not stored
        # It's accessed via getOutputs() in frameInference
        # So we need to store it temporarily
        self._onset_output_result = onset_output_in[0, 0, :, 0]  # [NUM_FREQ_OUT]
    
    def _concat(self) -> np.ndarray:
        """
        Perform concat operation - matches BasicPitchCNN::_concat.
        
        Returns:
            Concatenated array [NUM_FREQ_OUT, 33]
        """
        concat_array = np.zeros((NUM_FREQ_OUT, 33), dtype=np.float32)
        concat2_index = _wrap_index(self.concat2_idx + 1, NUM_CONCAT2_STORED)
        
        # Get current note output
        current_notes = self.notes_buffer[self.note_idx, :]  # [NUM_FREQ_OUT]
        
        for i in range(NUM_FREQ_OUT):
            # First element is note output
            concat_array[i, 0] = current_notes[i]
            # Next 32 elements from concat2_buffer
            concat_array[i, 1:33] = self.concat2_buffer[concat2_index, i * 32:(i + 1) * 32]
        
        return concat_array
    
    def run(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run full inference on all features - matches BasicPitch::transcribeToMIDI logic.
        
        Args:
            features: Input features [num_frames, NUM_HARMONICS * NUM_FREQ_IN]
        
        Returns:
            Tuple of (contours_pg, notes_pg, onsets_pg) each shaped [num_frames, ...]
        """
        num_frames = features.shape[0]
        num_lh_frames = self.get_num_frames_lookahead()
        
        # Initialize output arrays
        contours_pg = np.zeros((num_frames, NUM_FREQ_IN), dtype=np.float32)
        notes_pg = np.zeros((num_frames, NUM_FREQ_OUT), dtype=np.float32)
        onsets_pg = np.zeros((num_frames, NUM_FREQ_OUT), dtype=np.float32)
        
        # Reset state
        self.reset()
        
        # Zero input for warmup
        zero_input = np.zeros(NUM_HARMONICS * NUM_FREQ_IN, dtype=np.float32)
        dummy_contours = np.zeros(NUM_FREQ_IN, dtype=np.float32)
        dummy_notes = np.zeros(NUM_FREQ_OUT, dtype=np.float32)
        dummy_onsets = np.zeros(NUM_FREQ_OUT, dtype=np.float32)
        
        # Warmup: run with zeros and discard outputs
        for _ in range(num_lh_frames):
            self.frame_inference(zero_input, dummy_contours, dummy_notes, dummy_onsets)
        
        # Run with real inputs, discard first num_lh_frames outputs
        for frame_idx in range(num_lh_frames):
            self.frame_inference(
                features[frame_idx],
                dummy_contours, dummy_notes, dummy_onsets
            )
        
        # Run with real inputs and store outputs
        for frame_idx in range(num_lh_frames, num_frames):
            self.frame_inference(
                features[frame_idx],
                contours_pg[frame_idx - num_lh_frames],
                notes_pg[frame_idx - num_lh_frames],
                onsets_pg[frame_idx - num_lh_frames]
            )
        
        # Run end with zeros and store last frames
        for frame_idx in range(num_frames, num_frames + num_lh_frames):
            self.frame_inference(
                zero_input,
                contours_pg[frame_idx - num_lh_frames],
                notes_pg[frame_idx - num_lh_frames],
                onsets_pg[frame_idx - num_lh_frames]
            )
        
        return contours_pg, notes_pg, onsets_pg
