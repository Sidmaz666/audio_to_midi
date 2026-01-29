"""Feature extraction using ONNX model - matches Features.cpp exactly."""

import os
import numpy as np
import onnxruntime as ort
from pathlib import Path
from typing import Tuple

from .constants import NUM_HARMONICS, NUM_FREQ_IN, BASIC_PITCH_SAMPLE_RATE


class FeatureExtractor:
    """Extracts CQT + harmonic stacking features using ONNX model."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize ONNX session.
        
        Args:
            model_path: Path to features_model.onnx. If None, looks in src/models/
        """
        if model_path is None:
            src_dir = Path(__file__).parent.parent
            model_path = str(src_dir / "models" / "features_model.onnx")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found at {model_path}")
        
        # Configure ONNX Runtime session (matching Features.cpp)
        sess_options = ort.SessionOptions()
        sess_options.inter_op_num_threads = 1
        sess_options.intra_op_num_threads = 1
        
        # Create session
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Input/output names (matching Features.h)
        self.input_names = ["input_1"]
        self.output_names = ["harmonic_stacking"]
        
        # Verify output shape expectations
        output_shape = self.session.get_outputs()[0].shape
        assert output_shape[0] == 1 and output_shape[2] == NUM_FREQ_IN and output_shape[3] == NUM_HARMONICS, \
            f"Unexpected output shape: {output_shape}"
    
    def compute_features(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
        """
        Compute features from audio.
        
        Args:
            audio: Input audio as 1D numpy array (mono, float32)
            sample_rate: Sample rate of input audio
            
        Returns:
            Tuple of (features_array, num_frames) where:
            - features_array: Shape [num_frames, NUM_HARMONICS * NUM_FREQ_IN]
            - num_frames: Number of frames computed
        """
        # Ensure audio is float32 and 1D
        if audio.ndim > 1:
            # Convert to mono if needed
            audio = np.mean(audio, axis=-1)
        audio = audio.astype(np.float32)
        
        # Resample to BASIC_PITCH_SAMPLE_RATE if needed
        if sample_rate != BASIC_PITCH_SAMPLE_RATE:
            from scipy import signal
            num_samples = int(len(audio) * BASIC_PITCH_SAMPLE_RATE / sample_rate)
            audio = signal.resample(audio, num_samples).astype(np.float32)
        
        num_samples = len(audio)
        
        # Prepare input tensor: [1, num_samples, 1] (matching Features.cpp)
        input_shape = np.array([1, num_samples, 1], dtype=np.int64)
        input_tensor = np.expand_dims(audio, axis=(0, 2))  # Shape: [1, num_samples, 1]
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_names[0]: input_tensor}
        )
        
        # Extract output: shape should be [1, num_frames, NUM_FREQ_IN, NUM_HARMONICS]
        output = outputs[0]
        assert output.shape[0] == 1
        assert output.shape[2] == NUM_FREQ_IN
        assert output.shape[3] == NUM_HARMONICS
        
        num_frames = output.shape[1]
        
        # Reshape to [num_frames, NUM_HARMONICS * NUM_FREQ_IN]
        # The stacking order should match what the CNN expects
        # From the C++ code, it expects: [num_frames, NUM_HARMONICS * NUM_FREQ_IN]
        # Where each frame has NUM_HARMONICS stacked on top of NUM_FREQ_IN
        features = output[0]  # Remove batch dimension: [num_frames, NUM_FREQ_IN, NUM_HARMONICS]
        
        # Reshape to stack harmonics: [num_frames, NUM_HARMONICS * NUM_FREQ_IN]
        # Each harmonic is stacked vertically (NUM_FREQ_IN bins per harmonic)
        features_stacked = features.reshape(num_frames, NUM_HARMONICS * NUM_FREQ_IN)
        
        return features_stacked, num_frames
