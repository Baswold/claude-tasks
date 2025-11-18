"""
Gesture generation module.
Generates bone animation data from audio features using ML model or rule-based fallback.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import os
import json

logger = logging.getLogger(__name__)


class GestureGenerator:
    """Generates gesture sequences from audio features."""

    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize gesture generator.

        Args:
            model_path: Path to trained model file (ONNX or TorchScript)
            config: Configuration dictionary with gesture parameters
        """
        self.model_path = model_path
        self.model = None
        self.config = config or self._default_config()

        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
        else:
            logger.info("No model loaded, will use rule-based generation")

    def _default_config(self) -> Dict:
        """Return default configuration."""
        return {
            'gesture_intensity': 1.0,
            'smoothing': 0.5,
            'idle_motion_scale': 0.3,
            'emphasis_scale': 1.5,
            'breathing_rate': 0.2,  # Hz
            'head_nod_threshold': 0.7,
            'hand_gesture_threshold': 0.6
        }

    def _load_model(self, model_path: str):
        """
        Load trained ML model.

        Args:
            model_path: Path to model file

        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model format is unsupported
        """
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")

        file_size = os.path.getsize(model_path)
        if file_size == 0:
            logger.error(f"Model file is empty: {model_path}")
            raise ValueError(f"Model file is empty: {model_path}")

        logger.info(f"Loading model from {model_path} ({file_size / 1024 / 1024:.2f} MB)")

        try:
            # Try ONNX first
            if model_path.endswith('.onnx'):
                try:
                    import onnxruntime as ort
                except ImportError:
                    raise ImportError(
                        "onnxruntime is required to load ONNX models. "
                        "Install with: pip install onnxruntime"
                    )

                # Set session options for better performance
                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

                self.model = ort.InferenceSession(model_path, sess_options)
                self.model_type = 'onnx'

                # Validate model inputs/outputs
                input_info = self.model.get_inputs()[0]
                output_info = self.model.get_outputs()[0]
                logger.info(
                    f"Loaded ONNX model - Input: {input_info.name} {input_info.shape}, "
                    f"Output: {output_info.name} {output_info.shape}"
                )

            # Try TorchScript
            elif model_path.endswith('.pt') or model_path.endswith('.pth'):
                try:
                    import torch
                except ImportError:
                    raise ImportError(
                        "PyTorch is required to load TorchScript models. "
                        "Install with: pip install torch"
                    )

                self.model = torch.jit.load(model_path, map_location='cpu')
                self.model.eval()
                self.model_type = 'torch'
                logger.info(f"Loaded TorchScript model from {model_path}")

            else:
                supported_formats = ['.onnx', '.pt', '.pth']
                raise ValueError(
                    f"Unsupported model format: {model_path}. "
                    f"Supported formats: {supported_formats}"
                )

        except ImportError as e:
            logger.error(f"Missing dependencies for model loading: {e}")
            self.model = None
            raise
        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self.model = None
            raise RuntimeError(f"Failed to load model from {model_path}: {e}")

    def generate_gesture_sequence(self, audio_features: Dict[str, np.ndarray],
                                  fps: int = 24,
                                  bone_names: Optional[List[str]] = None) -> Dict[str, Dict[int, Dict]]:
        """
        Generate frame-by-frame bone rotations and positions.

        Args:
            audio_features: Dictionary of audio features from AudioProcessor
            fps: Target frames per second (must be > 0, typically 24, 30, or 60)
            bone_names: List of bone names to animate (None = use defaults)

        Returns:
            Dictionary in format:
            {
                'bone_name': {
                    frame_number: {
                        'rotation_quaternion': (w, x, y, z),
                        'location': (x, y, z)  # optional
                    }
                }
            }

        Raises:
            ValueError: If audio_features is invalid or fps is invalid
            TypeError: If inputs have wrong types
        """
        # Validate inputs
        if not isinstance(audio_features, dict):
            raise TypeError(f"audio_features must be a dict, got {type(audio_features)}")

        if not isinstance(fps, int) or fps <= 0:
            raise ValueError(f"fps must be a positive integer, got {fps}")

        if fps > 240:
            logger.warning(f"Very high FPS requested ({fps}). This may cause performance issues.")

        if bone_names is not None:
            if not isinstance(bone_names, list):
                raise TypeError(f"bone_names must be a list or None, got {type(bone_names)}")
            if len(bone_names) == 0:
                raise ValueError("bone_names list cannot be empty")
            if len(bone_names) > 100:
                logger.warning(f"Very large number of bones requested ({len(bone_names)})")

        # Validate audio features
        if 'rms' not in audio_features or 'times' not in audio_features:
            raise ValueError("audio_features must contain at least 'rms' and 'times' keys")

        rms = audio_features.get('rms', np.array([]))
        times = audio_features.get('times', np.array([]))

        if len(rms) == 0 or len(times) == 0:
            raise ValueError("audio_features 'rms' and 'times' cannot be empty")

        if len(rms) != len(times):
            raise ValueError(
                f"audio_features 'rms' and 'times' must have same length, "
                f"got rms={len(rms)}, times={len(times)}"
            )

        duration = times[-1] - times[0] if len(times) > 1 else 0
        logger.debug(
            f"Generating gestures: {len(rms)} feature frames, "
            f"{duration:.2f}s duration, {fps} FPS target"
        )

        # Choose generation method
        if self.model is not None:
            logger.info("Using ML model for gesture generation")
            return self._ml_generation(audio_features, fps, bone_names)
        else:
            logger.info("Using rule-based generation")
            return self.rule_based_generation(audio_features, fps, bone_names)

    def _ml_generation(self, audio_features: Dict[str, np.ndarray],
                       fps: int,
                       bone_names: Optional[List[str]]) -> Dict[str, Dict[int, Dict]]:
        """
        Generate gestures using ML model.

        Args:
            audio_features: Audio features
            fps: Frames per second
            bone_names: Bone names to animate

        Returns:
            Gesture dictionary
        """
        if self.model is None:
            logger.error("ML model not loaded")
            return self.rule_based_generation(audio_features, fps, bone_names)

        try:
            # Use default bone names if not specified
            if bone_names is None:
                bone_names = ['head', 'neck', 'spine', 'spine.001', 'spine.002',
                             'shoulder.L', 'shoulder.R', 'upper_arm.L', 'upper_arm.R',
                             'forearm.L', 'forearm.R', 'hand.L', 'hand.R']

            # Prepare input features
            input_features = self._prepare_ml_input(audio_features)

            # Run inference based on model type
            if self.model_type == 'onnx':
                predictions = self._run_onnx_inference(input_features)
            elif self.model_type == 'torch':
                predictions = self._run_torch_inference(input_features)
            else:
                logger.error(f"Unknown model type: {self.model_type}")
                return self.rule_based_generation(audio_features, fps, bone_names)

            # Convert predictions to gesture dictionary
            gestures = self._predictions_to_gestures(predictions, bone_names)

            # Apply smoothing for better quality
            gestures = self.smooth_gesture_sequence(gestures, window_size=5)

            logger.info(f"Generated {len(predictions)} frames using ML model for {len(bone_names)} bones")
            return gestures

        except Exception as e:
            logger.error(f"ML inference failed: {e}", exc_info=True)
            logger.warning("Falling back to rule-based generation")
            return self.rule_based_generation(audio_features, fps, bone_names)

    def _prepare_ml_input(self, audio_features: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Prepare audio features for ML model input.

        Combines all audio features into a single array compatible with model input.

        Args:
            audio_features: Dictionary of audio features

        Returns:
            Feature array of shape (time_steps, feature_dim)
        """
        # Extract features - handle both formats
        rms = audio_features.get('rms', np.array([]))
        zcr = audio_features.get('zcr', np.array([]))
        spectral_centroid = audio_features.get('spectral_centroid', np.array([]))
        mfcc = audio_features.get('mfcc', np.array([]))
        onset_strength = audio_features.get('onset_strength', np.array([]))

        if len(rms) == 0:
            raise ValueError("No RMS features found in audio_features")

        num_frames = len(rms)

        # Build feature list
        feature_list = []

        # Add scalar features
        if len(rms) > 0:
            feature_list.append(rms.reshape(-1, 1))
        if len(zcr) > 0 and len(zcr) == num_frames:
            feature_list.append(zcr.reshape(-1, 1))
        if len(spectral_centroid) > 0 and len(spectral_centroid) == num_frames:
            feature_list.append(spectral_centroid.reshape(-1, 1))
        if len(onset_strength) > 0 and len(onset_strength) == num_frames:
            feature_list.append(onset_strength.reshape(-1, 1))

        # Add MFCC features (transpose if needed)
        if len(mfcc) > 0:
            if mfcc.ndim == 2:
                # MFCC is (n_mfcc, time_steps), transpose to (time_steps, n_mfcc)
                if mfcc.shape[1] == num_frames:
                    feature_list.append(mfcc.T)
                elif mfcc.shape[0] == num_frames:
                    feature_list.append(mfcc)

        # Concatenate all features
        if len(feature_list) == 0:
            raise ValueError("No valid features to prepare for ML input")

        features = np.concatenate(feature_list, axis=1)

        # Normalize features (z-score normalization)
        features = self._normalize_features(features)

        logger.debug(f"Prepared ML input: shape {features.shape}")
        return features

    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Normalize features using z-score normalization.

        Args:
            features: Feature array (time_steps, feature_dim)

        Returns:
            Normalized features
        """
        mean = np.mean(features, axis=0, keepdims=True)
        std = np.std(features, axis=0, keepdims=True) + 1e-8
        return (features - mean) / std

    def _run_onnx_inference(self, input_features: np.ndarray) -> np.ndarray:
        """
        Run inference using ONNX model.

        Args:
            input_features: Input feature array (time_steps, feature_dim)

        Returns:
            Predictions array (time_steps, output_dim)
        """
        import onnxruntime as ort

        # ONNX expects batch dimension: (batch, time_steps, features)
        input_batch = input_features[np.newaxis, :, :].astype(np.float32)

        # Get input name from model
        input_name = self.model.get_inputs()[0].name

        # Run inference
        outputs = self.model.run(None, {input_name: input_batch})

        # Remove batch dimension
        predictions = outputs[0][0]  # (time_steps, output_dim)

        logger.debug(f"ONNX inference output shape: {predictions.shape}")
        return predictions

    def _run_torch_inference(self, input_features: np.ndarray) -> np.ndarray:
        """
        Run inference using TorchScript model.

        Args:
            input_features: Input feature array (time_steps, feature_dim)

        Returns:
            Predictions array (time_steps, output_dim)
        """
        import torch

        # Convert to torch tensor with batch dimension
        input_tensor = torch.from_numpy(input_features).float().unsqueeze(0)

        # Run inference
        with torch.no_grad():
            output_tensor = self.model(input_tensor)

        # Convert back to numpy and remove batch dimension
        predictions = output_tensor.squeeze(0).cpu().numpy()

        logger.debug(f"Torch inference output shape: {predictions.shape}")
        return predictions

    def _predictions_to_gestures(self, predictions: np.ndarray,
                                 bone_names: List[str]) -> Dict[str, Dict[int, Dict]]:
        """
        Convert model predictions to gesture dictionary format.

        Args:
            predictions: Model output (time_steps, output_dim)
                        Expected format: flat array of [bone0_quat(4) + bone0_loc(3), bone1_quat(4) + bone1_loc(3), ...]
            bone_names: List of bone names

        Returns:
            Gesture dictionary in standard format
        """
        num_frames = predictions.shape[0]
        output_dim = predictions.shape[1]

        # Calculate expected dimension (7 DOF per bone: 4 quaternion + 3 location)
        expected_dim = len(bone_names) * 7

        if output_dim != expected_dim:
            logger.warning(f"Output dimension mismatch: expected {expected_dim}, got {output_dim}")
            # Adjust bone_names or pad predictions as needed
            if output_dim < expected_dim:
                # Fewer outputs than expected - use fewer bones
                num_bones = output_dim // 7
                bone_names = bone_names[:num_bones]
            elif output_dim > expected_dim:
                # More outputs than expected - pad predictions
                padding = np.zeros((num_frames, expected_dim - output_dim))
                predictions = np.concatenate([predictions, padding], axis=1)

        # Build gesture dictionary
        gestures = {bone: {} for bone in bone_names}

        for frame_idx in range(num_frames):
            frame_data = predictions[frame_idx]

            for bone_idx, bone_name in enumerate(bone_names):
                # Extract bone data (7 values: 4 quaternion + 3 location)
                start_idx = bone_idx * 7

                # Get quaternion (w, x, y, z)
                quat = frame_data[start_idx:start_idx+4]

                # Normalize quaternion
                quat_norm = np.linalg.norm(quat)
                if quat_norm > 0:
                    quat = quat / quat_norm
                else:
                    quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

                # Get location (x, y, z)
                location = frame_data[start_idx+4:start_idx+7]

                gestures[bone_name][frame_idx] = {
                    'rotation_quaternion': tuple(quat),
                    'location': tuple(location)
                }

        return gestures

    def rule_based_generation(self, audio_features: Dict[str, np.ndarray],
                             fps: int = 24,
                             bone_names: Optional[List[str]] = None) -> Dict[str, Dict[int, Dict]]:
        """
        Generate gestures using rule-based system.

        Maps audio features to bone movements:
        - Energy -> gesture intensity
        - Pauses -> rest poses
        - Emphasis -> larger gestures
        - Continuous idle motion (breathing, subtle movements)

        Args:
            audio_features: Audio features from AudioProcessor
            fps: Target frames per second
            bone_names: Bone names to animate (uses defaults if None)

        Returns:
            Gesture dictionary
        """
        if bone_names is None:
            bone_names = ['head', 'neck', 'spine', 'spine.001', 'spine.002',
                         'shoulder.L', 'shoulder.R', 'upper_arm.L', 'upper_arm.R',
                         'forearm.L', 'forearm.R', 'hand.L', 'hand.R']

        # Extract key features
        rms = audio_features.get('rms', np.array([]))
        times = audio_features.get('times', np.array([]))

        if len(rms) == 0 or len(times) == 0:
            logger.error("No audio features available")
            return {}

        num_frames = len(times)
        gestures = {bone: {} for bone in bone_names}

        # Normalize energy
        rms_norm = rms / (np.max(rms) + 1e-6)

        # Generate per-frame animations
        for frame_idx in range(num_frames):
            time = times[frame_idx]
            energy = rms_norm[frame_idx]

            # Generate bone transforms for this frame
            self._generate_frame_poses(gestures, frame_idx, time, energy, bone_names)

        logger.info(f"Generated {num_frames} frames of rule-based gestures for {len(bone_names)} bones")
        return gestures

    def _generate_frame_poses(self, gestures: Dict, frame_idx: int,
                             time: float, energy: float, bone_names: List[str]):
        """
        Generate bone poses for a single frame.

        Args:
            gestures: Gesture dictionary to populate
            frame_idx: Current frame index
            time: Current time in seconds
            energy: Normalized audio energy (0-1)
            bone_names: List of bones to animate
        """
        config = self.config
        intensity = config['gesture_intensity']
        breathing_rate = config['breathing_rate']
        idle_scale = config['idle_motion_scale']

        # Breathing/idle motion (sine wave)
        breathing_phase = np.sin(2 * np.pi * breathing_rate * time)
        idle_phase = np.sin(2 * np.pi * breathing_rate * 0.5 * time)  # Slower for variety

        # Energy-based gesture intensity
        gesture_intensity = energy * intensity

        for bone in bone_names:
            # Base idle rotation (quaternion: w, x, y, z)
            rotation = self._get_idle_rotation(bone, breathing_phase, idle_phase, idle_scale)

            # Add energy-based gestures
            if gesture_intensity > 0.3:
                rotation = self._add_gesture_rotation(bone, rotation, gesture_intensity, time)

            gestures[bone][frame_idx] = {
                'rotation_quaternion': rotation,
                'location': self._get_bone_location(bone, breathing_phase, idle_scale)
            }

    def _get_idle_rotation(self, bone_name: str, breathing_phase: float,
                          idle_phase: float, scale: float) -> Tuple[float, float, float, float]:
        """
        Get idle rotation for a bone (subtle breathing and idle motion).

        Args:
            bone_name: Name of the bone
            breathing_phase: Breathing cycle phase (-1 to 1)
            idle_phase: Idle motion phase (-1 to 1)
            scale: Idle motion scale

        Returns:
            Quaternion (w, x, y, z)
        """
        # Identity quaternion
        w, x, y, z = 1.0, 0.0, 0.0, 0.0

        # Subtle rotations based on bone type
        if 'spine' in bone_name:
            # Spine rotates slightly with breathing
            angle = breathing_phase * 0.02 * scale
            x = np.sin(angle / 2)
            w = np.cos(angle / 2)

        elif 'head' in bone_name:
            # Head has gentle nodding motion
            angle_x = idle_phase * 0.03 * scale
            angle_z = np.sin(idle_phase * 0.7) * 0.02 * scale
            # Simple euler to quaternion (approximate)
            w = np.cos(angle_x / 2) * np.cos(angle_z / 2)
            x = np.sin(angle_x / 2) * np.cos(angle_z / 2)
            z = np.cos(angle_x / 2) * np.sin(angle_z / 2)

        elif 'neck' in bone_name:
            # Neck follows head slightly
            angle = idle_phase * 0.02 * scale
            x = np.sin(angle / 2)
            w = np.cos(angle / 2)

        elif 'shoulder' in bone_name:
            # Shoulders move with breathing
            angle = breathing_phase * 0.015 * scale
            y = np.sin(angle / 2)
            w = np.cos(angle / 2)

        elif 'arm' in bone_name or 'hand' in bone_name:
            # Arms have minimal idle motion
            angle = idle_phase * 0.01 * scale
            z = np.sin(angle / 2)
            w = np.cos(angle / 2)

        return (w, x, y, z)

    def _add_gesture_rotation(self, bone_name: str, base_rotation: Tuple,
                             intensity: float, time: float) -> Tuple[float, float, float, float]:
        """
        Add gesture motion to base rotation based on audio energy.

        Args:
            bone_name: Bone name
            base_rotation: Base quaternion
            intensity: Gesture intensity (0-1)
            time: Current time

        Returns:
            Modified quaternion
        """
        w, x, y, z = base_rotation

        # Different gestures for different bones
        if 'head' in bone_name:
            # Head nods and turns
            if intensity > self.config['head_nod_threshold']:
                nod_angle = intensity * 0.15
                x += np.sin(nod_angle)

        elif 'hand' in bone_name:
            # Hand gestures
            if intensity > self.config['hand_gesture_threshold']:
                gesture_angle = intensity * 0.2
                # Vary gesture based on time
                if '.L' in bone_name:
                    y += np.sin(time * 2) * gesture_angle
                else:
                    y -= np.sin(time * 2) * gesture_angle

        elif 'arm' in bone_name:
            # Arm movements
            if intensity > 0.5:
                arm_angle = (intensity - 0.5) * 0.15
                if '.L' in bone_name:
                    z += arm_angle
                else:
                    z -= arm_angle

        elif 'spine' in bone_name:
            # Spine leans slightly
            lean = intensity * 0.05
            x += lean

        # Re-normalize quaternion (approximate)
        magnitude = np.sqrt(w*w + x*x + y*y + z*z)
        if magnitude > 0:
            w, x, y, z = w/magnitude, x/magnitude, y/magnitude, z/magnitude

        return (w, x, y, z)

    def _get_bone_location(self, bone_name: str, breathing_phase: float,
                          scale: float) -> Tuple[float, float, float]:
        """
        Get bone location offset (mostly for subtle breathing motion).

        Args:
            bone_name: Bone name
            breathing_phase: Breathing phase
            scale: Motion scale

        Returns:
            Location offset (x, y, z)
        """
        # Most bones don't need location animation for sitting gestures
        x, y, z = 0.0, 0.0, 0.0

        # Spine moves slightly up/down with breathing
        if 'spine' in bone_name:
            z = breathing_phase * 0.002 * scale

        return (x, y, z)

    def smooth_gesture_sequence(self, gestures: Dict[str, Dict[int, Dict]],
                               window_size: int = 3) -> Dict[str, Dict[int, Dict]]:
        """
        Apply smoothing to gesture sequence to reduce jitter.

        Args:
            gestures: Gesture dictionary
            window_size: Smoothing window size (frames)

        Returns:
            Smoothed gesture dictionary
        """
        smoothed = {}

        for bone_name, frames in gestures.items():
            smoothed[bone_name] = {}
            frame_indices = sorted(frames.keys())

            for i, frame_idx in enumerate(frame_indices):
                # Get neighboring frames for smoothing
                start = max(0, i - window_size // 2)
                end = min(len(frame_indices), i + window_size // 2 + 1)
                neighbor_indices = frame_indices[start:end]

                # Average rotations (simple approach - proper quaternion slerp would be better)
                rotations = [frames[idx]['rotation_quaternion'] for idx in neighbor_indices]
                avg_rotation = tuple(np.mean([r[i] for r in rotations]) for i in range(4))

                # Average locations
                locations = [frames[idx]['location'] for idx in neighbor_indices]
                avg_location = tuple(np.mean([loc[i] for loc in locations]) for i in range(3))

                # Normalize quaternion
                magnitude = np.sqrt(sum(x*x for x in avg_rotation))
                if magnitude > 0:
                    avg_rotation = tuple(x / magnitude for x in avg_rotation)

                smoothed[bone_name][frame_idx] = {
                    'rotation_quaternion': avg_rotation,
                    'location': avg_location
                }

        logger.info("Applied smoothing to gesture sequence")
        return smoothed

    def export_to_json(self, gestures: Dict, filepath: str):
        """
        Export gesture sequence to JSON file.

        Args:
            gestures: Gesture dictionary
            filepath: Output file path
        """
        # Convert to serializable format
        export_data = {}
        for bone_name, frames in gestures.items():
            export_data[bone_name] = {
                str(frame_idx): {
                    'rotation_quaternion': list(data['rotation_quaternion']),
                    'location': list(data['location'])
                }
                for frame_idx, data in frames.items()
            }

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"Exported gestures to {filepath}")
