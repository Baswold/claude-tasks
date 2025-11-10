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
        """
        try:
            # Try ONNX first
            if model_path.endswith('.onnx'):
                import onnxruntime as ort
                self.model = ort.InferenceSession(model_path)
                self.model_type = 'onnx'
                logger.info(f"Loaded ONNX model from {model_path}")

            # Try TorchScript
            elif model_path.endswith('.pt') or model_path.endswith('.pth'):
                import torch
                self.model = torch.jit.load(model_path)
                self.model.eval()
                self.model_type = 'torch'
                logger.info(f"Loaded TorchScript model from {model_path}")

            else:
                logger.warning(f"Unknown model format: {model_path}")
                self.model = None

        except ImportError as e:
            logger.warning(f"Could not load model (missing dependencies): {e}")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def generate_gesture_sequence(self, audio_features: Dict[str, np.ndarray],
                                  fps: int = 24,
                                  bone_names: Optional[List[str]] = None) -> Dict[str, Dict[int, Dict]]:
        """
        Generate frame-by-frame bone rotations and positions.

        Args:
            audio_features: Dictionary of audio features from AudioProcessor
            fps: Target frames per second
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
        """
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
        # TODO: Implement ML inference
        # For now, fall back to rule-based
        logger.warning("ML generation not yet implemented, falling back to rule-based")
        return self.rule_based_generation(audio_features, fps, bone_names)

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
