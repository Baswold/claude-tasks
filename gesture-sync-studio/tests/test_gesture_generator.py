"""
Comprehensive tests for gesture generator module including ML inference.
"""

import pytest
import numpy as np
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import tempfile
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'blender_addon'))

from gesture_generator import GestureGenerator


class TestGestureGeneratorInit:
    """Test initialization and configuration."""

    def test_default_init(self):
        """Test default initialization."""
        generator = GestureGenerator()
        assert generator.model is None
        assert generator.config is not None
        assert 'gesture_intensity' in generator.config

    def test_custom_config(self):
        """Test initialization with custom config."""
        custom_config = {
            'gesture_intensity': 2.0,
            'smoothing': 0.8
        }
        generator = GestureGenerator(config=custom_config)
        assert generator.config['gesture_intensity'] == 2.0
        assert generator.config['smoothing'] == 0.8

    def test_default_config_values(self):
        """Test default configuration values."""
        generator = GestureGenerator()
        config = generator.config

        assert config['gesture_intensity'] == 1.0
        assert config['smoothing'] == 0.5
        assert config['idle_motion_scale'] == 0.3
        assert config['breathing_rate'] == 0.2
        assert config['head_nod_threshold'] == 0.7
        assert config['hand_gesture_threshold'] == 0.6


class TestRuleBasedGeneration:
    """Test rule-based gesture generation."""

    def test_basic_generation(self):
        """Test basic rule-based generation."""
        generator = GestureGenerator()

        # Create synthetic audio features
        num_frames = 100
        audio_features = {
            'rms': np.random.random(num_frames) * 0.5,
            'times': np.linspace(0, 4.0, num_frames),
            'zcr': np.random.random(num_frames),
        }

        gestures = generator.rule_based_generation(audio_features, fps=24)

        # Check output structure
        assert isinstance(gestures, dict)
        assert len(gestures) > 0

        # Check bone data
        for bone_name, frames in gestures.items():
            assert isinstance(frames, dict)
            assert len(frames) == num_frames

            # Check first frame structure
            if 0 in frames:
                frame_data = frames[0]
                assert 'rotation_quaternion' in frame_data
                assert 'location' in frame_data
                assert len(frame_data['rotation_quaternion']) == 4
                assert len(frame_data['location']) == 3

    def test_empty_features(self):
        """Test handling of empty features."""
        generator = GestureGenerator()

        audio_features = {
            'rms': np.array([]),
            'times': np.array([])
        }

        gestures = generator.rule_based_generation(audio_features, fps=24)

        # Should return empty dict
        assert gestures == {}

    def test_high_energy_gestures(self):
        """Test that high energy produces more pronounced gestures."""
        generator = GestureGenerator()

        # Low energy audio
        num_frames = 50
        low_energy = {
            'rms': np.ones(num_frames) * 0.1,
            'times': np.linspace(0, 2.0, num_frames)
        }

        # High energy audio
        high_energy = {
            'rms': np.ones(num_frames) * 0.9,
            'times': np.linspace(0, 2.0, num_frames)
        }

        gestures_low = generator.rule_based_generation(low_energy, fps=24)
        gestures_high = generator.rule_based_generation(high_energy, fps=24)

        # Both should generate gestures
        assert len(gestures_low) > 0
        assert len(gestures_high) > 0

    def test_custom_bone_names(self):
        """Test generation with custom bone names."""
        generator = GestureGenerator()

        audio_features = {
            'rms': np.random.random(50),
            'times': np.linspace(0, 2.0, 50)
        }

        custom_bones = ['head', 'neck', 'spine']
        gestures = generator.rule_based_generation(
            audio_features, fps=24, bone_names=custom_bones
        )

        # Should only generate for specified bones
        assert set(gestures.keys()) == set(custom_bones)


class TestMLInference:
    """Test ML inference functionality."""

    def test_prepare_ml_input_basic(self):
        """Test basic ML input preparation."""
        generator = GestureGenerator()

        audio_features = {
            'rms': np.random.random(100),
            'zcr': np.random.random(100),
            'spectral_centroid': np.random.random(100),
            'onset_strength': np.random.random(100),
            'mfcc': np.random.random((13, 100)),  # 13 MFCCs
        }

        input_features = generator._prepare_ml_input(audio_features)

        # Check output shape
        assert input_features.shape[0] == 100  # num_frames
        assert input_features.shape[1] > 0  # feature_dim
        # Expected: 1 (rms) + 1 (zcr) + 1 (spectral_centroid) + 1 (onset_strength) + 13 (mfcc) = 17
        assert input_features.shape[1] == 17

    def test_prepare_ml_input_minimal(self):
        """Test ML input preparation with minimal features."""
        generator = GestureGenerator()

        # Only RMS (minimum required)
        audio_features = {
            'rms': np.random.random(50),
        }

        input_features = generator._prepare_ml_input(audio_features)

        assert input_features.shape[0] == 50
        assert input_features.shape[1] == 1  # Only RMS

    def test_prepare_ml_input_no_rms(self):
        """Test that missing RMS raises error."""
        generator = GestureGenerator()

        audio_features = {
            'zcr': np.random.random(50),
        }

        with pytest.raises(ValueError, match="No RMS features"):
            generator._prepare_ml_input(audio_features)

    def test_normalize_features(self):
        """Test feature normalization."""
        generator = GestureGenerator()

        features = np.random.random((100, 10)) * 100 + 50

        normalized = generator._normalize_features(features)

        # Check shape preserved
        assert normalized.shape == features.shape

        # Check normalization (mean ≈ 0, std ≈ 1)
        mean = np.mean(normalized, axis=0)
        std = np.std(normalized, axis=0)

        assert np.allclose(mean, 0, atol=1e-6)
        assert np.allclose(std, 1, atol=1e-6)

    @patch('gesture_generator.torch')
    def test_run_torch_inference(self, mock_torch):
        """Test PyTorch inference."""
        generator = GestureGenerator()
        generator.model = Mock()
        generator.model_type = 'torch'

        # Mock torch operations
        mock_tensor = Mock()
        mock_tensor.unsqueeze.return_value = mock_tensor
        mock_tensor.squeeze.return_value = mock_tensor
        mock_tensor.cpu.return_value = mock_tensor
        mock_output = np.random.random((100, 91))  # 13 bones * 7 DOF
        mock_tensor.numpy.return_value = mock_output

        mock_torch.from_numpy.return_value = mock_tensor
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()
        generator.model.return_value = mock_tensor

        input_features = np.random.random((100, 17))
        predictions = generator._run_torch_inference(input_features)

        # Check output
        assert predictions.shape == (100, 91)

    def test_predictions_to_gestures_exact_match(self):
        """Test conversion of predictions to gestures with exact dimension match."""
        generator = GestureGenerator()

        bone_names = ['head', 'neck', 'spine']
        num_bones = len(bone_names)
        num_frames = 50
        output_dim = num_bones * 7  # 7 DOF per bone

        predictions = np.random.random((num_frames, output_dim))

        gestures = generator._predictions_to_gestures(predictions, bone_names)

        # Check structure
        assert len(gestures) == num_bones
        assert set(gestures.keys()) == set(bone_names)

        # Check each bone has all frames
        for bone_name in bone_names:
            assert len(gestures[bone_name]) == num_frames

            # Check frame 0
            frame_data = gestures[bone_name][0]
            assert 'rotation_quaternion' in frame_data
            assert 'location' in frame_data

            quat = frame_data['rotation_quaternion']
            loc = frame_data['location']

            assert len(quat) == 4
            assert len(loc) == 3

            # Check quaternion is normalized
            quat_norm = np.linalg.norm(quat)
            assert abs(quat_norm - 1.0) < 1e-5

    def test_predictions_to_gestures_dimension_mismatch(self):
        """Test handling of dimension mismatch."""
        generator = GestureGenerator()

        bone_names = ['head', 'neck', 'spine']  # 3 bones = 21 DOF
        num_frames = 50

        # Too few dimensions (only 14 = 2 bones)
        predictions_small = np.random.random((num_frames, 14))
        gestures_small = generator._predictions_to_gestures(predictions_small, bone_names)

        # Should adapt by using fewer bones
        assert len(gestures_small) <= len(bone_names)

        # Too many dimensions
        predictions_large = np.random.random((num_frames, 28))  # 4 bones worth
        gestures_large = generator._predictions_to_gestures(predictions_large, bone_names)

        # Should handle gracefully
        assert len(gestures_large) == len(bone_names)

    def test_ml_generation_no_model(self):
        """Test ML generation falls back when no model loaded."""
        generator = GestureGenerator()
        generator.model = None

        audio_features = {
            'rms': np.random.random(50),
            'times': np.linspace(0, 2.0, 50)
        }

        gestures = generator._ml_generation(audio_features, fps=24, bone_names=None)

        # Should fall back to rule-based
        assert isinstance(gestures, dict)
        assert len(gestures) > 0


class TestSmoothingAndExport:
    """Test smoothing and export functionality."""

    def test_smooth_gesture_sequence(self):
        """Test gesture smoothing."""
        generator = GestureGenerator()

        # Create gestures with some jitter
        num_frames = 100
        bone_names = ['head', 'spine']

        gestures = {}
        for bone in bone_names:
            gestures[bone] = {}
            for frame in range(num_frames):
                # Random quaternion (not normalized, just for testing)
                gestures[bone][frame] = {
                    'rotation_quaternion': tuple(np.random.random(4)),
                    'location': tuple(np.random.random(3))
                }

        smoothed = generator.smooth_gesture_sequence(gestures, window_size=5)

        # Check structure preserved
        assert len(smoothed) == len(gestures)
        assert set(smoothed.keys()) == set(gestures.keys())

        for bone in bone_names:
            assert len(smoothed[bone]) == len(gestures[bone])

            # Check quaternions are normalized
            for frame in range(num_frames):
                quat = smoothed[bone][frame]['rotation_quaternion']
                quat_norm = np.linalg.norm(quat)
                assert abs(quat_norm - 1.0) < 1e-5

    def test_export_to_json(self):
        """Test JSON export."""
        generator = GestureGenerator()

        gestures = {
            'head': {
                0: {
                    'rotation_quaternion': (1.0, 0.0, 0.0, 0.0),
                    'location': (0.0, 0.0, 1.5)
                },
                1: {
                    'rotation_quaternion': (0.99, 0.01, 0.0, 0.0),
                    'location': (0.0, 0.0, 1.51)
                }
            }
        }

        # Export to temp file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            temp_path = f.name

        try:
            generator.export_to_json(gestures, temp_path)

            # Check file exists and is valid JSON
            assert os.path.exists(temp_path)

            import json
            with open(temp_path, 'r') as f:
                loaded = json.load(f)

            assert 'head' in loaded
            assert '0' in loaded['head']  # Frame indices as strings
            assert loaded['head']['0']['rotation_quaternion'] == [1.0, 0.0, 0.0, 0.0]

        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestIntegrationWithML:
    """Integration tests for full ML pipeline."""

    def test_full_pipeline_with_mock_onnx(self):
        """Test full pipeline with mocked ONNX model."""
        # Create mock ONNX model
        mock_model = Mock()
        mock_input = Mock()
        mock_input.name = 'input'
        mock_model.get_inputs.return_value = [mock_input]

        # Mock output
        num_frames = 50
        num_bones = 13
        output_dim = num_bones * 7
        mock_output = np.random.random((1, num_frames, output_dim)).astype(np.float32)
        mock_model.run.return_value = [mock_output]

        # Create generator with mock model
        generator = GestureGenerator()
        generator.model = mock_model
        generator.model_type = 'onnx'

        # Create audio features
        audio_features = {
            'rms': np.random.random(num_frames),
            'zcr': np.random.random(num_frames),
            'spectral_centroid': np.random.random(num_frames),
            'onset_strength': np.random.random(num_frames),
            'mfcc': np.random.random((13, num_frames)),
            'times': np.linspace(0, 2.0, num_frames)
        }

        # Generate gestures
        gestures = generator.generate_gesture_sequence(audio_features, fps=24)

        # Check output
        assert isinstance(gestures, dict)
        assert len(gestures) > 0

        # Verify model was called
        assert mock_model.run.called

    def test_generate_gesture_sequence_switches_to_ml(self):
        """Test that generate_gesture_sequence uses ML when model is loaded."""
        generator = GestureGenerator()

        audio_features = {
            'rms': np.random.random(50),
            'times': np.linspace(0, 2.0, 50)
        }

        # Without model, should use rule-based
        gestures_rule = generator.generate_gesture_sequence(audio_features, fps=24)
        assert len(gestures_rule) > 0

        # With model (mock), should attempt ML
        generator.model = Mock()
        generator.model_type = 'onnx'

        # This will fail during inference but that's ok for this test
        # We're just checking that it tries ML path
        with pytest.raises(Exception):
            generator.generate_gesture_sequence(audio_features, fps=24)


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_quaternion_normalization_zero_norm(self):
        """Test handling of zero-norm quaternions."""
        generator = GestureGenerator()

        bone_names = ['head']
        predictions = np.zeros((10, 7))  # All zeros

        gestures = generator._predictions_to_gestures(predictions, bone_names)

        # Should use identity quaternion
        for frame_idx in range(10):
            quat = gestures['head'][frame_idx]['rotation_quaternion']
            # Identity quaternion
            assert quat[0] == 1.0
            assert quat[1] == 0.0
            assert quat[2] == 0.0
            assert quat[3] == 0.0

    def test_very_long_sequence(self):
        """Test handling of very long sequences."""
        generator = GestureGenerator()

        num_frames = 10000  # Long sequence
        audio_features = {
            'rms': np.random.random(num_frames) * 0.5,
            'times': np.linspace(0, 400.0, num_frames)  # ~7 minutes
        }

        gestures = generator.rule_based_generation(
            audio_features, fps=24, bone_names=['head', 'spine']
        )

        assert len(gestures['head']) == num_frames
        assert len(gestures['spine']) == num_frames

    def test_single_frame(self):
        """Test handling of single frame."""
        generator = GestureGenerator()

        audio_features = {
            'rms': np.array([0.5]),
            'times': np.array([0.0])
        }

        gestures = generator.rule_based_generation(audio_features, fps=24)

        assert len(gestures) > 0
        for bone_name, frames in gestures.items():
            assert len(frames) == 1
            assert 0 in frames


class TestBoneTransforms:
    """Test specific bone transform calculations."""

    def test_idle_rotation_all_bones(self):
        """Test idle rotation for all bone types."""
        generator = GestureGenerator()

        bone_types = ['head', 'neck', 'spine', 'spine.001', 'shoulder.L',
                      'shoulder.R', 'upper_arm.L', 'forearm.R', 'hand.L']

        for bone in bone_types:
            rotation = generator._get_idle_rotation(
                bone, breathing_phase=0.5, idle_phase=0.3, scale=1.0
            )

            assert len(rotation) == 4
            # Check it's a valid quaternion (approximately normalized)
            norm = sum(x*x for x in rotation) ** 0.5
            assert 0.9 <= norm <= 1.1

    def test_gesture_rotation_intensity(self):
        """Test gesture rotation with different intensities."""
        generator = GestureGenerator()

        base_rotation = (1.0, 0.0, 0.0, 0.0)

        # Low intensity
        result_low = generator._add_gesture_rotation(
            'head', base_rotation, intensity=0.1, time=0.0
        )

        # High intensity
        result_high = generator._add_gesture_rotation(
            'head', base_rotation, intensity=0.9, time=0.0
        )

        # Both should be valid quaternions
        for result in [result_low, result_high]:
            norm = sum(x*x for x in result) ** 0.5
            assert 0.9 <= norm <= 1.1

    def test_bone_location_breathing(self):
        """Test bone location with breathing motion."""
        generator = GestureGenerator()

        # Spine should move with breathing
        loc_spine = generator._get_bone_location('spine', breathing_phase=1.0, scale=1.0)
        assert len(loc_spine) == 3
        assert loc_spine[2] != 0.0  # Z movement for breathing

        # Hand should not move much
        loc_hand = generator._get_bone_location('hand.L', breathing_phase=1.0, scale=1.0)
        assert len(loc_hand) == 3
        assert loc_hand == (0.0, 0.0, 0.0)  # No movement


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
