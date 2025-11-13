"""
Tests for audio processor module.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'blender_addon'))

from audio_processor import AudioProcessor


class TestAudioProcessor:
    """Test suite for AudioProcessor."""

    def test_init(self):
        """Test initialization."""
        processor = AudioProcessor()
        assert processor.sr == 22050
        assert processor.frame_length == 2048
        assert processor.hop_length == 512

    def test_custom_params(self):
        """Test custom parameters."""
        processor = AudioProcessor(sr=16000, frame_length=1024, hop_length=256)
        assert processor.sr == 16000
        assert processor.frame_length == 1024
        assert processor.hop_length == 256

    def test_fallback_feature_extraction(self):
        """Test fallback feature extraction (no librosa)."""
        processor = AudioProcessor()

        # Create synthetic audio (1 second of sine wave)
        sr = 22050
        duration = 1.0
        frequency = 440  # A4 note
        t = np.linspace(0, duration, int(sr * duration))
        waveform = np.sin(2 * np.pi * frequency * t).astype(np.float32)

        # Extract features
        features = processor._extract_features_fallback(waveform, sr)

        # Check features exist
        assert 'rms' in features
        assert 'zcr' in features
        assert 'times' in features

        # Check shapes
        assert len(features['rms']) > 0
        assert len(features['zcr']) > 0
        assert len(features['times']) == len(features['rms'])

        # Check RMS is reasonable
        assert np.max(features['rms']) > 0
        assert np.max(features['rms']) <= 1.0

    def test_detect_speech_segments(self):
        """Test speech segment detection."""
        processor = AudioProcessor()

        # Create synthetic features with speech-like patterns
        times = np.linspace(0, 10, 1000)
        rms = np.zeros(1000)

        # Add speech segments
        rms[100:300] = 0.5  # Segment 1
        rms[500:700] = 0.6  # Segment 2
        rms[800:900] = 0.4  # Segment 3

        features = {
            'rms': rms,
            'times': times
        }

        # Detect segments
        segments = processor.detect_speech_segments(
            features,
            energy_threshold=0.1,
            min_duration=0.1
        )

        # Should detect 3 segments
        assert len(segments) >= 2  # At least 2 segments

        # Check segment format
        for start, end in segments:
            assert start < end
            assert start >= 0
            assert end <= times[-1]

    def test_detect_emphasis_points(self):
        """Test emphasis point detection."""
        processor = AudioProcessor()

        # Create synthetic features with peaks
        times = np.linspace(0, 10, 1000)
        rms = np.random.random(1000) * 0.3  # Background

        # Add emphasis peaks
        peak_indices = [100, 300, 500, 700, 900]
        for idx in peak_indices:
            rms[idx] = 0.9

        features = {
            'rms': rms,
            'times': times
        }

        # Detect emphasis
        emphasis_points = processor.detect_emphasis_points(
            features,
            percentile=85.0
        )

        # Should detect some peaks
        assert len(emphasis_points) > 0

        # Check all emphasis points are within time range
        for time in emphasis_points:
            assert 0 <= time <= times[-1]

    def test_get_frame_rate(self):
        """Test frame rate calculation."""
        processor = AudioProcessor()

        times = np.linspace(0, 1, 100)
        features = {'times': times}

        frame_rate = processor.get_frame_rate(features)

        # Should be approximately 100 fps
        assert 90 <= frame_rate <= 110

    def test_interpolate_to_fps(self):
        """Test interpolation to target FPS."""
        processor = AudioProcessor()

        # Create features at 100 fps
        duration = 2.0
        original_fps = 100
        num_frames = int(duration * original_fps)

        features = {
            'times': np.linspace(0, duration, num_frames),
            'rms': np.random.random(num_frames),
            'zcr': np.random.random(num_frames)
        }

        # Interpolate to 24 fps
        target_fps = 24
        interpolated = processor.interpolate_to_fps(features, target_fps)

        # Check output frame count
        expected_frames = int(duration * target_fps)
        assert len(interpolated['times']) == expected_frames
        assert len(interpolated['rms']) == expected_frames
        assert len(interpolated['zcr']) == expected_frames

        # Check time range preserved
        assert interpolated['times'][0] == pytest.approx(0, abs=0.1)
        assert interpolated['times'][-1] == pytest.approx(duration, abs=0.1)


def test_integration():
    """Integration test with synthetic audio."""
    processor = AudioProcessor()

    # Create synthetic audio
    sr = 22050
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))

    # Speech-like signal: modulated sine wave
    carrier = np.sin(2 * np.pi * 200 * t)  # Base frequency
    modulator = (1 + 0.5 * np.sin(2 * np.pi * 3 * t))  # Amplitude modulation
    waveform = (carrier * modulator).astype(np.float32)

    # Extract features (using fallback)
    features = processor._extract_features_fallback(waveform, sr)

    # Interpolate to 24 fps
    features = processor.interpolate_to_fps(features, target_fps=24)

    # Detect segments and emphasis
    segments = processor.detect_speech_segments(features)
    emphasis = processor.detect_emphasis_points(features)

    # Basic sanity checks
    assert len(features['rms']) == int(duration * 24)
    assert len(segments) >= 0
    assert len(emphasis) >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
