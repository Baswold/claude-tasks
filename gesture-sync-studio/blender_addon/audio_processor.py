"""
Audio processing module for gesture generation.
Extracts features from audio files for gesture synthesis.
"""

import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import librosa, but provide fallback if not available
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available. Audio processing will be limited.")


class AudioProcessor:
    """Analyzes audio files and extracts features for gesture generation."""

    def __init__(self, sr: int = 22050, frame_length: int = 2048, hop_length: int = 512,
                 enable_cache: bool = True, cache_dir: Optional[str] = None):
        """
        Initialize audio processor.

        Args:
            sr: Sample rate for audio processing
            frame_length: Length of each analysis frame
            hop_length: Number of samples between frames
            enable_cache: Enable feature caching for performance
            cache_dir: Directory for cache files (None = in-memory only)
        """
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.enable_cache = enable_cache

        # Initialize cache if enabled
        if self.enable_cache:
            try:
                from performance_cache import get_feature_cache
                self.feature_cache = get_feature_cache(cache_dir=cache_dir)
                logger.debug("Feature caching enabled")
            except ImportError:
                logger.warning("performance_cache module not available, caching disabled")
                self.enable_cache = False
                self.feature_cache = None
        else:
            self.feature_cache = None

    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return waveform and sample rate.

        Args:
            filepath: Path to audio file

        Returns:
            Tuple of (waveform, sample_rate)

        Raises:
            ImportError: If librosa is not available
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio file is invalid or corrupted
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError(
                "librosa is required for audio loading. "
                "Install with: pip install librosa"
            )

        # Validate file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        # Check file size
        file_size = os.path.getsize(filepath)
        if file_size == 0:
            raise ValueError(f"Audio file is empty: {filepath}")

        if file_size > 500 * 1024 * 1024:  # 500 MB
            logger.warning(
                f"Very large audio file ({file_size / 1024 / 1024:.1f} MB). "
                "Processing may take a long time."
            )

        # Validate file extension
        valid_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac']
        ext = os.path.splitext(filepath)[1].lower()
        if ext not in valid_extensions:
            logger.warning(
                f"Unusual audio format: {ext}. "
                f"Supported formats: {valid_extensions}"
            )

        try:
            logger.debug(f"Loading audio from {filepath} ({file_size / 1024:.1f} KB)")
            waveform, sr = librosa.load(filepath, sr=self.sr, mono=True)

            # Validate loaded audio
            if len(waveform) == 0:
                raise ValueError("Loaded audio has zero length")

            if np.isnan(waveform).any():
                raise ValueError("Loaded audio contains NaN values")

            if np.isinf(waveform).any():
                raise ValueError("Loaded audio contains infinite values")

            duration = len(waveform) / sr
            logger.info(
                f"Loaded audio: {filepath}, duration: {duration:.2f}s, "
                f"sample rate: {sr} Hz, samples: {len(waveform)}"
            )

            # Warn if audio is very quiet or very loud
            rms = np.sqrt(np.mean(waveform ** 2))
            if rms < 0.001:
                logger.warning("Audio is very quiet (low RMS). Gestures may be minimal.")
            elif rms > 0.9:
                logger.warning("Audio is very loud. Consider normalizing to prevent clipping.")

            return waveform, sr

        except Exception as e:
            logger.error(f"Failed to load audio file {filepath}: {e}", exc_info=True)
            raise RuntimeError(f"Failed to load audio file: {e}")

    def extract_features(self, waveform: np.ndarray, sr: int,
                        filepath: Optional[str] = None) -> Dict[str, np.ndarray]:
        """
        Extract relevant features from audio waveform.

        Features extracted:
        - RMS energy (volume/emphasis)
        - Zero-crossing rate (speech activity)
        - Spectral centroid (tone/pitch changes)
        - MFCC features (speech characteristics)
        - Onset strength (word/phrase boundaries)

        Args:
            waveform: Audio waveform as numpy array
            sr: Sample rate
            filepath: Optional file path for caching

        Returns:
            Dictionary with frame-by-frame features
        """
        # Check cache if enabled and filepath provided
        if self.enable_cache and filepath and self.feature_cache:
            cache_params = {
                'sr': self.sr,
                'frame_length': self.frame_length,
                'hop_length': self.hop_length
            }
            cached = self.feature_cache.get(filepath, cache_params)
            if cached is not None:
                logger.debug(f"Using cached features for {filepath}")
                return cached

        if not LIBROSA_AVAILABLE:
            features = self._extract_features_fallback(waveform, sr)
        else:
            features = self._extract_features_librosa(waveform, sr)

        # Cache the result if enabled
        if self.enable_cache and filepath and self.feature_cache:
            self.feature_cache.put(filepath, cache_params, features)

        return features

    def _extract_features_librosa(self, waveform: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Extract features using librosa (internal method).

        Args:
            waveform: Audio waveform
            sr: Sample rate

        Returns:
            Feature dictionary
        """
        features = {}

        try:
            # RMS Energy - indicates volume and emphasis
            features['rms'] = librosa.feature.rms(
                y=waveform,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )[0]

            # Zero-crossing rate - speech activity indicator
            features['zcr'] = librosa.feature.zero_crossing_rate(
                waveform,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )[0]

            # Spectral centroid - brightness of sound
            features['spectral_centroid'] = librosa.feature.spectral_centroid(
                y=waveform,
                sr=sr,
                n_fft=self.frame_length,
                hop_length=self.hop_length
            )[0]

            # MFCCs - speech characteristics
            features['mfcc'] = librosa.feature.mfcc(
                y=waveform,
                sr=sr,
                n_mfcc=13,
                n_fft=self.frame_length,
                hop_length=self.hop_length
            )

            # Onset strength - phrase/word boundaries
            features['onset_strength'] = librosa.onset.onset_strength(
                y=waveform,
                sr=sr,
                hop_length=self.hop_length
            )

            # Time axis for all features
            features['times'] = librosa.frames_to_time(
                np.arange(len(features['rms'])),
                sr=sr,
                hop_length=self.hop_length
            )

            logger.info(f"Extracted features: {len(features['rms'])} frames")

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            raise

        return features

    def _extract_features_fallback(self, waveform: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """
        Fallback feature extraction using only numpy when librosa is not available.

        Args:
            waveform: Audio waveform
            sr: Sample rate

        Returns:
            Basic features dictionary
        """
        num_frames = (len(waveform) - self.frame_length) // self.hop_length + 1
        features = {}

        # Simple RMS energy
        rms = []
        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            frame = waveform[start:end]
            rms.append(np.sqrt(np.mean(frame**2)))
        features['rms'] = np.array(rms)

        # Simple zero-crossing rate
        zcr = []
        for i in range(num_frames):
            start = i * self.hop_length
            end = start + self.frame_length
            frame = waveform[start:end]
            crossings = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
            zcr.append(crossings / len(frame))
        features['zcr'] = np.array(zcr)

        # Time axis
        features['times'] = np.arange(num_frames) * self.hop_length / sr

        logger.warning("Using fallback feature extraction (librosa not available)")

        return features

    def detect_speech_segments(self, features: Dict[str, np.ndarray],
                               energy_threshold: float = 0.02,
                               min_duration: float = 0.1) -> List[Tuple[float, float]]:
        """
        Identify speech vs silence segments based on energy.

        Args:
            features: Feature dictionary from extract_features()
            energy_threshold: RMS threshold for speech detection
            min_duration: Minimum duration for a speech segment (seconds)

        Returns:
            List of (start_time, end_time) tuples for speech segments
        """
        rms = features['rms']
        times = features['times']

        # Normalize RMS
        if np.max(rms) > 0:
            rms_norm = rms / np.max(rms)
        else:
            rms_norm = rms

        # Detect speech frames
        is_speech = rms_norm > energy_threshold

        # Find continuous segments
        segments = []
        in_segment = False
        start_time = 0

        for i, (is_active, time) in enumerate(zip(is_speech, times)):
            if is_active and not in_segment:
                # Start new segment
                start_time = time
                in_segment = True
            elif not is_active and in_segment:
                # End segment
                duration = time - start_time
                if duration >= min_duration:
                    segments.append((start_time, time))
                in_segment = False

        # Handle last segment
        if in_segment and (times[-1] - start_time) >= min_duration:
            segments.append((start_time, times[-1]))

        logger.info(f"Detected {len(segments)} speech segments")
        return segments

    def detect_emphasis_points(self, features: Dict[str, np.ndarray],
                               percentile: float = 85.0) -> List[float]:
        """
        Find points of emphasis/stress in speech based on energy peaks.

        Args:
            features: Feature dictionary from extract_features()
            percentile: Percentile threshold for emphasis detection

        Returns:
            List of timestamps where emphasis occurs
        """
        rms = features['rms']
        times = features['times']

        # Find peaks above threshold
        threshold = np.percentile(rms, percentile)

        # Simple peak detection: local maxima above threshold
        emphasis_points = []
        for i in range(1, len(rms) - 1):
            if rms[i] > threshold and rms[i] > rms[i-1] and rms[i] > rms[i+1]:
                emphasis_points.append(times[i])

        logger.info(f"Detected {len(emphasis_points)} emphasis points")
        return emphasis_points

    def get_frame_rate(self, features: Dict[str, np.ndarray]) -> float:
        """
        Calculate the frame rate of extracted features.

        Args:
            features: Feature dictionary

        Returns:
            Frames per second
        """
        if 'times' in features and len(features['times']) > 1:
            time_per_frame = features['times'][1] - features['times'][0]
            return 1.0 / time_per_frame if time_per_frame > 0 else 0
        return 0

    def interpolate_to_fps(self, features: Dict[str, np.ndarray],
                          target_fps: int = 24) -> Dict[str, np.ndarray]:
        """
        Interpolate features to match target animation FPS.

        Args:
            features: Original feature dictionary
            target_fps: Target frames per second

        Returns:
            Interpolated feature dictionary
        """
        if 'times' not in features or len(features['times']) == 0:
            return features

        # Create new time axis
        duration = features['times'][-1]
        num_frames = int(duration * target_fps)
        new_times = np.linspace(0, duration, num_frames)

        interpolated = {'times': new_times}

        # Interpolate each feature
        for key, values in features.items():
            if key == 'times':
                continue

            if values.ndim == 1:
                # 1D feature
                interpolated[key] = np.interp(new_times, features['times'], values)
            elif values.ndim == 2:
                # 2D feature (like MFCC)
                interpolated[key] = np.zeros((values.shape[0], num_frames))
                for i in range(values.shape[0]):
                    interpolated[key][i] = np.interp(new_times, features['times'], values[i])

        logger.info(f"Interpolated features to {target_fps} FPS ({num_frames} frames)")
        return interpolated
