"""
Audio preprocessing utilities for improved gesture generation.

Provides:
- Noise reduction
- Normalization
- Voice activity detection
- Audio enhancement
- Format conversion
- Chunk/segment extraction
"""

import numpy as np
from typing import Tuple, List, Optional, Dict
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    logger.warning("librosa or soundfile not available. Audio preprocessing limited.")


class AudioPreprocessor:
    """
    Advanced audio preprocessing for optimal gesture generation.
    """

    def __init__(self, target_sr: int = 22050):
        """
        Initialize preprocessor.

        Args:
            target_sr: Target sample rate
        """
        self.target_sr = target_sr

    def normalize_audio(
        self,
        waveform: np.ndarray,
        target_db: float = -20.0,
        method: str = 'peak'
    ) -> np.ndarray:
        """
        Normalize audio to target level.

        Args:
            waveform: Audio waveform
            target_db: Target level in dB
            method: Normalization method ('peak', 'rms', 'lufs')

        Returns:
            Normalized waveform
        """
        if len(waveform) == 0:
            return waveform

        if method == 'peak':
            # Peak normalization
            peak = np.abs(waveform).max()
            if peak > 0:
                target_amp = 10 ** (target_db / 20.0)
                waveform = waveform * (target_amp / peak)

        elif method == 'rms':
            # RMS normalization
            rms = np.sqrt(np.mean(waveform ** 2))
            if rms > 0:
                target_rms = 10 ** (target_db / 20.0)
                waveform = waveform * (target_rms / rms)

        elif method == 'lufs':
            # Simplified LUFS-like normalization
            # This is a basic approximation
            rms = np.sqrt(np.mean(waveform ** 2))
            if rms > 0:
                # LUFS is roughly RMS but with frequency weighting
                # For simplicity, use RMS-based approach
                current_lufs = 20 * np.log10(rms) - 0.691
                target_lufs = target_db
                gain_db = target_lufs - current_lufs
                gain = 10 ** (gain_db / 20.0)
                waveform = waveform * gain

        # Clip to prevent distortion
        waveform = np.clip(waveform, -1.0, 1.0)

        logger.debug(f"Normalized audio using {method} method to {target_db} dB")
        return waveform

    def reduce_noise(
        self,
        waveform: np.ndarray,
        sr: int,
        noise_floor_db: float = -40.0
    ) -> np.ndarray:
        """
        Simple noise reduction using spectral gating.

        Args:
            waveform: Audio waveform
            sr: Sample rate
            noise_floor_db: Noise floor threshold in dB

        Returns:
            Noise-reduced waveform
        """
        if not AUDIO_LIBS_AVAILABLE:
            logger.warning("librosa not available, skipping noise reduction")
            return waveform

        # Compute STFT
        D = librosa.stft(waveform)
        magnitude, phase = librosa.magphase(D)

        # Convert to dB
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)

        # Apply spectral gate
        mask = magnitude_db > noise_floor_db
        magnitude_gated = magnitude * mask

        # Reconstruct
        D_gated = magnitude_gated * phase
        waveform_clean = librosa.istft(D_gated, length=len(waveform))

        logger.debug(f"Applied noise reduction with {noise_floor_db} dB threshold")
        return waveform_clean

    def apply_high_pass_filter(
        self,
        waveform: np.ndarray,
        sr: int,
        cutoff_freq: float = 80.0
    ) -> np.ndarray:
        """
        Apply high-pass filter to remove low-frequency noise.

        Args:
            waveform: Audio waveform
            sr: Sample rate
            cutoff_freq: Cutoff frequency in Hz

        Returns:
            Filtered waveform
        """
        try:
            from scipy.signal import butter, filtfilt
        except ImportError:
            logger.warning("scipy not available, skipping filtering")
            return waveform

        # Design Butterworth high-pass filter
        nyquist = sr / 2
        normalized_cutoff = cutoff_freq / nyquist
        b, a = butter(4, normalized_cutoff, btype='high')

        # Apply filter
        filtered = filtfilt(b, a, waveform)

        logger.debug(f"Applied high-pass filter at {cutoff_freq} Hz")
        return filtered

    def remove_silence(
        self,
        waveform: np.ndarray,
        sr: int,
        threshold_db: float = -40.0,
        min_silence_duration: float = 0.3
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Remove silence from audio.

        Args:
            waveform: Audio waveform
            sr: Sample rate
            threshold_db: Silence threshold in dB
            min_silence_duration: Minimum silence duration to remove (seconds)

        Returns:
            Tuple of (trimmed waveform, list of removed segments)
        """
        if not AUDIO_LIBS_AVAILABLE:
            logger.warning("librosa not available, skipping silence removal")
            return waveform, []

        # Detect non-silent intervals
        intervals = librosa.effects.split(
            waveform,
            top_db=-threshold_db,
            frame_length=2048,
            hop_length=512
        )

        # Filter out very short non-silent intervals
        min_samples = int(min_silence_duration * sr)

        # Concatenate non-silent segments
        segments = []
        removed_segments = []
        last_end = 0

        for start, end in intervals:
            if end - start >= min_samples:
                # Record removed silence
                if start > last_end:
                    removed_segments.append((last_end / sr, start / sr))

                segments.append(waveform[start:end])
                last_end = end

        # Final silence
        if last_end < len(waveform):
            removed_segments.append((last_end / sr, len(waveform) / sr))

        if segments:
            trimmed = np.concatenate(segments)
        else:
            trimmed = waveform

        logger.info(f"Removed {len(removed_segments)} silence segments")
        return trimmed, removed_segments

    def detect_voice_activity(
        self,
        waveform: np.ndarray,
        sr: int,
        frame_length: int = 2048,
        hop_length: int = 512,
        energy_threshold: float = 0.02
    ) -> np.ndarray:
        """
        Detect voice activity (VAD).

        Args:
            waveform: Audio waveform
            sr: Sample rate
            frame_length: Frame length for analysis
            hop_length: Hop length
            energy_threshold: Energy threshold for voice detection

        Returns:
            Boolean array indicating voice activity per frame
        """
        if not AUDIO_LIBS_AVAILABLE:
            logger.warning("librosa not available for VAD")
            return np.ones(len(waveform) // hop_length, dtype=bool)

        # Compute RMS energy
        rms = librosa.feature.rms(
            y=waveform,
            frame_length=frame_length,
            hop_length=hop_length
        )[0]

        # Normalize
        if np.max(rms) > 0:
            rms_norm = rms / np.max(rms)
        else:
            rms_norm = rms

        # Threshold
        is_voice = rms_norm > energy_threshold

        logger.debug(f"VAD detected {np.sum(is_voice)} voice frames out of {len(is_voice)}")
        return is_voice

    def enhance_speech(
        self,
        waveform: np.ndarray,
        sr: int,
        boost_freq_range: Tuple[float, float] = (300, 3400)
    ) -> np.ndarray:
        """
        Enhance speech frequencies for clearer gesture detection.

        Args:
            waveform: Audio waveform
            sr: Sample rate
            boost_freq_range: Frequency range to boost (Hz)

        Returns:
            Enhanced waveform
        """
        try:
            from scipy.signal import butter, filtfilt
        except ImportError:
            logger.warning("scipy not available, skipping enhancement")
            return waveform

        # Design band-pass filter for speech range
        nyquist = sr / 2
        low = boost_freq_range[0] / nyquist
        high = boost_freq_range[1] / nyquist

        b, a = butter(4, [low, high], btype='band')

        # Extract speech band
        speech_band = filtfilt(b, a, waveform)

        # Boost by adding back to original
        enhanced = waveform + 0.3 * speech_band

        # Normalize to prevent clipping
        peak = np.abs(enhanced).max()
        if peak > 1.0:
            enhanced = enhanced / peak

        logger.debug(f"Enhanced speech frequencies {boost_freq_range[0]}-{boost_freq_range[1]} Hz")
        return enhanced

    def extract_chunk(
        self,
        waveform: np.ndarray,
        sr: int,
        start_time: float,
        duration: float
    ) -> np.ndarray:
        """
        Extract a chunk of audio.

        Args:
            waveform: Audio waveform
            sr: Sample rate
            start_time: Start time in seconds
            duration: Duration in seconds

        Returns:
            Audio chunk
        """
        start_sample = int(start_time * sr)
        end_sample = int((start_time + duration) * sr)

        # Bounds checking
        start_sample = max(0, start_sample)
        end_sample = min(len(waveform), end_sample)

        chunk = waveform[start_sample:end_sample]

        logger.debug(f"Extracted {duration}s chunk from {start_time}s")
        return chunk

    def resample(
        self,
        waveform: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate.

        Args:
            waveform: Audio waveform
            orig_sr: Original sample rate
            target_sr: Target sample rate

        Returns:
            Resampled waveform
        """
        if orig_sr == target_sr:
            return waveform

        if not AUDIO_LIBS_AVAILABLE:
            logger.warning("librosa not available for resampling")
            return waveform

        resampled = librosa.resample(
            waveform,
            orig_sr=orig_sr,
            target_sr=target_sr
        )

        logger.debug(f"Resampled from {orig_sr} Hz to {target_sr} Hz")
        return resampled

    def convert_to_mono(self, waveform: np.ndarray) -> np.ndarray:
        """
        Convert stereo to mono.

        Args:
            waveform: Audio waveform (can be 1D or 2D)

        Returns:
            Mono waveform
        """
        if waveform.ndim == 1:
            return waveform

        if waveform.ndim == 2:
            # Average channels
            return np.mean(waveform, axis=0)

        logger.warning(f"Unexpected waveform shape: {waveform.shape}")
        return waveform

    def full_preprocessing_pipeline(
        self,
        filepath: str,
        output_path: Optional[str] = None,
        normalize: bool = True,
        reduce_noise: bool = True,
        remove_silence: bool = False,
        enhance_speech: bool = True,
        high_pass_filter: bool = True
    ) -> Tuple[np.ndarray, int, Dict]:
        """
        Run complete preprocessing pipeline.

        Args:
            filepath: Input audio file
            output_path: Optional output file path
            normalize: Apply normalization
            reduce_noise: Apply noise reduction
            remove_silence: Remove silence
            enhance_speech: Enhance speech frequencies
            high_pass_filter: Apply high-pass filter

        Returns:
            Tuple of (processed waveform, sample rate, processing stats)
        """
        if not AUDIO_LIBS_AVAILABLE:
            raise ImportError("librosa and soundfile required for preprocessing")

        logger.info(f"Preprocessing audio: {filepath}")

        # Load audio
        waveform, sr = librosa.load(filepath, sr=self.target_sr, mono=True)

        stats = {
            'original_length': len(waveform) / sr,
            'original_peak': float(np.abs(waveform).max()),
            'original_rms': float(np.sqrt(np.mean(waveform ** 2)))
        }

        # Apply high-pass filter
        if high_pass_filter:
            waveform = self.apply_high_pass_filter(waveform, sr)

        # Noise reduction
        if reduce_noise:
            waveform = self.reduce_noise(waveform, sr)

        # Remove silence
        if remove_silence:
            waveform, removed_segments = self.remove_silence(waveform, sr)
            stats['removed_silence_segments'] = len(removed_segments)
            stats['new_length'] = len(waveform) / sr

        # Enhance speech
        if enhance_speech:
            waveform = self.enhance_speech(waveform, sr)

        # Normalize
        if normalize:
            waveform = self.normalize_audio(waveform, target_db=-20.0, method='rms')

        stats['final_peak'] = float(np.abs(waveform).max())
        stats['final_rms'] = float(np.sqrt(np.mean(waveform ** 2)))

        # Save if output path provided
        if output_path:
            sf.write(output_path, waveform, sr)
            logger.info(f"Saved preprocessed audio to {output_path}")

        logger.info(f"Preprocessing complete. Duration: {len(waveform)/sr:.2f}s")

        return waveform, sr, stats


def batch_preprocess_directory(
    input_dir: str,
    output_dir: str,
    **preprocessing_kwargs
) -> List[Dict]:
    """
    Batch preprocess all audio files in a directory.

    Args:
        input_dir: Input directory
        output_dir: Output directory
        **preprocessing_kwargs: Arguments for preprocessing pipeline

    Returns:
        List of processing statistics for each file
    """
    if not AUDIO_LIBS_AVAILABLE:
        raise ImportError("librosa and soundfile required for batch preprocessing")

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    audio_files = []

    for ext in audio_extensions:
        audio_files.extend(input_path.glob(f'*{ext}'))

    logger.info(f"Found {len(audio_files)} audio files to process")

    preprocessor = AudioPreprocessor()
    results = []

    for audio_file in audio_files:
        try:
            output_file = output_path / f"{audio_file.stem}_processed.wav"

            _, _, stats = preprocessor.full_preprocessing_pipeline(
                str(audio_file),
                str(output_file),
                **preprocessing_kwargs
            )

            stats['input_file'] = str(audio_file)
            stats['output_file'] = str(output_file)
            results.append(stats)

            logger.info(f"Processed: {audio_file.name}")

        except Exception as e:
            logger.error(f"Failed to process {audio_file}: {e}")
            results.append({
                'input_file': str(audio_file),
                'error': str(e)
            })

    logger.info(f"Batch processing complete: {len(results)} files")
    return results
