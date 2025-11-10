"""
Data loader for audio-gesture training dataset.
"""

import torch
import torch.utils.data as data
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import audio processing
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available")


class GestureDataset(data.Dataset):
    """PyTorch dataset for audio-gesture pairs."""

    def __init__(
        self,
        dataset_path: str,
        audio_dir: str = "audio",
        motion_dir: str = "motion",
        metadata_file: str = "metadata.json",
        sample_rate: int = 22050,
        n_mfcc: int = 26,
        frame_length: int = 2048,
        hop_length: int = 512,
        fps: int = 24,
        max_duration: Optional[float] = None
    ):
        """
        Initialize dataset.

        Args:
            dataset_path: Root path to dataset
            audio_dir: Subdirectory with audio files
            motion_dir: Subdirectory with motion files
            metadata_file: Metadata JSON file
            sample_rate: Audio sample rate
            n_mfcc: Number of MFCC coefficients
            frame_length: FFT frame length
            hop_length: Hop length for audio processing
            fps: Target FPS for motion data
            max_duration: Maximum clip duration (None = no limit)
        """
        self.dataset_path = Path(dataset_path)
        self.audio_dir = self.dataset_path / audio_dir
        self.motion_dir = self.dataset_path / motion_dir
        self.metadata_file = self.dataset_path / metadata_file

        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.fps = fps
        self.max_duration = max_duration

        # Load metadata
        self.samples = self._load_metadata()

        logger.info(f"Loaded dataset with {len(self.samples)} samples from {dataset_path}")

    def _load_metadata(self) -> List[Dict]:
        """
        Load metadata file or create from directory listing.

        Returns:
            List of sample dictionaries
        """
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
                return metadata.get('samples', [])
        else:
            # Auto-discover pairs by matching filenames
            logger.info("No metadata file found, auto-discovering audio-motion pairs")
            return self._discover_pairs()

    def _discover_pairs(self) -> List[Dict]:
        """
        Auto-discover audio-motion pairs by matching filenames.

        Returns:
            List of sample dictionaries
        """
        samples = []

        # Find all audio files
        audio_files = list(self.audio_dir.glob("*.wav")) + list(self.audio_dir.glob("*.mp3"))

        for audio_path in audio_files:
            # Look for matching motion file
            stem = audio_path.stem
            motion_candidates = [
                self.motion_dir / f"{stem}.json",
                self.motion_dir / f"{stem}.bvh",
            ]

            for motion_path in motion_candidates:
                if motion_path.exists():
                    samples.append({
                        'audio': str(audio_path.name),
                        'motion': str(motion_path.name),
                        'id': stem
                    })
                    break

        logger.info(f"Discovered {len(samples)} audio-motion pairs")
        return samples

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a training sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (audio_features, motion_sequence)
            - audio_features: (time_steps, feature_dim)
            - motion_sequence: (time_steps, bone_dim)
        """
        sample = self.samples[idx]

        # Load and process audio
        audio_path = self.audio_dir / sample['audio']
        audio_features = self.preprocess_audio(str(audio_path))

        # Load and process motion
        motion_path = self.motion_dir / sample['motion']
        motion_sequence = self.preprocess_motion(str(motion_path))

        # Align temporal dimensions
        audio_features, motion_sequence = self._align_sequences(
            audio_features, motion_sequence
        )

        # Convert to tensors
        audio_tensor = torch.from_numpy(audio_features).float()
        motion_tensor = torch.from_numpy(motion_sequence).float()

        return audio_tensor, motion_tensor

    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """
        Extract audio features.

        Args:
            audio_path: Path to audio file

        Returns:
            Feature array of shape (time_steps, feature_dim)
        """
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required for audio processing")

        # Load audio
        waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Truncate if max_duration specified
        if self.max_duration:
            max_samples = int(self.max_duration * sr)
            waveform = waveform[:max_samples]

        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )

        # Extract additional features
        rms = librosa.feature.rms(
            y=waveform,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]

        zcr = librosa.feature.zero_crossing_rate(
            waveform,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]

        spectral_centroid = librosa.feature.spectral_centroid(
            y=waveform,
            sr=sr,
            n_fft=self.frame_length,
            hop_length=self.hop_length
        )[0]

        # Combine features
        # Shape: (n_mfcc + 3, time_steps)
        features = np.vstack([mfcc, rms[np.newaxis, :], zcr[np.newaxis, :],
                             spectral_centroid[np.newaxis, :]])

        # Transpose to (time_steps, feature_dim)
        features = features.T

        return features

    def preprocess_motion(self, motion_path: str) -> np.ndarray:
        """
        Load and process motion data.

        Args:
            motion_path: Path to motion file (JSON or BVH)

        Returns:
            Motion array of shape (time_steps, bone_dim)
        """
        if motion_path.endswith('.json'):
            return self._load_motion_json(motion_path)
        elif motion_path.endswith('.bvh'):
            return self._load_motion_bvh(motion_path)
        else:
            raise ValueError(f"Unsupported motion format: {motion_path}")

    def _load_motion_json(self, motion_path: str) -> np.ndarray:
        """
        Load motion from JSON format.

        JSON format:
        {
            "fps": 24,
            "bones": {
                "bone_name": {
                    "frames": [
                        {"time": 0.0, "rotation": [w, x, y, z], "location": [x, y, z]},
                        ...
                    ]
                }
            }
        }

        Args:
            motion_path: Path to JSON file

        Returns:
            Motion array of shape (time_steps, bone_dim)
        """
        with open(motion_path, 'r') as f:
            data = json.load(f)

        fps = data.get('fps', self.fps)
        bones_data = data.get('bones', {})

        # Get list of bones and determine number of frames
        bone_names = sorted(bones_data.keys())
        num_frames = 0

        for bone_name, bone_data in bones_data.items():
            frames = bone_data.get('frames', [])
            num_frames = max(num_frames, len(frames))

        # Each bone: 4 (quaternion) + 3 (location) = 7 values
        bone_dim = len(bone_names) * 7
        motion_sequence = np.zeros((num_frames, bone_dim))

        # Fill motion data
        for bone_idx, bone_name in enumerate(bone_names):
            bone_data = bones_data[bone_name]
            frames = bone_data.get('frames', [])

            for frame_idx, frame_data in enumerate(frames):
                if frame_idx >= num_frames:
                    break

                # Get rotation (quaternion)
                rotation = frame_data.get('rotation', [1, 0, 0, 0])
                location = frame_data.get('location', [0, 0, 0])

                # Store in flat array
                start_idx = bone_idx * 7
                motion_sequence[frame_idx, start_idx:start_idx+4] = rotation
                motion_sequence[frame_idx, start_idx+4:start_idx+7] = location

        return motion_sequence

    def _load_motion_bvh(self, motion_path: str) -> np.ndarray:
        """
        Load motion from BVH format.

        Args:
            motion_path: Path to BVH file

        Returns:
            Motion array
        """
        # TODO: Implement BVH parser
        # For now, raise not implemented
        raise NotImplementedError("BVH loading not yet implemented")

    def _align_sequences(
        self,
        audio_features: np.ndarray,
        motion_sequence: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align audio and motion sequences to same temporal dimension.

        Args:
            audio_features: Audio feature array (T1, F)
            motion_sequence: Motion sequence array (T2, M)

        Returns:
            Aligned (audio_features, motion_sequence) with same T
        """
        audio_len = audio_features.shape[0]
        motion_len = motion_sequence.shape[0]

        # Use shorter length to avoid extrapolation
        target_len = min(audio_len, motion_len)

        # Truncate or interpolate to match
        if audio_len != target_len:
            # Simple linear interpolation
            indices = np.linspace(0, audio_len - 1, target_len)
            audio_features = np.array([
                audio_features[int(i)] for i in indices
            ])

        if motion_len != target_len:
            indices = np.linspace(0, motion_len - 1, target_len)
            motion_sequence = np.array([
                motion_sequence[int(i)] for i in indices
            ])

        return audio_features, motion_sequence


def collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function for DataLoader to handle variable-length sequences.

    Args:
        batch: List of (audio, motion) tuples

    Returns:
        Batched and padded tensors
    """
    # Find max length in batch
    max_len = max(audio.shape[0] for audio, _ in batch)

    # Pad sequences
    audio_batch = []
    motion_batch = []

    for audio, motion in batch:
        # Pad audio
        audio_pad = torch.zeros(max_len, audio.shape[1])
        audio_pad[:audio.shape[0]] = audio
        audio_batch.append(audio_pad)

        # Pad motion
        motion_pad = torch.zeros(max_len, motion.shape[1])
        motion_pad[:motion.shape[0]] = motion
        motion_batch.append(motion_pad)

    # Stack into batch
    audio_batch = torch.stack(audio_batch)
    motion_batch = torch.stack(motion_batch)

    return audio_batch, motion_batch


def create_dataloader(
    dataset: GestureDataset,
    batch_size: int = 16,
    shuffle: bool = True,
    num_workers: int = 4
) -> data.DataLoader:
    """
    Create PyTorch DataLoader for training.

    Args:
        dataset: GestureDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes

    Returns:
        DataLoader instance
    """
    return data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
