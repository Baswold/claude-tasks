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

        BVH (Biovision Hierarchy) is a common motion capture file format.
        Structure:
        - HIERARCHY section: Defines skeleton structure
        - MOTION section: Contains frame data

        Args:
            motion_path: Path to BVH file

        Returns:
            Motion array of shape (time_steps, bone_dim)
            Each bone has 7 values: 4 (quaternion rotation) + 3 (location)

        Raises:
            ValueError: If BVH file is malformed
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(motion_path):
            raise FileNotFoundError(f"BVH file not found: {motion_path}")

        try:
            with open(motion_path, 'r') as f:
                lines = f.readlines()

            # Parse BVH file
            hierarchy, motion_data = self._parse_bvh_file(lines)

            # Extract bone information
            bone_names = hierarchy['bone_names']
            bone_channels = hierarchy['bone_channels']
            num_frames = motion_data['num_frames']
            frame_time = motion_data['frame_time']
            channel_data = motion_data['channel_data']

            # Convert to our format (quaternions + locations)
            motion_sequence = self._bvh_to_motion_array(
                bone_names, bone_channels, channel_data, num_frames
            )

            logger.info(
                f"Loaded BVH: {len(bone_names)} bones, {num_frames} frames, "
                f"{frame_time:.4f}s per frame ({1/frame_time:.1f} FPS)"
            )

            return motion_sequence

        except Exception as e:
            logger.error(f"Failed to parse BVH file {motion_path}: {e}")
            raise ValueError(f"Failed to parse BVH file: {e}")

    def _parse_bvh_file(self, lines: List[str]) -> Tuple[Dict, Dict]:
        """
        Parse BVH file into hierarchy and motion sections.

        Args:
            lines: Lines from BVH file

        Returns:
            Tuple of (hierarchy_dict, motion_dict)
        """
        # Find section boundaries
        hierarchy_start = -1
        motion_start = -1

        for i, line in enumerate(lines):
            if line.strip().startswith('HIERARCHY'):
                hierarchy_start = i
            elif line.strip().startswith('MOTION'):
                motion_start = i
                break

        if hierarchy_start == -1 or motion_start == -1:
            raise ValueError("Invalid BVH file: missing HIERARCHY or MOTION section")

        # Parse hierarchy
        hierarchy_lines = lines[hierarchy_start:motion_start]
        hierarchy = self._parse_bvh_hierarchy(hierarchy_lines)

        # Parse motion
        motion_lines = lines[motion_start:]
        motion_data = self._parse_bvh_motion(motion_lines, hierarchy)

        return hierarchy, motion_data

    def _parse_bvh_hierarchy(self, lines: List[str]) -> Dict:
        """
        Parse BVH HIERARCHY section.

        Args:
            lines: Lines from HIERARCHY section

        Returns:
            Dictionary with bone information
        """
        bone_names = []
        bone_channels = {}
        bone_parents = {}
        bone_offsets = {}

        current_bone = None
        parent_stack = []

        for line in lines:
            stripped = line.strip()

            # Parse bone declarations
            if stripped.startswith('ROOT') or stripped.startswith('JOINT'):
                parts = stripped.split()
                if len(parts) >= 2:
                    bone_name = parts[1]
                    bone_names.append(bone_name)
                    current_bone = bone_name

                    if parent_stack:
                        bone_parents[bone_name] = parent_stack[-1]
                    else:
                        bone_parents[bone_name] = None

                    parent_stack.append(bone_name)

            # Parse offsets
            elif stripped.startswith('OFFSET'):
                parts = stripped.split()
                if len(parts) >= 4 and current_bone:
                    offset = [float(parts[1]), float(parts[2]), float(parts[3])]
                    bone_offsets[current_bone] = offset

            # Parse channels
            elif stripped.startswith('CHANNELS'):
                parts = stripped.split()
                if len(parts) >= 2 and current_bone:
                    num_channels = int(parts[1])
                    channels = parts[2:2+num_channels]
                    bone_channels[current_bone] = channels

            # Handle end of bone
            elif stripped.startswith('}'):
                if parent_stack:
                    parent_stack.pop()

        return {
            'bone_names': bone_names,
            'bone_channels': bone_channels,
            'bone_parents': bone_parents,
            'bone_offsets': bone_offsets
        }

    def _parse_bvh_motion(self, lines: List[str], hierarchy: Dict) -> Dict:
        """
        Parse BVH MOTION section.

        Args:
            lines: Lines from MOTION section
            hierarchy: Parsed hierarchy information

        Returns:
            Dictionary with motion data
        """
        num_frames = 0
        frame_time = 0.0
        channel_data = []

        reading_frames = False

        for line in lines:
            stripped = line.strip()

            if stripped.startswith('Frames:'):
                num_frames = int(stripped.split()[1])

            elif stripped.startswith('Frame Time:'):
                frame_time = float(stripped.split()[2])

            elif stripped and not stripped.startswith('MOTION'):
                # This is frame data
                try:
                    frame_values = [float(x) for x in stripped.split()]
                    channel_data.append(frame_values)
                except ValueError:
                    # Skip invalid lines
                    continue

        return {
            'num_frames': num_frames,
            'frame_time': frame_time,
            'channel_data': np.array(channel_data) if channel_data else np.array([])
        }

    def _bvh_to_motion_array(
        self,
        bone_names: List[str],
        bone_channels: Dict[str, List[str]],
        channel_data: np.ndarray,
        num_frames: int
    ) -> np.ndarray:
        """
        Convert BVH channel data to motion array format.

        Args:
            bone_names: List of bone names
            bone_channels: Dictionary mapping bones to their channels
            channel_data: Raw channel data from BVH
            num_frames: Number of frames

        Returns:
            Motion array of shape (num_frames, bone_dim)
            bone_dim = num_bones * 7 (4 quaternion + 3 location)
        """
        num_bones = len(bone_names)
        bone_dim = num_bones * 7
        motion_array = np.zeros((num_frames, bone_dim))

        # Build channel index mapping
        channel_idx = 0
        bone_channel_indices = {}

        for bone_name in bone_names:
            channels = bone_channels.get(bone_name, [])
            bone_channel_indices[bone_name] = {
                'start': channel_idx,
                'channels': channels
            }
            channel_idx += len(channels)

        # Convert frame by frame
        for frame_idx in range(min(num_frames, len(channel_data))):
            frame_data = channel_data[frame_idx]

            for bone_idx, bone_name in enumerate(bone_names):
                if bone_name not in bone_channel_indices:
                    continue

                bone_info = bone_channel_indices[bone_name]
                start_idx = bone_info['start']
                channels = bone_info['channels']

                # Extract channel values
                position = [0.0, 0.0, 0.0]
                rotation_euler = [0.0, 0.0, 0.0]  # XYZ Euler angles in degrees

                for i, channel in enumerate(channels):
                    if start_idx + i >= len(frame_data):
                        break

                    value = frame_data[start_idx + i]

                    # Position channels
                    if channel == 'Xposition':
                        position[0] = value
                    elif channel == 'Yposition':
                        position[1] = value
                    elif channel == 'Zposition':
                        position[2] = value

                    # Rotation channels (typically in ZXY order for BVH)
                    elif channel == 'Xrotation':
                        rotation_euler[0] = value
                    elif channel == 'Yrotation':
                        rotation_euler[1] = value
                    elif channel == 'Zrotation':
                        rotation_euler[2] = value

                # Convert Euler angles to quaternion
                quaternion = self._euler_to_quaternion(
                    np.radians(rotation_euler[0]),
                    np.radians(rotation_euler[1]),
                    np.radians(rotation_euler[2])
                )

                # Store in motion array
                base_idx = bone_idx * 7
                motion_array[frame_idx, base_idx:base_idx+4] = quaternion
                motion_array[frame_idx, base_idx+4:base_idx+7] = position

        return motion_array

    def _euler_to_quaternion(self, x: float, y: float, z: float) -> np.ndarray:
        """
        Convert Euler angles (XYZ order) to quaternion.

        Args:
            x: Rotation around X axis (radians)
            y: Rotation around Y axis (radians)
            z: Rotation around Z axis (radians)

        Returns:
            Quaternion as numpy array [w, x, y, z]
        """
        # Compute half angles
        cx = np.cos(x * 0.5)
        sx = np.sin(x * 0.5)
        cy = np.cos(y * 0.5)
        sy = np.sin(y * 0.5)
        cz = np.cos(z * 0.5)
        sz = np.sin(z * 0.5)

        # Quaternion multiplication for XYZ order
        w = cx * cy * cz + sx * sy * sz
        qx = sx * cy * cz - cx * sy * sz
        qy = cx * sy * cz + sx * cy * sz
        qz = cx * cy * sz - sx * sy * cz

        return np.array([w, qx, qy, qz])

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
