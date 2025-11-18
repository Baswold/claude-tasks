"""
Advanced visualization utilities for gesture analysis and debugging.

Provides tools to visualize:
- Audio features over time
- Bone trajectories and motion paths
- Gesture intensity heatmaps
- Quaternion rotation analysis
- Motion velocity and acceleration
- Comparative analysis (ML vs rule-based)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d import Axes3D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    logger.warning("matplotlib not available. Visualization features disabled.")


class GestureVisualizer:
    """
    Comprehensive visualization toolkit for gesture analysis.
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer.

        Args:
            style: Matplotlib style to use
        """
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib is required for visualization. "
                "Install with: pip install matplotlib"
            )

        try:
            plt.style.use(style)
        except:
            logger.warning(f"Could not load style '{style}', using default")

        self.figure_count = 0

    def plot_audio_features(
        self,
        audio_features: Dict[str, np.ndarray],
        output_path: str,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Create comprehensive audio features visualization.

        Args:
            audio_features: Dictionary of audio features
            output_path: Path to save figure
            figsize: Figure size (width, height)
        """
        times = audio_features.get('times', np.array([]))
        if len(times) == 0:
            logger.error("No time data in audio features")
            return

        # Determine number of subplots based on available features
        feature_plots = []

        if 'rms' in audio_features:
            feature_plots.append(('RMS Energy', audio_features['rms'], 'blue'))

        if 'zcr' in audio_features:
            feature_plots.append(('Zero Crossing Rate', audio_features['zcr'], 'green'))

        if 'spectral_centroid' in audio_features:
            sc = audio_features['spectral_centroid']
            sc_norm = sc / np.max(sc) if np.max(sc) > 0 else sc
            feature_plots.append(('Spectral Centroid (normalized)', sc_norm, 'orange'))

        if 'onset_strength' in audio_features:
            onset = audio_features['onset_strength']
            onset_norm = onset / np.max(onset) if np.max(onset) > 0 else onset
            feature_plots.append(('Onset Strength (normalized)', onset_norm, 'red'))

        n_plots = len(feature_plots)
        if n_plots == 0:
            logger.error("No features to plot")
            return

        # MFCC heatmap
        has_mfcc = 'mfcc' in audio_features and audio_features['mfcc'].ndim == 2

        fig = plt.figure(figsize=figsize)

        if has_mfcc:
            n_plots += 1

        # Plot individual features
        for i, (name, data, color) in enumerate(feature_plots, 1):
            ax = plt.subplot(n_plots, 1, i)

            # Ensure data matches times length
            if len(data) == len(times):
                ax.plot(times, data, color=color, alpha=0.7, linewidth=1.5)
                ax.fill_between(times, data, alpha=0.3, color=color)
            else:
                logger.warning(f"Skipping {name}: length mismatch")
                continue

            ax.set_ylabel(name, fontweight='bold')
            ax.grid(True, alpha=0.3)

            if i == 1:
                ax.set_title('Audio Features Over Time', fontsize=14, fontweight='bold')

        # Plot MFCC heatmap if available
        if has_mfcc:
            ax = plt.subplot(n_plots, 1, n_plots)
            mfcc = audio_features['mfcc']

            im = ax.imshow(
                mfcc,
                aspect='auto',
                origin='lower',
                extent=[times[0], times[-1], 0, mfcc.shape[0]],
                cmap='viridis',
                interpolation='bilinear'
            )

            ax.set_ylabel('MFCC Coefficients', fontweight='bold')
            ax.set_xlabel('Time (s)', fontweight='bold')

            plt.colorbar(im, ax=ax, label='Amplitude')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved audio features visualization to {output_path}")

    def plot_gesture_timeseries(
        self,
        gestures: Dict[str, Dict[int, Dict]],
        output_path: str,
        bones_to_plot: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 10)
    ):
        """
        Plot gesture motion over time for selected bones.

        Args:
            gestures: Gesture dictionary
            output_path: Path to save figure
            bones_to_plot: List of bones to visualize (None = auto-select interesting bones)
            figsize: Figure size
        """
        if not gestures:
            logger.error("No gestures to plot")
            return

        # Auto-select bones if not specified
        if bones_to_plot is None:
            all_bones = list(gestures.keys())
            # Prefer these bones for visualization
            priority_bones = ['head', 'hand.L', 'hand.R', 'spine', 'neck']
            bones_to_plot = [b for b in priority_bones if b in all_bones]

            if len(bones_to_plot) < 4 and all_bones:
                # Add more bones if needed
                for bone in all_bones:
                    if bone not in bones_to_plot:
                        bones_to_plot.append(bone)
                    if len(bones_to_plot) >= 6:
                        break

        if not bones_to_plot:
            logger.error("No bones to plot")
            return

        # Create figure
        fig, axes = plt.subplots(3, 1, figsize=figsize)

        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])

        for bone_name in bones_to_plot:
            if bone_name not in gestures:
                continue

            frames = gestures[bone_name]
            frame_indices = sorted(frames.keys())

            if not frame_indices:
                continue

            # Convert frames to seconds (assuming 24 FPS)
            times = [idx / 24.0 for idx in frame_indices]

            # Extract rotation magnitudes
            rotations = [np.array(frames[idx]['rotation_quaternion']) for idx in frame_indices]
            rotation_mags = [np.linalg.norm(r - identity_quat) for r in rotations]

            # Extract positions
            positions = [np.array(frames[idx]['location']) for idx in frame_indices]
            positions_array = np.array(positions)

            # Plot rotation magnitude
            axes[0].plot(times, rotation_mags, label=bone_name, alpha=0.7, linewidth=1.5)

            # Plot position (Y component - vertical movement)
            if positions_array.shape[1] >= 2:
                axes[1].plot(times, positions_array[:, 2], label=bone_name, alpha=0.7, linewidth=1.5)

            # Plot velocity (frame-to-frame rotation change)
            if len(rotations) > 1:
                velocities = []
                vel_times = []
                for i in range(len(rotations) - 1):
                    vel = np.linalg.norm(rotations[i+1] - rotations[i])
                    velocities.append(vel)
                    vel_times.append(times[i])

                axes[2].plot(vel_times, velocities, label=bone_name, alpha=0.7, linewidth=1.5)

        # Configure axes
        axes[0].set_ylabel('Rotation Magnitude', fontweight='bold')
        axes[0].set_title('Bone Motion Analysis', fontsize=14, fontweight='bold')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)

        axes[1].set_ylabel('Z Position (m)', fontweight='bold')
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1].grid(True, alpha=0.3)

        axes[2].set_ylabel('Angular Velocity', fontweight='bold')
        axes[2].set_xlabel('Time (s)', fontweight='bold')
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved gesture timeseries to {output_path}")

    def plot_bone_trajectory_3d(
        self,
        gestures: Dict[str, Dict[int, Dict]],
        output_path: str,
        bone_name: str = 'hand.R',
        figsize: Tuple[int, int] = (10, 10)
    ):
        """
        Plot 3D trajectory of a bone's movement.

        Args:
            gestures: Gesture dictionary
            output_path: Path to save figure
            bone_name: Bone to visualize
            figsize: Figure size
        """
        if bone_name not in gestures:
            logger.error(f"Bone {bone_name} not found in gestures")
            return

        frames = gestures[bone_name]
        frame_indices = sorted(frames.keys())

        positions = [np.array(frames[idx]['location']) for idx in frame_indices]
        positions_array = np.array(positions)

        if positions_array.shape[1] < 3:
            logger.error("Insufficient position data for 3D plot")
            return

        # Create 3D plot
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory
        xs = positions_array[:, 0]
        ys = positions_array[:, 1]
        zs = positions_array[:, 2]

        # Color by time
        colors = plt.cm.viridis(np.linspace(0, 1, len(frame_indices)))

        ax.scatter(xs, ys, zs, c=colors, s=20, alpha=0.6)

        # Plot path
        ax.plot(xs, ys, zs, 'gray', alpha=0.3, linewidth=1)

        # Mark start and end
        ax.scatter([xs[0]], [ys[0]], [zs[0]], c='green', s=100, marker='o', label='Start')
        ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], c='red', s=100, marker='X', label='End')

        ax.set_xlabel('X Position (m)', fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontweight='bold')
        ax.set_zlabel('Z Position (m)', fontweight='bold')
        ax.set_title(f'3D Trajectory: {bone_name}', fontsize=14, fontweight='bold')
        ax.legend()

        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved 3D trajectory to {output_path}")

    def plot_comparison(
        self,
        gestures_a: Dict[str, Dict[int, Dict]],
        gestures_b: Dict[str, Dict[int, Dict]],
        output_path: str,
        label_a: str = 'Method A',
        label_b: str = 'Method B',
        bones_to_compare: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Compare two gesture generation methods side by side.

        Args:
            gestures_a: First gesture set
            gestures_b: Second gesture set
            output_path: Path to save figure
            label_a: Label for first method
            label_b: Label for second method
            bones_to_compare: Bones to compare (None = auto-select)
            figsize: Figure size
        """
        # Auto-select bones
        if bones_to_compare is None:
            common_bones = set(gestures_a.keys()) & set(gestures_b.keys())
            priority = ['head', 'hand.L', 'hand.R', 'spine']
            bones_to_compare = [b for b in priority if b in common_bones][:4]

        if not bones_to_compare:
            logger.error("No common bones to compare")
            return

        n_bones = len(bones_to_compare)
        fig, axes = plt.subplots(n_bones, 1, figsize=figsize)

        if n_bones == 1:
            axes = [axes]

        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])

        for i, bone_name in enumerate(bones_to_compare):
            ax = axes[i]

            # Plot method A
            if bone_name in gestures_a:
                frames_a = gestures_a[bone_name]
                frame_indices_a = sorted(frames_a.keys())
                times_a = [idx / 24.0 for idx in frame_indices_a]
                rotations_a = [np.array(frames_a[idx]['rotation_quaternion']) for idx in frame_indices_a]
                mags_a = [np.linalg.norm(r - identity_quat) for r in rotations_a]

                ax.plot(times_a, mags_a, label=label_a, alpha=0.7, linewidth=2, color='blue')

            # Plot method B
            if bone_name in gestures_b:
                frames_b = gestures_b[bone_name]
                frame_indices_b = sorted(frames_b.keys())
                times_b = [idx / 24.0 for idx in frame_indices_b]
                rotations_b = [np.array(frames_b[idx]['rotation_quaternion']) for idx in frame_indices_b]
                mags_b = [np.linalg.norm(r - identity_quat) for r in rotations_b]

                ax.plot(times_b, mags_b, label=label_b, alpha=0.7, linewidth=2, color='red', linestyle='--')

            ax.set_ylabel(f'{bone_name}\nRotation', fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)

            if i == 0:
                ax.set_title(f'Comparison: {label_a} vs {label_b}', fontsize=14, fontweight='bold')

        axes[-1].set_xlabel('Time (s)', fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved comparison plot to {output_path}")

    def create_gesture_heatmap(
        self,
        gestures: Dict[str, Dict[int, Dict]],
        output_path: str,
        metric: str = 'rotation',
        figsize: Tuple[int, int] = (14, 8)
    ):
        """
        Create heatmap showing gesture intensity across bones and time.

        Args:
            gestures: Gesture dictionary
            output_path: Path to save figure
            metric: Metric to visualize ('rotation', 'velocity', 'position')
            figsize: Figure size
        """
        bone_names = sorted(gestures.keys())
        if not bone_names:
            logger.error("No bones in gestures")
            return

        # Get common frame range
        all_frames = set()
        for frames in gestures.values():
            all_frames.update(frames.keys())

        frame_indices = sorted(all_frames)
        if not frame_indices:
            logger.error("No frames in gestures")
            return

        # Build matrix
        matrix = np.zeros((len(bone_names), len(frame_indices)))

        identity_quat = np.array([1.0, 0.0, 0.0, 0.0])

        for bone_idx, bone_name in enumerate(bone_names):
            frames = gestures[bone_name]

            for frame_idx, frame_num in enumerate(frame_indices):
                if frame_num not in frames:
                    continue

                frame_data = frames[frame_num]

                if metric == 'rotation':
                    quat = np.array(frame_data['rotation_quaternion'])
                    value = np.linalg.norm(quat - identity_quat)
                elif metric == 'position':
                    pos = np.array(frame_data['location'])
                    value = np.linalg.norm(pos)
                elif metric == 'velocity' and frame_idx > 0:
                    prev_frame_num = frame_indices[frame_idx - 1]
                    if prev_frame_num in frames:
                        quat_curr = np.array(frame_data['rotation_quaternion'])
                        quat_prev = np.array(frames[prev_frame_num]['rotation_quaternion'])
                        value = np.linalg.norm(quat_curr - quat_prev)
                    else:
                        value = 0
                else:
                    value = 0

                matrix[bone_idx, frame_idx] = value

        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)

        times = [idx / 24.0 for idx in frame_indices]

        im = ax.imshow(
            matrix,
            aspect='auto',
            origin='lower',
            extent=[times[0], times[-1], 0, len(bone_names)],
            cmap='hot',
            interpolation='bilinear'
        )

        ax.set_yticks(range(len(bone_names)))
        ax.set_yticklabels(bone_names)
        ax.set_xlabel('Time (s)', fontweight='bold')
        ax.set_ylabel('Bone', fontweight='bold')
        ax.set_title(f'Gesture Intensity Heatmap ({metric.capitalize()})', fontsize=14, fontweight='bold')

        plt.colorbar(im, ax=ax, label='Magnitude')

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved gesture heatmap to {output_path}")

    def create_full_report(
        self,
        audio_features: Dict[str, np.ndarray],
        gestures: Dict[str, Dict[int, Dict]],
        output_dir: str,
        prefix: str = 'report'
    ):
        """
        Create comprehensive visualization report.

        Args:
            audio_features: Audio features
            gestures: Generated gestures
            output_dir: Output directory
            prefix: Filename prefix
        """
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Generating visualization report in {output_dir}")

        # Audio features
        self.plot_audio_features(
            audio_features,
            str(output_dir_path / f"{prefix}_audio.png")
        )

        # Gesture timeseries
        self.plot_gesture_timeseries(
            gestures,
            str(output_dir_path / f"{prefix}_timeseries.png")
        )

        # Heatmaps
        self.create_gesture_heatmap(
            gestures,
            str(output_dir_path / f"{prefix}_heatmap_rotation.png"),
            metric='rotation'
        )

        self.create_gesture_heatmap(
            gestures,
            str(output_dir_path / f"{prefix}_heatmap_velocity.png"),
            metric='velocity'
        )

        # 3D trajectories for interesting bones
        for bone in ['hand.R', 'hand.L', 'head']:
            if bone in gestures:
                self.plot_bone_trajectory_3d(
                    gestures,
                    str(output_dir_path / f"{prefix}_trajectory_{bone}.png"),
                    bone_name=bone
                )

        logger.info(f"Visualization report complete: {output_dir}")


def is_visualization_available() -> bool:
    """Check if visualization is available."""
    return MATPLOTLIB_AVAILABLE
