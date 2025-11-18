#!/usr/bin/env python3
"""
Demo script for ML-based gesture generation.

This script demonstrates:
1. Loading and processing audio
2. Generating gestures with both rule-based and ML models
3. Exporting results to various formats
4. Visualizing the generated gestures

Usage:
    python demo_ml_generation.py --audio speech.wav --model model.onnx
    python demo_ml_generation.py --audio speech.wav --rule-based
    python demo_ml_generation.py --help
"""

import sys
import argparse
import logging
from pathlib import Path
import json
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'blender_addon'))

import numpy as np
from audio_processor import AudioProcessor
from gesture_generator import GestureGenerator


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate gestures from audio using ML or rule-based methods'
    )

    # Input/output
    parser.add_argument(
        '--audio', '-a',
        type=str,
        required=True,
        help='Path to input audio file (WAV, MP3, etc.)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='output_gestures.json',
        help='Path to output JSON file (default: output_gestures.json)'
    )

    # Model selection
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument(
        '--model', '-m',
        type=str,
        help='Path to trained ML model (ONNX or TorchScript)'
    )
    model_group.add_argument(
        '--rule-based', '-r',
        action='store_true',
        help='Use rule-based generation (no ML model)'
    )

    # Configuration
    parser.add_argument(
        '--fps',
        type=int,
        default=24,
        help='Target frames per second (default: 24)'
    )
    parser.add_argument(
        '--bones',
        type=str,
        nargs='+',
        help='Specific bone names to animate (default: all standard bones)'
    )
    parser.add_argument(
        '--intensity',
        type=float,
        default=1.0,
        help='Gesture intensity multiplier (default: 1.0)'
    )
    parser.add_argument(
        '--smoothing',
        type=int,
        default=5,
        help='Smoothing window size in frames (default: 5)'
    )

    # Audio processing
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=22050,
        help='Audio sample rate (default: 22050)'
    )

    # Visualization
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots (requires matplotlib)'
    )
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Print detailed statistics'
    )

    # Advanced
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run performance benchmark'
    )

    return parser.parse_args()


def print_statistics(gestures, audio_features, processing_time):
    """Print detailed statistics about generated gestures."""
    print("\n" + "="*60)
    print("GENERATION STATISTICS")
    print("="*60)

    # Audio info
    duration = audio_features['times'][-1] - audio_features['times'][0]
    print(f"\nAudio:")
    print(f"  Duration: {duration:.2f}s")
    print(f"  Feature frames: {len(audio_features['rms'])}")
    print(f"  Average RMS: {np.mean(audio_features['rms']):.4f}")
    print(f"  Peak RMS: {np.max(audio_features['rms']):.4f}")

    # Gesture info
    num_bones = len(gestures)
    num_frames = len(list(gestures.values())[0]) if gestures else 0

    print(f"\nGestures:")
    print(f"  Bones animated: {num_bones}")
    print(f"  Animation frames: {num_frames}")
    print(f"  Animation duration: {num_frames / 24:.2f}s (at 24 FPS)")

    # Performance
    print(f"\nPerformance:")
    print(f"  Processing time: {processing_time:.2f}s")
    print(f"  Frames per second: {num_frames / processing_time:.1f}")
    print(f"  Realtime factor: {duration / processing_time:.2f}x")

    # Bone statistics
    print(f"\nBone Statistics:")
    for bone_name in sorted(gestures.keys()):
        frames = gestures[bone_name]
        if len(frames) == 0:
            continue

        # Calculate average rotation magnitude
        rotations = [np.array(f['rotation_quaternion']) for f in frames.values()]
        # Distance from identity quaternion (1, 0, 0, 0)
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        distances = [np.linalg.norm(r - identity) for r in rotations]
        avg_rotation = np.mean(distances)

        print(f"  {bone_name:20s} - Avg rotation: {avg_rotation:.4f}")

    print("="*60 + "\n")


def visualize_gestures(gestures, audio_features, output_path):
    """Generate visualization plots."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("matplotlib not available. Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Audio features
    times = audio_features['times']
    axes[0].plot(times, audio_features['rms'], label='RMS Energy', color='blue', alpha=0.7)
    if 'onset_strength' in audio_features:
        onset = audio_features['onset_strength']
        if len(onset) == len(times):
            axes[0].plot(times, onset / np.max(onset), label='Onset Strength (norm)',
                        color='red', alpha=0.5)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title('Audio Features')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Bone rotation magnitudes over time
    bone_samples = ['head', 'hand.L', 'hand.R', 'spine']
    identity = np.array([1.0, 0.0, 0.0, 0.0])

    for bone_name in bone_samples:
        if bone_name not in gestures:
            continue

        frames = gestures[bone_name]
        frame_indices = sorted(frames.keys())
        rotations = [np.array(frames[i]['rotation_quaternion']) for i in frame_indices]
        magnitudes = [np.linalg.norm(r - identity) for r in rotations]

        frame_times = [i / 24.0 for i in frame_indices]  # Convert to seconds
        axes[1].plot(frame_times, magnitudes, label=bone_name, alpha=0.7)

    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Rotation Magnitude')
    axes[1].set_title('Bone Rotation Magnitudes Over Time')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Motion velocity (frame-to-frame change)
    for bone_name in bone_samples:
        if bone_name not in gestures:
            continue

        frames = gestures[bone_name]
        frame_indices = sorted(frames.keys())

        if len(frame_indices) < 2:
            continue

        velocities = []
        for i in range(len(frame_indices) - 1):
            f1 = frame_indices[i]
            f2 = frame_indices[i + 1]

            r1 = np.array(frames[f1]['rotation_quaternion'])
            r2 = np.array(frames[f2]['rotation_quaternion'])

            velocity = np.linalg.norm(r2 - r1)
            velocities.append(velocity)

        frame_times = [i / 24.0 for i in frame_indices[:-1]]
        axes[2].plot(frame_times, velocities, label=bone_name, alpha=0.7)

    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Angular Velocity')
    axes[2].set_title('Bone Motion Velocity (Frame-to-Frame Change)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    viz_path = output_path.replace('.json', '_visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved visualization to {viz_path}")
    plt.close()


def run_benchmark(audio_processor, generator, audio_path, iterations=5):
    """Run performance benchmark."""
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)

    timings = {
        'audio_load': [],
        'feature_extraction': [],
        'gesture_generation': [],
        'total': []
    }

    for i in range(iterations):
        print(f"\nIteration {i+1}/{iterations}...")

        # Load audio
        start = time.time()
        waveform, sr = audio_processor.load_audio(audio_path)
        timings['audio_load'].append(time.time() - start)

        # Extract features
        start = time.time()
        audio_features = audio_processor.extract_features(waveform, sr)
        timings['feature_extraction'].append(time.time() - start)

        # Generate gestures
        start = time.time()
        gestures = generator.generate_gesture_sequence(audio_features, fps=24)
        timings['gesture_generation'].append(time.time() - start)

        timings['total'].append(sum([
            timings['audio_load'][-1],
            timings['feature_extraction'][-1],
            timings['gesture_generation'][-1]
        ]))

    print(f"\nBenchmark Results (avg over {iterations} iterations):")
    print(f"  Audio loading:       {np.mean(timings['audio_load'])*1000:6.1f} ms ± {np.std(timings['audio_load'])*1000:.1f}")
    print(f"  Feature extraction:  {np.mean(timings['feature_extraction'])*1000:6.1f} ms ± {np.std(timings['feature_extraction'])*1000:.1f}")
    print(f"  Gesture generation:  {np.mean(timings['gesture_generation'])*1000:6.1f} ms ± {np.std(timings['gesture_generation'])*1000:.1f}")
    print(f"  Total:               {np.mean(timings['total'])*1000:6.1f} ms ± {np.std(timings['total'])*1000:.1f}")
    print("="*60 + "\n")


def main():
    """Main demo function."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print header
    print("\n" + "="*60)
    print("GESTURE SYNC STUDIO - ML GENERATION DEMO")
    print("="*60 + "\n")

    try:
        # Initialize audio processor
        logger.info("Initializing audio processor...")
        audio_processor = AudioProcessor(sr=args.sample_rate)

        # Initialize gesture generator
        logger.info("Initializing gesture generator...")
        config = {
            'gesture_intensity': args.intensity,
            'smoothing': 0.5
        }

        if args.model and not args.rule_based:
            logger.info(f"Loading ML model from {args.model}")
            generator = GestureGenerator(model_path=args.model, config=config)
        else:
            logger.info("Using rule-based generation")
            generator = GestureGenerator(config=config)

        # Run benchmark if requested
        if args.benchmark:
            run_benchmark(audio_processor, generator, args.audio)

        # Load audio
        logger.info(f"Loading audio from {args.audio}")
        start_time = time.time()
        waveform, sr = audio_processor.load_audio(args.audio)
        load_time = time.time() - start_time
        logger.info(f"Audio loaded in {load_time:.2f}s")

        # Extract features
        logger.info("Extracting audio features...")
        start_time = time.time()
        audio_features = audio_processor.extract_features(waveform, sr)
        feature_time = time.time() - start_time
        logger.info(f"Features extracted in {feature_time:.2f}s")

        # Generate gestures
        logger.info("Generating gestures...")
        start_time = time.time()
        gestures = generator.generate_gesture_sequence(
            audio_features,
            fps=args.fps,
            bone_names=args.bones
        )
        generation_time = time.time() - start_time
        logger.info(f"Gestures generated in {generation_time:.2f}s")

        # Apply smoothing if requested
        if args.smoothing > 1:
            logger.info(f"Applying smoothing (window={args.smoothing})...")
            gestures = generator.smooth_gesture_sequence(gestures, window_size=args.smoothing)

        # Export to JSON
        logger.info(f"Exporting to {args.output}")
        generator.export_to_json(gestures, args.output)

        # Print statistics
        total_time = load_time + feature_time + generation_time
        if args.stats:
            print_statistics(gestures, audio_features, total_time)

        # Visualize if requested
        if args.visualize:
            logger.info("Generating visualization...")
            visualize_gestures(gestures, audio_features, args.output)

        # Success message
        print(f"\n✓ Success! Generated {len(gestures)} bone animations")
        print(f"  Output: {args.output}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Method: {'ML model' if args.model else 'Rule-based'}\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        print(f"\n✗ Error: {e}\n")
        sys.exit(1)


if __name__ == '__main__':
    main()
