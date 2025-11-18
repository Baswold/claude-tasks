# Gesture Sync Studio Examples

This directory contains example scripts and files for getting started with Gesture Sync Studio.

## Demo Scripts

### demo_ml_generation.py

Comprehensive command-line demo showing both ML-based and rule-based gesture generation.

**Basic Usage:**

```bash
# Rule-based generation (no ML model needed)
python demo_ml_generation.py --audio speech.wav --rule-based

# ML-based generation with ONNX model
python demo_ml_generation.py --audio speech.wav --model path/to/model.onnx

# ML-based generation with TorchScript model
python demo_ml_generation.py --audio speech.wav --model path/to/model.pt
```

**Advanced Options:**

```bash
# Generate with custom FPS and bone list
python demo_ml_generation.py \
    --audio speech.wav \
    --rule-based \
    --fps 30 \
    --bones head neck spine hand.L hand.R

# Generate with visualization and statistics
python demo_ml_generation.py \
    --audio speech.wav \
    --rule-based \
    --visualize \
    --stats

# Adjust gesture intensity and smoothing
python demo_ml_generation.py \
    --audio speech.wav \
    --rule-based \
    --intensity 1.5 \
    --smoothing 7

# Run performance benchmark
python demo_ml_generation.py \
    --audio speech.wav \
    --rule-based \
    --benchmark
```

**Output:**
- JSON file with bone animations (default: `output_gestures.json`)
- Visualization plots (with `--visualize`): PNG file showing audio features and bone motions
- Statistics (with `--stats`): Detailed information about generated gestures

## Output File Format

### JSON Animation Format

```json
{
  "bone_name": {
    "frame_index": {
      "rotation_quaternion": [w, x, y, z],
      "location": [x, y, z]
    }
  }
}
```

Example:
```json
{
  "head": {
    "0": {
      "rotation_quaternion": [1.0, 0.0, 0.0, 0.0],
      "location": [0.0, 0.0, 1.5]
    },
    "1": {
      "rotation_quaternion": [0.99, 0.01, 0.0, 0.0],
      "location": [0.0, 0.0, 1.501]
    }
  }
}
```

## Blender Integration

### Quick Start with Blender

1. **Open Blender** with the Gesture Sync Studio addon installed
2. **Open or create** a rigged character model
3. **Load** your audio file in the addon panel
4. **Select** the armature object
5. **Click** "Generate Gesture Animation"

### Creating Compatible Characters

To create a compatible rigged character:

1. Model a character in sitting position
2. Create an armature with standard bone names:
   - head, neck, spine, spine.001, spine.002
   - shoulder.L/R, upper_arm.L/R, forearm.L/R, hand.L/R
3. Weight paint the mesh to the armature
4. Test with sample audio

## Audio Files

### Supported Formats

- WAV (recommended)
- MP3
- FLAC
- OGG
- M4A
- AAC

### Audio Recommendations

Good test audio should have:
- Clear speech
- Minimal background noise
- Natural pacing with pauses
- Varied intonation

You can use:
- Your own voice recordings
- Podcast clips
- Free speech datasets (LibriSpeech, Common Voice)
- Text-to-speech output

## Performance Tips

1. **Use appropriate FPS**: 24 FPS is standard for animation. Higher values increase file size.
2. **Limit bone count**: Only animate bones you need
3. **Use ONNX models**: Generally faster than TorchScript for inference
4. **Enable smoothing**: Window size 5-7 gives good results
5. **Shorter audio clips**: Process in chunks for very long audio

## Troubleshooting

### Missing Dependencies

```bash
# For audio processing
pip install librosa soundfile

# For ML inference (ONNX)
pip install onnxruntime

# For ML inference (PyTorch)
pip install torch

# For visualization
pip install matplotlib
```

### Common Errors

**"librosa not available"**
- Install librosa: `pip install librosa`

**"Model file not found"**
- Check the model path is correct
- Use `--rule-based` if you don't have a trained model

**"Audio file not found"**
- Check the audio file path is correct

**"Audio is very quiet"**
- Normalize your audio file
- Increase `--intensity` parameter

## Development Examples

### Custom Audio Features

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'blender_addon'))

from audio_processor import AudioProcessor

processor = AudioProcessor()
waveform, sr = processor.load_audio('speech.wav')
features = processor.extract_features(waveform, sr)

# Custom processing
speech_segments = processor.detect_speech_segments(features)
emphasis_points = processor.detect_emphasis_points(features)
```

### Custom Gesture Generation

```python
from gesture_generator import GestureGenerator

# Custom configuration
config = {
    'gesture_intensity': 2.0,
    'breathing_rate': 0.15,
    'head_nod_threshold': 0.5
}

generator = GestureGenerator(config=config)
gestures = generator.generate_gesture_sequence(features, fps=30)

# Apply custom smoothing
smoothed = generator.smooth_gesture_sequence(gestures, window_size=10)
```

## Note

Example files are not included in the repository to keep it lightweight.
You can create your own test audio or download examples from the project releases page.
