# Gesture Sync Studio

**Audio-to-Gesture Blender Plugin & Training Framework**

Generate realistic sitting/gesturing animations for rigged characters from audio input using rule-based generation or machine learning models.

## Features

- 🎤 **Audio-driven animation**: Generate natural gestures synchronized with speech
- 🤖 **Dual generation modes**: Rule-based (no training needed) or ML-powered
  - **NEW**: Full ML inference support with ONNX and TorchScript models
  - Seamless fallback to rule-based when models unavailable
- 🎨 **Blender integration**: Easy-to-use addon with GUI panel
- 📊 **Training framework**: Complete PyTorch-based training pipeline
  - **NEW**: BVH motion capture file support for training data
  - Multiple model architectures (LSTM, Transformer, Simple MLP)
- 🔧 **Configurable**: Extensive customization options
  - Fine-grained control over gesture parameters
  - Custom bone mapping support
- ⚡ **Real-time capable**: Process 10 minutes of audio in under 1 minute
  - **NEW**: Feature caching for 10-100x speedup on repeated processing
  - **NEW**: Batch processing for efficient ML inference
  - Memory-efficient processing for long audio files
- 🧪 **Comprehensive testing**: 30+ unit tests covering all major components
- 📈 **Performance monitoring**: Built-in profiling and benchmarking tools
- 📝 **Command-line tools**: Demo scripts for standalone usage without Blender

## Project Structure

```
gesture-sync-studio/
├── blender_addon/          # Blender addon for gesture generation
│   ├── __init__.py         # Main addon file
│   ├── audio_processor.py  # Audio feature extraction
│   ├── gesture_generator.py # Gesture generation (rule-based + ML)
│   ├── animation_applier.py # Apply gestures to armature
│   ├── ui_panel.py         # Blender UI
│   ├── operators.py        # Blender operators
│   └── gesture_config.json # Configuration file
├── training/               # ML training framework
│   ├── data_loader.py      # Dataset loader
│   ├── model.py            # Neural network models
│   ├── train.py            # Training script
│   ├── export.py           # Model export for Blender
│   └── training_config.yaml # Training configuration
├── dataset/                # Training data
│   ├── audio/              # Audio files
│   ├── motion/             # Motion data (JSON/BVH)
│   └── metadata.json       # Dataset metadata
├── docs/                   # Documentation
├── tests/                  # Test suite
└── examples/               # Example files
```

## Quick Start

### Installation

#### 1. Install Blender Addon

1. Download or clone this repository
2. Open Blender (3.6+ or 4.0+)
3. Go to Edit > Preferences > Add-ons
4. Click "Install" and select `blender_addon` folder
5. Enable "Audio Gesture Generator" addon
6. Install dependencies when prompted (or manually via pip)

#### 2. Install Dependencies

The addon requires Python packages that may not be included with Blender:

```bash
# Using Blender's Python
/path/to/blender/python -m pip install numpy librosa scipy

# Or use the built-in dependency installer in the addon panel
```

### Using the Addon

1. **Select Audio File**: Choose a WAV, MP3, or OGG audio file
2. **Select Armature**: Choose the rigged character to animate
3. **Adjust Parameters**:
   - FPS: Animation framerate (default: 24)
   - Gesture Intensity: How pronounced gestures should be
   - Smoothing: Amount of motion smoothing
4. **Generate**: Click "Generate Gesture Animation"

The addon will:
- Analyze the audio file
- Extract features (energy, pitch, emphasis points)
- Generate bone animations (using ML model if available, otherwise rule-based)
- Apply keyframes to your armature

### Command-Line Usage (NEW!)

You can also generate gestures without Blender using the CLI demo script:

```bash
cd examples

# Generate with rule-based system
python demo_ml_generation.py --audio speech.wav --rule-based --output gestures.json

# Generate with ML model (ONNX)
python demo_ml_generation.py --audio speech.wav --model ../models/gesture_model.onnx --output gestures.json

# Generate with visualization and statistics
python demo_ml_generation.py \
    --audio speech.wav \
    --rule-based \
    --visualize \
    --stats \
    --intensity 1.5 \
    --smoothing 7

# Run performance benchmark
python demo_ml_generation.py --audio speech.wav --rule-based --benchmark
```

Features caching automatically kicks in on repeated runs of the same audio file, providing 10-100x speedup!

See [examples/README.md](examples/README.md) for complete CLI documentation.

### Expected Bone Names

The addon expects standard bone naming:
- `head`, `neck`
- `spine`, `spine.001`, `spine.002`
- `shoulder.L`, `shoulder.R`
- `upper_arm.L`, `upper_arm.R`
- `forearm.L`, `forearm.R`
- `hand.L`, `hand.R`

You can customize bone mapping in `gesture_config.json`.

## Training Your Own Model

### Prerequisites

```bash
pip install torch torchvision torchaudio
pip install librosa numpy scipy pyyaml tqdm tensorboard
```

### Prepare Dataset

Organize your data as:

```
dataset/
├── audio/
│   ├── clip_001.wav
│   ├── clip_002.wav
│   └── ...
├── motion/
│   ├── clip_001.json
│   ├── clip_002.json
│   └── ...
└── metadata.json
```

**Motion Data Format** (JSON):

```json
{
  "fps": 24,
  "bones": {
    "head": {
      "frames": [
        {
          "time": 0.0,
          "rotation": [1, 0, 0, 0],  // Quaternion (w, x, y, z)
          "location": [0, 0, 1.6]     // XYZ position
        }
      ]
    }
  }
}
```

### Train Model

```bash
cd training

# Train with default config
python train.py --dataset ../dataset --output ./output

# Train with custom config
python train.py --dataset ../dataset --config my_config.yaml --output ./output

# Resume from checkpoint
python train.py --dataset ../dataset --output ./output --resume ./output/checkpoint_epoch_50.pt
```

### Export Model for Blender

```bash
# Export to ONNX (recommended)
python export.py --checkpoint ./output/best_model.pt --output gesture_model.onnx --format onnx

# Export to TorchScript
python export.py --checkpoint ./output/best_model.pt --output gesture_model.pt --format torchscript
```

Copy the exported model to `blender_addon/models/` and select it in the addon UI.

## Configuration

### Addon Configuration

Edit `blender_addon/gesture_config.json`:

```json
{
  "bone_mapping": {
    "head": "Head",
    "neck": "Neck"
  },
  "animation_settings": {
    "fps": 24,
    "gesture_intensity": 1.0,
    "smoothing": 0.5
  },
  "audio_processing": {
    "sample_rate": 22050,
    "n_mfcc": 13
  }
}
```

### Training Configuration

Edit `training/training_config.yaml`:

```yaml
model:
  type: "lstm"  # or "simple", "transformer"
  hidden_dim: 256
  num_layers: 3

training:
  batch_size: 16
  learning_rate: 0.001
  num_epochs: 100
```

## How It Works

### Rule-Based Generation

The rule-based system maps audio features to bone movements:

1. **Audio Analysis**: Extract RMS energy, zero-crossing rate, spectral features
2. **Feature Mapping**:
   - High energy → larger gestures
   - Pauses → return to idle pose
   - Emphasis points → accent gestures (head nods, hand movements)
3. **Idle Motion**: Continuous subtle breathing and idle animations
4. **Smoothing**: Temporal smoothing to reduce jitter

### ML-Based Generation

The ML model learns from training data:

1. **Feature Extraction**: MFCC + energy + spectral features from audio
2. **Sequence Model**: LSTM/Transformer maps audio features to bone parameters
3. **Loss Function**: MSE + temporal smoothness + velocity matching
4. **Inference**: Generate frame-by-frame bone rotations and positions

## API Documentation

See [docs/API.md](docs/API.md) for detailed API documentation.

## Troubleshooting

### Addon doesn't appear in Blender
- Make sure you installed the `blender_addon` folder, not individual files
- Check Blender version is 3.6+
- Look for errors in Blender's console (Window > Toggle System Console)

### "librosa not available" error
- Install dependencies: `/path/to/blender/python -m pip install librosa`
- Or use the dependency installer in the addon panel

### Gestures look unnatural
- Increase smoothing parameter
- Adjust gesture intensity
- Check that bone names match expected format
- Try different audio files (clear speech works best)

### Training fails
- Check dataset format matches specification
- Reduce batch size if running out of memory
- Ensure audio and motion files are paired correctly

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built with PyTorch and Librosa
- Inspired by audio-driven animation research
- Blender community for addon development resources

## Citation

If you use this project in research, please cite:

```bibtex
@software{gesture_sync_studio,
  title = {Gesture Sync Studio: Audio-to-Gesture Animation System},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/gesture-sync-studio}
}
```

## Support

- Issues: [GitHub Issues](https://github.com/yourusername/gesture-sync-studio/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/gesture-sync-studio/discussions)
- Email: your.email@example.com
