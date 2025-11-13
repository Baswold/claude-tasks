# Training Guide: Custom Gesture Models

Learn how to train your own audio-to-gesture models for personalized animation styles.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Collection](#data-collection)
3. [Training Process](#training-process)
4. [Model Evaluation](#model-evaluation)
5. [Deployment](#deployment)

## Prerequisites

### Hardware Requirements

**Minimum:**
- CPU: 4+ cores
- RAM: 16GB
- Storage: 50GB for dataset + models
- GPU: Optional but highly recommended

**Recommended:**
- CPU: 8+ cores
- RAM: 32GB
- Storage: 100GB+ SSD
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)

### Software Requirements

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install librosa numpy scipy pyyaml tqdm tensorboard
pip install onnx onnxruntime
```

**Verify GPU access:**
```python
import torch
print(torch.cuda.is_available())  # Should print True if GPU is accessible
```

## Data Collection

You need paired audio and motion data. Minimum recommended: **30 minutes** of high-quality data.

### Method 1: Manual Animation in Blender

**Setup:**
1. Create a rigged character in sitting pose
2. Import or record audio dialogue
3. Manually animate gestures synchronized to audio
4. Export using the data collection script

**Collection Script:**

Save this as `blender_addon/data_collection_helper.py`:

```python
import bpy
import json
from pathlib import Path

def export_animation_for_training(armature, audio_path, output_path, fps=24):
    """
    Export animation as training data.

    Args:
        armature: Blender armature object
        audio_path: Path to audio file
        output_path: Output JSON file path
        fps: Frames per second
    """
    scene = bpy.context.scene
    scene.render.fps = fps

    # Get frame range
    start_frame = scene.frame_start
    end_frame = scene.frame_end

    # Export data
    export_data = {
        "fps": fps,
        "duration": (end_frame - start_frame) / fps,
        "audio_file": audio_path,
        "bones": {}
    }

    # For each bone
    for bone in armature.pose.bones:
        frames_data = []

        # For each frame
        for frame in range(start_frame, end_frame + 1):
            scene.frame_set(frame)

            # Get bone transform
            rotation = bone.rotation_quaternion
            location = bone.location

            frames_data.append({
                "time": (frame - start_frame) / fps,
                "rotation": [rotation.w, rotation.x, rotation.y, rotation.z],
                "location": [location.x, location.y, location.z]
            })

        export_data["bones"][bone.name] = {"frames": frames_data}

    # Save to file
    with open(output_path, 'w') as f:
        json.dump(export_data, f, indent=2)

    print(f"Exported {len(export_data['bones'])} bones, {len(frames_data)} frames")

# Usage in Blender:
# armature = bpy.data.objects["Armature"]
# export_animation_for_training(
#     armature,
#     "/path/to/audio.wav",
#     "/path/to/output/motion_001.json"
# )
```

**Workflow:**
1. Import audio into Blender's video sequencer
2. Animate character's gestures (focus on head, hands, upper body)
3. Run export script for each animation
4. Organize files into dataset folder

### Method 2: Motion Capture

**Using webcam + MediaPipe:**

```python
import cv2
import mediapipe as mp
import json

# Initialize MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Capture video
cap = cv2.VideoCapture(0)

frames_data = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process frame
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        # Extract relevant joint positions
        # Convert to your bone format
        frame_data = extract_bones_from_mediapipe(results.pose_landmarks)
        frames_data.append(frame_data)

    # Show preview
    cv2.imshow('Capture', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()

# Save motion data
# ... convert to required JSON format
```

### Method 3: Existing Motion Capture Data

Use publicly available datasets:

- **CMU Graphics Lab Motion Capture Database**: mocap.cs.cmu.edu
- **AMASS**: amass.is.tue.mpg.de
- **Mixamo**: mixamo.com (requires conversion)

Convert BVH files to JSON format using the data loader's conversion utilities.

### Dataset Organization

```
dataset/
├── audio/
│   ├── sitting_conversation_001.wav
│   ├── sitting_conversation_002.wav
│   ├── presentation_001.wav
│   └── ... (30-100 clips)
├── motion/
│   ├── sitting_conversation_001.json
│   ├── sitting_conversation_002.json
│   ├── presentation_001.json
│   └── ... (matching audio files)
└── metadata.json
```

**metadata.json:**
```json
{
  "samples": [
    {
      "id": "sitting_conversation_001",
      "audio": "sitting_conversation_001.wav",
      "motion": "sitting_conversation_001.json",
      "duration": 12.5,
      "speaker": "person_a",
      "style": "conversational"
    }
  ]
}
```

## Training Process

### Step 1: Prepare Configuration

Edit `training/training_config.yaml`:

```yaml
model:
  type: "lstm"
  input_dim: 29
  hidden_dim: 256
  output_dim: 84
  num_layers: 3
  dropout: 0.1

training:
  batch_size: 8  # Reduce if out of memory
  learning_rate: 0.001
  num_epochs: 100
  validation_split: 0.2

data:
  audio_sr: 22050
  fps: 24
  n_mfcc: 26
```

### Step 2: Start Training

```bash
cd training

# Basic training
python train.py \
    --dataset ../dataset \
    --config training_config.yaml \
    --output ./output

# With custom settings
python train.py \
    --dataset ../dataset \
    --config my_config.yaml \
    --output ./experiment_1 \
    --gpu 0
```

### Step 3: Monitor Progress

**Using TensorBoard:**
```bash
tensorboard --logdir=./output/runs
```

Open `http://localhost:6006` in browser to see:
- Training/validation loss curves
- Learning rate schedule
- Loss component breakdown

**Expected training time:**
- Small dataset (30 min): 2-4 hours on GPU
- Medium dataset (2 hours): 8-12 hours on GPU
- Large dataset (10+ hours): 1-3 days on GPU

### Step 4: Resume Training (if interrupted)

```bash
python train.py \
    --dataset ../dataset \
    --config training_config.yaml \
    --output ./output \
    --resume ./output/checkpoint_epoch_50.pt
```

## Model Evaluation

### Quantitative Metrics

After training, check the output logs:

```
Epoch 99:
  Train loss: 0.0123
  Val loss: 0.0156
```

**Good loss values:**
- Total loss < 0.02: Excellent
- Total loss < 0.05: Good
- Total loss < 0.10: Acceptable
- Total loss > 0.10: May need more training/data

### Qualitative Evaluation

**Visual Inspection:**
1. Export the trained model
2. Use it in Blender addon
3. Generate animations from test audio
4. Check for:
   - Natural-looking gestures
   - Smooth transitions
   - Appropriate synchronization with audio
   - Variety in movements

**Test Cases:**
- Calm, conversational speech
- Excited, emphatic speech
- Pauses and silence
- Different speakers
- Various speaking speeds

### Comparison

Generate animations using:
1. Rule-based generation
2. Your trained model

Compare:
- Naturalness
- Variety
- Synchronization quality
- Jitter/smoothness

## Deployment

### Step 1: Export Model

```bash
# Export to ONNX (recommended for Blender)
python export.py \
    --checkpoint ./output/best_model.pt \
    --output gesture_model.onnx \
    --format onnx

# Or TorchScript
python export.py \
    --checkpoint ./output/best_model.pt \
    --output gesture_model.pt \
    --format torchscript
```

### Step 2: Install in Blender

1. Copy exported model to `blender_addon/models/`
2. Open Blender
3. In addon panel, uncheck "Use Rule-Based"
4. Set "Model Path" to your exported model
5. Generate animation to test

### Step 3: Performance Optimization

If model is too slow:

**Reduce model size:**
```yaml
# In training_config.yaml
model:
  hidden_dim: 128  # Instead of 256
  num_layers: 2    # Instead of 3
```

**Use simpler architecture:**
```yaml
model:
  type: "simple"  # Instead of "lstm" or "transformer"
```

**Quantization** (advanced):
```python
import torch

# Load model
model = torch.load("gesture_model.pt")

# Quantize
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# Save
torch.jit.save(quantized_model, "gesture_model_quantized.pt")
```

## Advanced Topics

### Transfer Learning

Start from a pre-trained model:

```bash
python train.py \
    --dataset ../dataset \
    --config training_config.yaml \
    --output ./fine_tuned \
    --resume ./pretrained_model/best_model.pt
```

Reduce learning rate for fine-tuning:
```yaml
training:
  learning_rate: 0.0001  # Lower than default 0.001
```

### Data Augmentation

Modify `training_config.yaml`:
```yaml
augmentation:
  enabled: true
  time_stretch: true      # Vary speech speed
  pitch_shift: true       # Vary voice pitch
  noise_injection: true   # Add background noise
```

### Multi-Style Models

Train separate models for different styles:
- Conversational (subtle gestures)
- Presentation (emphatic gestures)
- Excited (large, energetic gestures)

Or use style conditioning (advanced - requires code modification).

### Ensemble Models

Combine multiple models:
1. Train 3-5 models with different random seeds
2. Average their predictions at inference time
3. Results in smoother, more robust animations

## Troubleshooting

### Out of Memory

**Solutions:**
- Reduce batch_size (try 4, 2, or even 1)
- Reduce hidden_dim (try 128 instead of 256)
- Use gradient accumulation
- Use mixed precision training (if GPU supports it)

### Loss not decreasing

**Solutions:**
- Check data quality (audio and motion properly aligned?)
- Increase learning rate (try 0.01)
- Reduce learning rate (try 0.0001)
- Add more training data
- Simplify model (fewer layers)
- Increase model capacity (more hidden_dim)

### Overfitting (training loss low, validation loss high)

**Solutions:**
- Add more training data
- Increase dropout (try 0.2-0.3)
- Add data augmentation
- Reduce model size
- Early stopping (stop when validation loss stops improving)

### Gestures look unnatural

**Solutions:**
- Increase smoothness_weight in loss function
- Add more diverse training data
- Check that training data has natural gestures
- Increase temporal context (use LSTM/Transformer instead of simple MLP)

## Tips for Success

1. **Start small**: Train on 30 min of data first, expand later
2. **Quality over quantity**: 1 hour of high-quality data > 10 hours of poor data
3. **Diverse data**: Include various speaking styles, speeds, emotions
4. **Monitor training**: Use TensorBoard, check for overfitting
5. **Iterate**: Try different model architectures and hyperparameters
6. **Document**: Keep notes on what works and what doesn't

## Next Steps

Once you have a working model:

1. Create a model zoo with different styles
2. Share your model with the community
3. Contribute improvements back to the project
4. Experiment with advanced architectures (Transformers, GANs)
5. Combine with other animation techniques (facial animation, finger animation)

## Resources

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Librosa Documentation](https://librosa.org/doc/latest/)
- [Deep Learning for Animation](https://research.adobe.com/machine-learning/)
- [Project GitHub](https://github.com/yourusername/gesture-sync-studio)
