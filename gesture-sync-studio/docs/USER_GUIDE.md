# User Guide: Gesture Sync Studio

Complete guide to using the Audio Gesture Generator Blender addon.

## Table of Contents

1. [Installation](#installation)
2. [Interface Overview](#interface-overview)
3. [Basic Workflow](#basic-workflow)
4. [Advanced Features](#advanced-features)
5. [Tips & Best Practices](#tips--best-practices)
6. [Troubleshooting](#troubleshooting)

## Installation

### Step 1: Install Blender

- Download Blender 3.6 or higher from [blender.org](https://www.blender.org)
- Install and launch Blender

### Step 2: Install the Addon

1. Download the `blender_addon` folder from the project
2. In Blender, go to **Edit → Preferences → Add-ons**
3. Click **Install** button
4. Navigate to the `blender_addon` folder and select it
5. Enable the addon by checking the box next to "Audio Gesture Generator"

### Step 3: Install Dependencies

The addon will show a "Dependencies" panel if required packages are missing.

**Option A: Automatic Installation**
1. In the Dependencies panel, click **Install Dependencies**
2. Wait for installation to complete
3. Restart Blender

**Option B: Manual Installation**
```bash
# Find your Blender Python path
# Windows: C:\Program Files\Blender Foundation\Blender X.X\X.X\python\bin\python.exe
# Mac: /Applications/Blender.app/Contents/Resources/X.X/python/bin/python3.X
# Linux: /usr/bin/blender-python

# Install packages
/path/to/blender/python -m pip install numpy librosa scipy numba
```

## Interface Overview

The addon panel appears in the **3D Viewport → Sidebar (press N) → Gesture Gen tab**.

### Main Panel Sections

1. **Audio Input**
   - File picker for selecting audio files (WAV, MP3, OGG)

2. **Target Armature**
   - Select the character rig to animate
   - Shows validation status

3. **Generation Settings**
   - **FPS**: Animation frame rate (default: 24)
   - **Start Frame**: Where to begin the animation in timeline
   - **Use Rule-Based**: Toggle between rule-based and ML generation
   - **Model Path**: Select trained ML model (if not using rule-based)

4. **Gesture Parameters**
   - **Gesture Intensity**: Controls how pronounced gestures are (0-3)
   - **Smoothing**: Amount of motion smoothing (0-1)

5. **Advanced Settings** (expandable)
   - **Energy Threshold**: Sensitivity for speech detection
   - **Emphasis Percentile**: Threshold for detecting emphasis in speech

### Utility Panel

Additional tools for working with generated animations:

- **Smooth Animation**: Apply additional smoothing to existing animation
- **Cleanup Keyframes**: Remove redundant keyframes to optimize
- **Export Animation**: Save animation to JSON file
- **Analyze Audio**: View audio statistics and analysis

### Help Panel

Quick reference for:
- Expected bone names
- Supported file formats
- Usage tips

## Basic Workflow

### Step-by-Step Guide

#### 1. Prepare Your Character

Ensure your character has an armature with these bones:
- **Core**: head, neck, spine (possibly spine.001, spine.002)
- **Arms (L/R)**: shoulder, upper_arm, forearm, hand
- **Sitting pose**: Character should be in sitting position

If your bone names are different, you can customize the mapping in `gesture_config.json`.

#### 2. Select Audio

1. Click the folder icon in **Audio Input**
2. Navigate to your audio file
3. Select a clear speech recording (WAV or MP3)
4. For best results:
   - Clear speech without heavy background noise
   - Natural speaking pace
   - Good recording quality

#### 3. Select Armature

1. In **Target Armature**, click the picker
2. Select your character's armature object
3. Look for green checkmark (✓ Valid armature)
4. If you see errors about missing bones, check bone naming

#### 4. Adjust Parameters

**For Subtle Gestures:**
- Gesture Intensity: 0.5 - 0.8
- Smoothing: 0.6 - 0.8

**For Expressive Gestures:**
- Gesture Intensity: 1.2 - 2.0
- Smoothing: 0.3 - 0.5

**FPS Recommendations:**
- 24 FPS: Standard animation
- 30 FPS: Smoother motion
- 60 FPS: Very smooth (but larger file size)

#### 5. Generate Animation

1. Click **Generate Gesture Animation** button
2. Wait for processing (you'll see progress in the console)
3. When complete, you'll see a success message
4. Play back the animation in Blender's timeline

#### 6. Refine (Optional)

- Adjust parameters and regenerate if needed
- Use **Smooth Animation** for additional smoothing
- Use **Cleanup Keyframes** to reduce file size
- Manually tweak specific keyframes as needed

## Advanced Features

### Using ML Models

If you've trained a custom model:

1. Uncheck **Use Rule-Based**
2. Set **Model Path** to your `.onnx` or `.pt` file
3. Generate as normal

ML models can produce more natural and varied gestures but require training data.

### Custom Bone Mapping

Edit `gesture_config.json` in the addon folder:

```json
{
  "bone_mapping": {
    "head": "MyHead",
    "neck": "MyNeck",
    "spine": "MySpine"
  }
}
```

Map standard names (left side) to your rig's bone names (right side).

### Batch Processing

For multiple characters or audio files:

1. Create a Python script in Blender's text editor
2. Use the operator programmatically:

```python
import bpy

# Set properties
props = bpy.context.scene.gesture_generator_props
props.audio_file = "/path/to/audio.wav"
props.target_armature = bpy.data.objects["Armature"]
props.fps = 24
props.gesture_intensity = 1.0

# Generate
bpy.ops.gesture.generate()
```

### Exporting Animations

**Export to JSON:**
1. Generate animation
2. Click **Export Animation** in Utilities panel
3. Choose save location
4. Use JSON for data analysis or reimport

**Export to other formats:**
- Use Blender's built-in export (FBX, Alembic, etc.)
- Animation will be included with armature

## Tips & Best Practices

### Audio Quality

✅ **Good:**
- Clear speech recordings
- Minimal background noise
- Natural speaking pace
- Consistent volume

❌ **Avoid:**
- Heavy music background
- Multiple speakers talking over each other
- Very fast or mumbled speech
- Extreme volume fluctuations

### Character Setup

✅ **Best Results:**
- Character in sitting position
- Arms in neutral pose (hands on lap or armrests)
- Well-rigged armature with proper weights
- Standard bone hierarchy

⚠️ **Common Issues:**
- Character standing (sitting gestures won't look right)
- Poor weight painting
- IK constraints conflicting with keyframes

### Performance

**For Long Audio (10+ minutes):**
- Use lower FPS (24 instead of 60)
- Process in chunks
- Consider using lower smoothing values

**Memory Usage:**
- Large audio files: Process sections separately
- Many characters: Generate one at a time

### Artistic Direction

**Combine with Manual Animation:**
1. Generate base gestures
2. Add manual keyframes for specific moments
3. Use NLA editor to blend generated and manual animation

**Vary Gesture Intensity:**
- Low intensity for calm speech
- High intensity for excited/emphatic speech
- Adjust per section using multiple generations

## Troubleshooting

### Problem: "No audio file selected" error

**Solution:**
- Make sure you've selected a valid audio file
- Check file format is WAV, MP3, or OGG
- Verify file path is accessible

### Problem: "Invalid armature" error

**Solution:**
- Ensure you've selected an Armature object, not a mesh
- Check in outliner that object type is ARMATURE
- Select the armature, not the character mesh

### Problem: Gestures look robotic/jerky

**Solutions:**
- Increase **Smoothing** parameter (try 0.7-0.9)
- Use **Smooth Animation** utility
- Lower **Gesture Intensity** for subtler motion
- Check FPS isn't too low (use 30+ for smoother results)

### Problem: No movement generated

**Solutions:**
- Check audio actually has sound (not silent)
- Increase **Gesture Intensity**
- Lower **Energy Threshold** in Advanced settings
- Verify bone names match expected format

### Problem: Missing bones warning

**Solutions:**
- Check bone naming matches expectations (see Help panel)
- Create custom bone mapping in config file
- Add missing bones to your rig
- Some bones are optional (hands, multiple spine bones)

### Problem: Animation too long/short

**Solutions:**
- Check audio duration matches expected length
- Verify FPS setting is correct
- Check start frame is set properly
- Audio might have long silence at start/end

### Problem: Out of memory error

**Solutions:**
- Close other Blender scenes
- Process shorter audio clips
- Reduce batch size (if using ML model)
- Restart Blender

### Problem: Slow generation

**Expected times:**
- Rule-based: ~10-30 seconds for 1 minute of audio
- ML-based: ~30-90 seconds for 1 minute of audio
- Very long (5+ minutes): Several minutes

**If unusually slow:**
- Check CPU usage (should be high during generation)
- Close other programs
- For ML: Check if GPU is being used

## Keyboard Shortcuts

While in 3D Viewport:
- **N**: Toggle sidebar (show/hide addon panel)
- **Space**: Play/pause animation
- **Shift + Left/Right**: Jump to next/previous keyframe
- **G**: Grab/move selected bone (for manual adjustment)

## Additional Resources

- [Training Guide](TRAINING_GUIDE.md) - How to train custom models
- [API Documentation](API.md) - For developers
- [GitHub Issues](https://github.com/yourusername/gesture-sync-studio/issues) - Report bugs
- [Examples](../examples/) - Sample audio and animations

## Getting Help

If you encounter issues:

1. Check this guide's Troubleshooting section
2. Look at Blender's console (Window → Toggle System Console)
3. Check GitHub Issues for similar problems
4. Create a new issue with:
   - Error message
   - Blender version
   - Steps to reproduce
   - Audio file characteristics (if relevant)
