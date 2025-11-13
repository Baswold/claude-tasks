# Examples

This directory contains example files for getting started with Gesture Sync Studio.

## Contents

### Audio Files
- `sample_audio.wav` - Example speech audio for testing (to be added)
- Various speech samples showing different styles and pacing

### Blender Files
- `rigged_character.blend` - Example rigged character in sitting pose (to be added)
- Pre-configured with proper bone naming

### Generated Animations
- Example outputs from both rule-based and ML generation

## Quick Start Example

1. **Open Blender** with the addon installed
2. **Open** `rigged_character.blend`
3. **Load** `sample_audio.wav` in the addon panel
4. **Select** the armature object
5. **Click** "Generate Gesture Animation"

## Creating Your Own Examples

To create a compatible rigged character:

1. Model a character in sitting position
2. Create an armature with standard bone names:
   - head, neck, spine
   - shoulder.L/R, upper_arm.L/R, forearm.L/R, hand.L/R
3. Weight paint the mesh to the armature
4. Test with sample audio

## Audio Recommendations

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

## Note

Example files are not included in the repository to keep it lightweight.
You can create your own or download examples from the project releases page.
