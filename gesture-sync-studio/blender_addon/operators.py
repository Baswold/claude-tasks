"""
Blender operators for Audio Gesture Generator.
"""

import bpy
from bpy.types import Operator
import logging
import os
import json

logger = logging.getLogger(__name__)

# Import addon modules (will work when loaded as addon)
try:
    from . import audio_processor
    from . import gesture_generator
    from . import animation_applier
except ImportError:
    # Fallback for development
    import audio_processor
    import gesture_generator
    import animation_applier


class GESTURE_OT_Generate(Operator):
    """Generate gesture animation from audio."""

    bl_idname = "gesture.generate"
    bl_label = "Generate Gesture Animation"
    bl_description = "Generate gesture animation from audio file"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        """Execute the gesture generation."""
        props = context.scene.gesture_generator_props

        # Validate inputs
        if not props.audio_file or not os.path.exists(props.audio_file):
            self.report({'ERROR'}, "Please select a valid audio file")
            return {'CANCELLED'}

        if not props.target_armature or props.target_armature.type != 'ARMATURE':
            self.report({'ERROR'}, "Please select a valid armature")
            return {'CANCELLED'}

        try:
            # Step 1: Load and process audio
            self.report({'INFO'}, "Loading audio file...")
            processor = audio_processor.AudioProcessor()
            waveform, sr = processor.load_audio(props.audio_file)

            # Step 2: Extract features
            self.report({'INFO'}, "Extracting audio features...")
            features = processor.extract_features(waveform, sr)

            # Step 3: Interpolate to target FPS
            features = processor.interpolate_to_fps(features, props.fps)

            # Step 4: Generate gestures
            self.report({'INFO'}, "Generating gestures...")

            # Load configuration
            config = {
                'gesture_intensity': props.gesture_intensity,
                'smoothing': props.smoothing,
                'idle_motion_scale': 0.3,
                'emphasis_scale': 1.5,
                'breathing_rate': 0.2,
                'head_nod_threshold': 0.7,
                'hand_gesture_threshold': 0.6
            }

            # Initialize generator
            model_path = None if props.use_rule_based else props.model_path
            generator = gesture_generator.GestureGenerator(model_path=model_path, config=config)

            # Generate gesture sequence
            gestures = generator.generate_gesture_sequence(
                audio_features=features,
                fps=props.fps
            )

            # Apply smoothing if requested
            if props.smoothing > 0:
                window_size = max(3, int(props.smoothing * 10))
                gestures = generator.smooth_gesture_sequence(gestures, window_size=window_size)

            # Step 5: Apply to armature
            self.report({'INFO'}, "Applying animation to armature...")

            applier = animation_applier.AnimationApplier()

            # Validate armature
            is_valid, missing_bones = applier.validate_armature(props.target_armature)
            if missing_bones:
                self.report({'WARNING'},
                           f"Some bones not found: {', '.join(missing_bones[:5])}")

            # Apply animation
            applier.apply_bone_animation(
                armature=props.target_armature,
                bone_data=gestures,
                fps=props.fps,
                start_frame=props.start_frame
            )

            # Calculate duration
            num_frames = len(features['times'])
            duration = num_frames / props.fps

            self.report({'INFO'},
                       f"Animation generated successfully! Duration: {duration:.2f}s ({num_frames} frames)")

            return {'FINISHED'}

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            self.report({'ERROR'}, f"Generation failed: {str(e)}")
            return {'CANCELLED'}


class GESTURE_OT_SmoothAnimation(Operator):
    """Smooth existing animation to reduce jitter."""

    bl_idname = "gesture.smooth_animation"
    bl_label = "Smooth Animation"
    bl_description = "Apply smoothing to existing animation"
    bl_options = {'REGISTER', 'UNDO'}

    influence: bpy.props.FloatProperty(
        name="Influence",
        description="Smoothing influence",
        default=0.5,
        min=0.0,
        max=1.0
    )

    iterations: bpy.props.IntProperty(
        name="Iterations",
        description="Number of smoothing passes",
        default=1,
        min=1,
        max=10
    )

    def execute(self, context):
        """Execute smoothing."""
        props = context.scene.gesture_generator_props

        if not props.target_armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        try:
            applier = animation_applier.AnimationApplier()
            applier.smooth_animation(
                armature=props.target_armature,
                influence=self.influence,
                iterations=self.iterations
            )

            self.report({'INFO'}, "Animation smoothed successfully")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Smoothing failed: {str(e)}")
            return {'CANCELLED'}

    def invoke(self, context, event):
        """Show dialog for parameters."""
        return context.window_manager.invoke_props_dialog(self)


class GESTURE_OT_CleanupKeyframes(Operator):
    """Remove redundant keyframes to optimize animation."""

    bl_idname = "gesture.cleanup_keyframes"
    bl_label = "Cleanup Keyframes"
    bl_description = "Remove redundant keyframes"
    bl_options = {'REGISTER', 'UNDO'}

    threshold: bpy.props.FloatProperty(
        name="Threshold",
        description="Difference threshold for removing keyframes",
        default=0.001,
        min=0.0001,
        max=0.1
    )

    def execute(self, context):
        """Execute cleanup."""
        props = context.scene.gesture_generator_props

        if not props.target_armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        try:
            applier = animation_applier.AnimationApplier()
            applier.cleanup_keyframes(
                armature=props.target_armature,
                threshold=self.threshold
            )

            self.report({'INFO'}, "Keyframes cleaned up successfully")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Cleanup failed: {str(e)}")
            return {'CANCELLED'}

    def invoke(self, context, event):
        """Show dialog for parameters."""
        return context.window_manager.invoke_props_dialog(self)


class GESTURE_OT_ExportAnimation(Operator):
    """Export animation to JSON file."""

    bl_idname = "gesture.export_animation"
    bl_label = "Export Animation"
    bl_description = "Export animation to JSON file"

    filepath: bpy.props.StringProperty(
        subtype='FILE_PATH',
        default="gesture_animation.json"
    )

    def execute(self, context):
        """Execute export."""
        props = context.scene.gesture_generator_props

        if not props.target_armature:
            self.report({'ERROR'}, "No armature selected")
            return {'CANCELLED'}

        try:
            applier = animation_applier.AnimationApplier()
            anim_data = applier.export_animation_to_dict(props.target_armature)

            with open(self.filepath, 'w') as f:
                json.dump(anim_data, f, indent=2)

            self.report({'INFO'}, f"Animation exported to {self.filepath}")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Export failed: {str(e)}")
            return {'CANCELLED'}

    def invoke(self, context, event):
        """Show file browser."""
        context.window_manager.fileselect_add(self)
        return {'RUNNING_MODAL'}


class GESTURE_OT_AnalyzeAudio(Operator):
    """Analyze audio file and show statistics."""

    bl_idname = "gesture.analyze_audio"
    bl_label = "Analyze Audio"
    bl_description = "Analyze audio file and show statistics"

    def execute(self, context):
        """Execute analysis."""
        props = context.scene.gesture_generator_props

        if not props.audio_file or not os.path.exists(props.audio_file):
            self.report({'ERROR'}, "Please select a valid audio file")
            return {'CANCELLED'}

        try:
            # Load and analyze audio
            processor = audio_processor.AudioProcessor()
            waveform, sr = processor.load_audio(props.audio_file)
            features = processor.extract_features(waveform, sr)

            # Detect segments and emphasis
            segments = processor.detect_speech_segments(
                features,
                energy_threshold=props.energy_threshold
            )
            emphasis_points = processor.detect_emphasis_points(
                features,
                percentile=props.emphasis_percentile
            )

            # Calculate statistics
            duration = len(waveform) / sr
            num_frames = len(features['times'])
            frame_rate = processor.get_frame_rate(features)

            # Report statistics
            self.report({'INFO'}, f"Audio Analysis:")
            self.report({'INFO'}, f"Duration: {duration:.2f}s")
            self.report({'INFO'}, f"Sample rate: {sr} Hz")
            self.report({'INFO'}, f"Feature frames: {num_frames}")
            self.report({'INFO'}, f"Frame rate: {frame_rate:.2f} FPS")
            self.report({'INFO'}, f"Speech segments: {len(segments)}")
            self.report({'INFO'}, f"Emphasis points: {len(emphasis_points)}")

            # Log detailed info
            logger.info(f"Audio file: {props.audio_file}")
            logger.info(f"Duration: {duration:.2f}s, {num_frames} frames @ {frame_rate:.2f} FPS")
            logger.info(f"Speech segments: {len(segments)}")
            logger.info(f"Emphasis points: {len(emphasis_points)}")

            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Analysis failed: {str(e)}")
            logger.error(f"Audio analysis failed: {e}", exc_info=True)
            return {'CANCELLED'}


# Registration
classes = (
    GESTURE_OT_Generate,
    GESTURE_OT_SmoothAnimation,
    GESTURE_OT_CleanupKeyframes,
    GESTURE_OT_ExportAnimation,
    GESTURE_OT_AnalyzeAudio,
)


def register():
    """Register operators."""
    for cls in classes:
        bpy.utils.register_class(cls)

    logger.info("Operators registered")


def unregister():
    """Unregister operators."""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    logger.info("Operators unregistered")


if __name__ == "__main__":
    register()
