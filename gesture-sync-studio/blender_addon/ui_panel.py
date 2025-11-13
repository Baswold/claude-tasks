"""
Blender UI panel for Audio Gesture Generator addon.
"""

import bpy
from bpy.props import StringProperty, IntProperty, FloatProperty, BoolProperty, PointerProperty, EnumProperty
from bpy.types import Panel, PropertyGroup
import logging

logger = logging.getLogger(__name__)


class GestureGeneratorProperties(PropertyGroup):
    """Properties for gesture generator settings."""

    # Audio file path
    audio_file: StringProperty(
        name="Audio File",
        description="Path to audio file (WAV, MP3, OGG)",
        default="",
        subtype='FILE_PATH'
    )

    # Armature object
    target_armature: PointerProperty(
        name="Target Armature",
        description="Armature to apply gestures to",
        type=bpy.types.Object,
        poll=lambda self, obj: obj.type == 'ARMATURE'
    )

    # Animation settings
    fps: IntProperty(
        name="FPS",
        description="Frames per second for animation",
        default=24,
        min=1,
        max=120
    )

    start_frame: IntProperty(
        name="Start Frame",
        description="Starting frame for animation",
        default=1,
        min=1
    )

    # Generation mode
    use_rule_based: BoolProperty(
        name="Use Rule-Based",
        description="Use rule-based generation instead of ML model",
        default=True
    )

    model_path: StringProperty(
        name="Model Path",
        description="Path to trained ML model (ONNX or TorchScript)",
        default="",
        subtype='FILE_PATH'
    )

    # Gesture parameters
    gesture_intensity: FloatProperty(
        name="Gesture Intensity",
        description="Overall intensity of gestures",
        default=1.0,
        min=0.0,
        max=3.0
    )

    smoothing: FloatProperty(
        name="Smoothing",
        description="Amount of smoothing to apply",
        default=0.5,
        min=0.0,
        max=1.0
    )

    # Audio processing settings
    energy_threshold: FloatProperty(
        name="Energy Threshold",
        description="Threshold for speech detection",
        default=0.02,
        min=0.001,
        max=0.5
    )

    emphasis_percentile: FloatProperty(
        name="Emphasis Percentile",
        description="Percentile for emphasis detection",
        default=85.0,
        min=50.0,
        max=99.0
    )

    # Advanced settings
    show_advanced: BoolProperty(
        name="Show Advanced",
        description="Show advanced settings",
        default=False
    )


class GESTURE_PT_MainPanel(Panel):
    """Main UI panel in 3D Viewport."""

    bl_label = "Audio Gesture Generator"
    bl_idname = "GESTURE_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Gesture Gen'

    def draw(self, context):
        layout = self.layout
        props = context.scene.gesture_generator_props

        # Audio file selection
        box = layout.box()
        box.label(text="Audio Input:", icon='SOUND')
        box.prop(props, "audio_file")

        # Armature selection
        box = layout.box()
        box.label(text="Target Armature:", icon='ARMATURE_DATA')
        box.prop(props, "target_armature")

        # Validate armature
        if props.target_armature:
            if props.target_armature.type == 'ARMATURE':
                box.label(text="✓ Valid armature", icon='CHECKMARK')
            else:
                box.label(text="✗ Not an armature!", icon='ERROR')

        # Generation settings
        box = layout.box()
        box.label(text="Generation Settings:", icon='SETTINGS')

        row = box.row()
        row.prop(props, "fps")
        row.prop(props, "start_frame")

        box.prop(props, "use_rule_based")

        if not props.use_rule_based:
            box.prop(props, "model_path")
            if props.model_path and not props.model_path.endswith(('.onnx', '.pt', '.pth')):
                box.label(text="Warning: Expected .onnx or .pt file", icon='ERROR')

        # Gesture parameters
        box = layout.box()
        box.label(text="Gesture Parameters:", icon='POSE_HLT')
        box.prop(props, "gesture_intensity", slider=True)
        box.prop(props, "smoothing", slider=True)

        # Advanced settings (collapsible)
        box = layout.box()
        row = box.row()
        row.prop(props, "show_advanced",
                icon='TRIA_DOWN' if props.show_advanced else 'TRIA_RIGHT',
                emboss=False)

        if props.show_advanced:
            box.label(text="Audio Processing:", icon='AUDIO')
            box.prop(props, "energy_threshold")
            box.prop(props, "emphasis_percentile")

        # Generate button
        layout.separator()
        row = layout.row()
        row.scale_y = 2.0
        row.operator("gesture.generate", icon='PLAY')

        # Status/info
        layout.separator()
        box = layout.box()
        box.label(text="Status:", icon='INFO')

        # Check if ready to generate
        ready = True
        if not props.audio_file:
            box.label(text="• No audio file selected", icon='DOT')
            ready = False
        if not props.target_armature:
            box.label(text="• No armature selected", icon='DOT')
            ready = False

        if ready:
            box.label(text="• Ready to generate!", icon='CHECKMARK')


class GESTURE_PT_UtilityPanel(Panel):
    """Utility panel with additional tools."""

    bl_label = "Utilities"
    bl_idname = "GESTURE_PT_utility_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Gesture Gen'
    bl_parent_id = "GESTURE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        props = context.scene.gesture_generator_props

        # Animation utilities
        box = layout.box()
        box.label(text="Animation Tools:", icon='ACTION')

        if props.target_armature:
            box.operator("gesture.smooth_animation", icon='SMOOTHCURVE')
            box.operator("gesture.cleanup_keyframes", icon='TRASH')
            box.operator("gesture.export_animation", icon='EXPORT')
        else:
            box.label(text="Select an armature first", icon='INFO')

        # Audio preview/analysis
        box = layout.box()
        box.label(text="Audio Analysis:", icon='GRAPH')

        if props.audio_file:
            box.operator("gesture.analyze_audio", icon='ANALYZE')
        else:
            box.label(text="Select an audio file first", icon='INFO')


class GESTURE_PT_HelpPanel(Panel):
    """Help panel with instructions and info."""

    bl_label = "Help"
    bl_idname = "GESTURE_PT_help_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Gesture Gen'
    bl_parent_id = "GESTURE_PT_main_panel"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout

        box = layout.box()
        box.label(text="Quick Start:", icon='HELP')
        box.label(text="1. Select an audio file")
        box.label(text="2. Select target armature")
        box.label(text="3. Adjust parameters")
        box.label(text="4. Click 'Generate'")

        layout.separator()

        box = layout.box()
        box.label(text="Expected Bones:", icon='BONE_DATA')
        bones = [
            "head", "neck", "spine",
            "shoulder.L/R", "upper_arm.L/R",
            "forearm.L/R", "hand.L/R"
        ]
        for bone in bones:
            box.label(text=f"• {bone}")

        layout.separator()

        box = layout.box()
        box.label(text="Supported Formats:", icon='FILE')
        box.label(text="Audio: WAV, MP3, OGG")
        box.label(text="Models: ONNX, TorchScript (.pt)")

        layout.separator()

        box = layout.box()
        box.label(text="Tips:", icon='LIGHT')
        box.label(text="• Higher FPS = smoother motion")
        box.label(text="• Lower intensity for subtle gestures")
        box.label(text="• Use smoothing to reduce jitter")
        box.label(text="• Rule-based works without ML model")


# Registration
classes = (
    GestureGeneratorProperties,
    GESTURE_PT_MainPanel,
    GESTURE_PT_UtilityPanel,
    GESTURE_PT_HelpPanel,
)


def register():
    """Register UI classes."""
    for cls in classes:
        bpy.utils.register_class(cls)

    # Add properties to scene
    bpy.types.Scene.gesture_generator_props = PointerProperty(type=GestureGeneratorProperties)

    logger.info("UI panels registered")


def unregister():
    """Unregister UI classes."""
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

    # Remove properties
    del bpy.types.Scene.gesture_generator_props

    logger.info("UI panels unregistered")


if __name__ == "__main__":
    register()
