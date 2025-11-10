"""
Audio Gesture Generator - Blender Addon
Generates realistic sitting/gesturing animations from audio input.
"""

bl_info = {
    "name": "Audio Gesture Generator",
    "author": "Gesture Sync Studio",
    "version": (1, 0, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar > Gesture Gen",
    "description": "Generate realistic sitting gestures and animations from audio files",
    "warning": "Requires librosa and numpy (install via pip)",
    "doc_url": "https://github.com/yourusername/gesture-sync-studio",
    "category": "Animation",
}

import bpy
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add addon directory to path for imports
addon_dir = Path(__file__).parent
if str(addon_dir) not in sys.path:
    sys.path.insert(0, str(addon_dir))


# Check for required dependencies
DEPENDENCIES = {
    'numpy': 'numpy',
    'librosa': 'librosa',
}

missing_dependencies = []
for module_name, import_name in DEPENDENCIES.items():
    try:
        __import__(import_name)
    except ImportError:
        missing_dependencies.append(module_name)
        logger.warning(f"Missing dependency: {module_name}")


# Import addon modules
from . import ui_panel
from . import operators
from . import audio_processor
from . import gesture_generator
from . import animation_applier


# Module list for registration
modules = (
    ui_panel,
    operators,
)


class GESTURE_OT_InstallDependencies(bpy.types.Operator):
    """Install required dependencies via pip."""

    bl_idname = "gesture.install_dependencies"
    bl_label = "Install Dependencies"
    bl_description = "Install required Python packages (numpy, librosa)"

    def execute(self, context):
        """Execute dependency installation."""
        import subprocess
        import sys

        python_exe = sys.executable

        try:
            self.report({'INFO'}, "Installing dependencies...")

            # Install packages
            packages = ['numpy', 'librosa', 'scipy', 'numba']

            for package in packages:
                self.report({'INFO'}, f"Installing {package}...")
                subprocess.check_call([python_exe, "-m", "pip", "install", package])

            self.report({'INFO'}, "Dependencies installed successfully! Please restart Blender.")
            return {'FINISHED'}

        except Exception as e:
            self.report({'ERROR'}, f"Installation failed: {str(e)}")
            logger.error(f"Dependency installation failed: {e}", exc_info=True)
            return {'CANCELLED'}


class GESTURE_PT_DependencyPanel(bpy.types.Panel):
    """Panel to show dependency status and installation."""

    bl_label = "Dependencies"
    bl_idname = "GESTURE_PT_dependency_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Gesture Gen'
    bl_order = 0

    def draw(self, context):
        layout = self.layout

        if missing_dependencies:
            box = layout.box()
            box.label(text="Missing Dependencies:", icon='ERROR')

            for dep in missing_dependencies:
                box.label(text=f"• {dep}")

            layout.separator()
            layout.operator("gesture.install_dependencies", icon='IMPORT')

            layout.separator()
            box = layout.box()
            box.label(text="Manual Installation:", icon='INFO')
            box.label(text="Run in terminal:")
            box.label(text=f"{sys.executable} -m pip install librosa numpy")

        else:
            box = layout.box()
            box.label(text="All dependencies installed", icon='CHECKMARK')


def register():
    """Register addon classes and properties."""

    logger.info(f"Registering {bl_info['name']} v{bl_info['version']}")

    # Register dependency installer
    bpy.utils.register_class(GESTURE_OT_InstallDependencies)
    bpy.utils.register_class(GESTURE_PT_DependencyPanel)

    # Register main modules
    for module in modules:
        try:
            module.register()
        except Exception as e:
            logger.error(f"Failed to register module {module.__name__}: {e}")

    logger.info(f"{bl_info['name']} registered successfully")

    # Show warning if dependencies missing
    if missing_dependencies:
        logger.warning(f"Missing dependencies: {', '.join(missing_dependencies)}")
        logger.warning("Addon may not function correctly until dependencies are installed")


def unregister():
    """Unregister addon classes and properties."""

    logger.info(f"Unregistering {bl_info['name']}")

    # Unregister main modules (in reverse order)
    for module in reversed(modules):
        try:
            module.unregister()
        except Exception as e:
            logger.error(f"Failed to unregister module {module.__name__}: {e}")

    # Unregister dependency installer
    bpy.utils.unregister_class(GESTURE_PT_DependencyPanel)
    bpy.utils.unregister_class(GESTURE_OT_InstallDependencies)

    logger.info(f"{bl_info['name']} unregistered")


if __name__ == "__main__":
    register()
