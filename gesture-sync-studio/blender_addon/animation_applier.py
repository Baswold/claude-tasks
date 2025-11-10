"""
Animation application module for Blender.
Applies generated gesture data to Blender armatures.
"""

import logging
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

try:
    import bpy
    import mathutils
    BLENDER_AVAILABLE = True
except ImportError:
    BLENDER_AVAILABLE = False
    logger.warning("Blender Python API (bpy) not available. This module must run inside Blender.")


class AnimationApplier:
    """Applies generated gestures to Blender armature."""

    def __init__(self, bone_mapping: Optional[Dict[str, str]] = None):
        """
        Initialize animation applier.

        Args:
            bone_mapping: Dictionary mapping standard bone names to armature bone names
                         e.g., {'head': 'Head', 'spine': 'Spine'}
        """
        if not BLENDER_AVAILABLE:
            raise ImportError("This module requires Blender's bpy module")

        self.bone_mapping = bone_mapping or {}

    def validate_armature(self, armature: 'bpy.types.Object') -> tuple[bool, List[str]]:
        """
        Check if armature has required bones.

        Args:
            armature: Blender armature object

        Returns:
            Tuple of (is_valid, list_of_missing_bones)
        """
        if not armature or armature.type != 'ARMATURE':
            logger.error("Invalid armature object")
            return False, []

        # Standard bone names we expect
        expected_bones = [
            'head', 'neck', 'spine',
            'shoulder.L', 'shoulder.R',
            'upper_arm.L', 'upper_arm.R',
            'forearm.L', 'forearm.R',
            'hand.L', 'hand.R'
        ]

        # Get actual bone names in armature
        armature_bones = set(armature.data.bones.keys())

        # Check which bones are missing
        missing_bones = []
        for bone_name in expected_bones:
            # Apply bone mapping if exists
            mapped_name = self.bone_mapping.get(bone_name, bone_name)

            # Check if bone exists (case-insensitive)
            found = False
            for actual_bone in armature_bones:
                if actual_bone.lower() == mapped_name.lower():
                    found = True
                    # Update mapping with exact name
                    self.bone_mapping[bone_name] = actual_bone
                    break

            if not found:
                missing_bones.append(bone_name)

        is_valid = len(missing_bones) == 0

        if is_valid:
            logger.info(f"Armature validation passed: {armature.name}")
        else:
            logger.warning(f"Armature validation failed. Missing bones: {missing_bones}")

        return is_valid, missing_bones

    def get_available_bones(self, armature: 'bpy.types.Object',
                           gesture_bones: Set[str]) -> Set[str]:
        """
        Get intersection of bones in armature and gesture data.

        Args:
            armature: Blender armature object
            gesture_bones: Set of bone names from gesture data

        Returns:
            Set of bone names that exist in both
        """
        armature_bones = set(armature.data.bones.keys())
        available = set()

        for gesture_bone in gesture_bones:
            mapped_name = self.bone_mapping.get(gesture_bone, gesture_bone)

            # Case-insensitive search
            for armature_bone in armature_bones:
                if armature_bone.lower() == mapped_name.lower():
                    available.add(gesture_bone)
                    break

        logger.info(f"Found {len(available)} bones available for animation")
        return available

    def apply_bone_animation(self, armature: 'bpy.types.Object',
                            bone_data: Dict[str, Dict[int, Dict]],
                            fps: int = 24,
                            start_frame: int = 1):
        """
        Create keyframes for each bone based on gesture data.

        Args:
            armature: Blender armature object
            bone_data: Gesture dictionary with bone animations
            fps: Frames per second
            start_frame: Starting frame number in timeline
        """
        if not armature or armature.type != 'ARMATURE':
            logger.error("Invalid armature object")
            return

        # Set scene FPS
        bpy.context.scene.render.fps = fps

        # Get available bones
        available_bones = self.get_available_bones(armature, set(bone_data.keys()))

        if not available_bones:
            logger.error("No matching bones found between armature and gesture data")
            return

        # Switch to pose mode
        bpy.context.view_layer.objects.active = armature
        if armature.mode != 'POSE':
            bpy.ops.object.mode_set(mode='POSE')

        # Clear existing animation (optional - could be made configurable)
        if armature.animation_data and armature.animation_data.action:
            logger.info("Clearing existing animation")
            armature.animation_data.action = None

        # Apply animation to each bone
        for bone_name in available_bones:
            self._apply_single_bone_animation(
                armature, bone_name, bone_data[bone_name], start_frame
            )

        logger.info(f"Applied animation to {len(available_bones)} bones")

    def _apply_single_bone_animation(self, armature: 'bpy.types.Object',
                                     bone_name: str,
                                     frame_data: Dict[int, Dict],
                                     start_frame: int):
        """
        Apply animation to a single bone.

        Args:
            armature: Armature object
            bone_name: Name of bone to animate
            frame_data: Dictionary of frame_index -> transform data
            start_frame: Starting frame
        """
        # Get mapped bone name
        mapped_name = self.bone_mapping.get(bone_name, bone_name)

        # Find actual bone (case-insensitive)
        pose_bone = None
        for bone in armature.pose.bones:
            if bone.name.lower() == mapped_name.lower():
                pose_bone = bone
                break

        if not pose_bone:
            logger.warning(f"Bone not found in armature: {bone_name} (mapped to {mapped_name})")
            return

        # Set rotation mode to quaternion
        pose_bone.rotation_mode = 'QUATERNION'

        # Insert keyframes for each frame
        for frame_idx, transform in sorted(frame_data.items()):
            frame_number = start_frame + frame_idx

            # Set rotation
            if 'rotation_quaternion' in transform:
                quat = transform['rotation_quaternion']
                pose_bone.rotation_quaternion = mathutils.Quaternion(quat)
                pose_bone.keyframe_insert(data_path='rotation_quaternion', frame=frame_number)

            # Set location (if provided)
            if 'location' in transform:
                loc = transform['location']
                # Add to base location rather than replace
                base_loc = pose_bone.location.copy()
                pose_bone.location = mathutils.Vector(loc) + base_loc
                pose_bone.keyframe_insert(data_path='location', frame=frame_number)

        logger.debug(f"Applied {len(frame_data)} keyframes to bone: {bone_name}")

    def smooth_animation(self, armature: 'bpy.types.Object',
                        influence: float = 0.5,
                        iterations: int = 1):
        """
        Apply smoothing/interpolation to reduce jitter using Blender's smoothing.

        Args:
            armature: Armature object
            influence: Smoothing influence (0-1)
            iterations: Number of smoothing iterations
        """
        if not armature or not armature.animation_data or not armature.animation_data.action:
            logger.warning("No animation data to smooth")
            return

        action = armature.animation_data.action

        # Smooth each F-curve
        for fcurve in action.fcurves:
            for _ in range(iterations):
                for keyframe in fcurve.keyframe_points:
                    keyframe.interpolation = 'BEZIER'
                    keyframe.handle_left_type = 'AUTO_CLAMPED'
                    keyframe.handle_right_type = 'AUTO_CLAMPED'

            # Update curve
            fcurve.update()

        logger.info(f"Applied smoothing to animation (influence={influence}, iterations={iterations})")

    def create_action(self, name: str = "GestureAnimation") -> 'bpy.types.Action':
        """
        Create new Action for the animation.

        Args:
            name: Name for the action

        Returns:
            Created Action object
        """
        action = bpy.data.actions.new(name=name)
        logger.info(f"Created new action: {name}")
        return action

    def set_action(self, armature: 'bpy.types.Object', action: 'bpy.types.Action'):
        """
        Set action on armature.

        Args:
            armature: Armature object
            action: Action to apply
        """
        if not armature.animation_data:
            armature.animation_data_create()

        armature.animation_data.action = action
        logger.info(f"Set action '{action.name}' on armature '{armature.name}'")

    def get_animation_duration(self, armature: 'bpy.types.Object') -> Optional[int]:
        """
        Get duration of current animation in frames.

        Args:
            armature: Armature object

        Returns:
            Number of frames, or None if no animation
        """
        if not armature.animation_data or not armature.animation_data.action:
            return None

        action = armature.animation_data.action
        return int(action.frame_range[1] - action.frame_range[0])

    def bake_animation(self, armature: 'bpy.types.Object',
                      start_frame: int = 1,
                      end_frame: Optional[int] = None):
        """
        Bake animation to keyframes on every frame (useful for complex rigs).

        Args:
            armature: Armature object
            start_frame: Start frame
            end_frame: End frame (None = use action duration)
        """
        if not end_frame:
            duration = self.get_animation_duration(armature)
            end_frame = start_frame + (duration if duration else 100)

        # Select armature
        bpy.context.view_layer.objects.active = armature
        armature.select_set(True)

        # Bake action
        bpy.ops.nla.bake(
            frame_start=start_frame,
            frame_end=end_frame,
            only_selected=False,
            visual_keying=True,
            clear_constraints=False,
            clear_parents=False,
            use_current_action=True,
            bake_types={'POSE'}
        )

        logger.info(f"Baked animation from frame {start_frame} to {end_frame}")

    def cleanup_keyframes(self, armature: 'bpy.types.Object', threshold: float = 0.001):
        """
        Remove redundant keyframes to optimize animation.

        Args:
            armature: Armature object
            threshold: Difference threshold for removing keyframes
        """
        if not armature.animation_data or not armature.animation_data.action:
            logger.warning("No animation data to clean")
            return

        action = armature.animation_data.action
        removed_count = 0

        for fcurve in action.fcurves:
            keyframes = fcurve.keyframe_points
            if len(keyframes) <= 2:
                continue

            # Mark keyframes for removal
            to_remove = []
            for i in range(1, len(keyframes) - 1):
                prev_val = keyframes[i - 1].co.y
                curr_val = keyframes[i].co.y
                next_val = keyframes[i + 1].co.y

                # Check if keyframe is redundant (linear interpolation)
                expected = (prev_val + next_val) / 2
                if abs(curr_val - expected) < threshold:
                    to_remove.append(i)

            # Remove marked keyframes (in reverse to maintain indices)
            for i in reversed(to_remove):
                keyframes.remove(keyframes[i])
                removed_count += 1

        logger.info(f"Removed {removed_count} redundant keyframes")

    def export_animation_to_dict(self, armature: 'bpy.types.Object') -> Dict:
        """
        Export current animation to dictionary format.

        Args:
            armature: Armature object

        Returns:
            Animation dictionary
        """
        if not armature.animation_data or not armature.animation_data.action:
            logger.warning("No animation to export")
            return {}

        action = armature.animation_data.action
        export_data = {
            'name': action.name,
            'fps': bpy.context.scene.render.fps,
            'bones': {}
        }

        # Group F-curves by bone
        bone_fcurves = {}
        for fcurve in action.fcurves:
            # Parse data path to get bone name
            # Format: pose.bones["BoneName"].rotation_quaternion
            if 'pose.bones' in fcurve.data_path:
                bone_name = fcurve.data_path.split('"')[1]
                if bone_name not in bone_fcurves:
                    bone_fcurves[bone_name] = []
                bone_fcurves[bone_name].append(fcurve)

        # Extract keyframe data
        for bone_name, fcurves in bone_fcurves.items():
            export_data['bones'][bone_name] = {
                'frames': []
            }

            # Get all frame numbers
            frame_set = set()
            for fcurve in fcurves:
                for kf in fcurve.keyframe_points:
                    frame_set.add(int(kf.co.x))

            # Export each frame
            for frame in sorted(frame_set):
                frame_data = {'frame': frame}

                for fcurve in fcurves:
                    # Evaluate F-curve at this frame
                    value = fcurve.evaluate(frame)

                    # Determine property
                    if 'rotation_quaternion' in fcurve.data_path:
                        if 'rotation_quaternion' not in frame_data:
                            frame_data['rotation_quaternion'] = [0, 0, 0, 0]
                        frame_data['rotation_quaternion'][fcurve.array_index] = value

                    elif 'location' in fcurve.data_path:
                        if 'location' not in frame_data:
                            frame_data['location'] = [0, 0, 0]
                        frame_data['location'][fcurve.array_index] = value

                export_data['bones'][bone_name]['frames'].append(frame_data)

        logger.info(f"Exported animation with {len(bone_fcurves)} bones")
        return export_data
