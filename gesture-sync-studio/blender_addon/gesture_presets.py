"""
Gesture preset library for quick configuration and style templates.

Provides pre-configured gesture styles for different scenarios:
- Presentation/Teaching
- Casual conversation
- Emphatic/Energetic
- Subtle/Reserved
- Custom presets
"""

import json
from typing import Dict, List, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class GesturePreset:
    """
    Represents a gesture generation preset with specific parameters.
    """

    def __init__(
        self,
        name: str,
        description: str,
        config: Dict,
        tags: Optional[List[str]] = None
    ):
        """
        Initialize preset.

        Args:
            name: Preset name
            description: Human-readable description
            config: Configuration dictionary
            tags: Optional tags for categorization
        """
        self.name = name
        self.description = description
        self.config = config
        self.tags = tags or []

    def to_dict(self) -> Dict:
        """Convert preset to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'config': self.config,
            'tags': self.tags
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'GesturePreset':
        """Create preset from dictionary."""
        return cls(
            name=data['name'],
            description=data['description'],
            config=data['config'],
            tags=data.get('tags', [])
        )


# Built-in presets
BUILTIN_PRESETS = {
    'presentation': GesturePreset(
        name='Presentation/Teaching',
        description='Confident, clear gestures suitable for presentations and teaching',
        config={
            'gesture_intensity': 1.3,
            'smoothing': 0.6,
            'idle_motion_scale': 0.2,
            'emphasis_scale': 1.8,
            'breathing_rate': 0.18,
            'head_nod_threshold': 0.6,
            'hand_gesture_threshold': 0.5
        },
        tags=['professional', 'teaching', 'formal']
    ),

    'casual': GesturePreset(
        name='Casual Conversation',
        description='Natural, relaxed gestures for everyday conversation',
        config={
            'gesture_intensity': 1.0,
            'smoothing': 0.5,
            'idle_motion_scale': 0.3,
            'emphasis_scale': 1.3,
            'breathing_rate': 0.2,
            'head_nod_threshold': 0.7,
            'hand_gesture_threshold': 0.6
        },
        tags=['casual', 'conversation', 'natural']
    ),

    'emphatic': GesturePreset(
        name='Emphatic/Energetic',
        description='Larger, more pronounced gestures for energetic speech',
        config={
            'gesture_intensity': 1.8,
            'smoothing': 0.4,
            'idle_motion_scale': 0.4,
            'emphasis_scale': 2.2,
            'breathing_rate': 0.22,
            'head_nod_threshold': 0.5,
            'hand_gesture_threshold': 0.4
        },
        tags=['energetic', 'emphatic', 'expressive']
    ),

    'subtle': GesturePreset(
        name='Subtle/Reserved',
        description='Minimal, controlled gestures for formal or reserved scenarios',
        config={
            'gesture_intensity': 0.6,
            'smoothing': 0.7,
            'idle_motion_scale': 0.15,
            'emphasis_scale': 0.9,
            'breathing_rate': 0.16,
            'head_nod_threshold': 0.8,
            'hand_gesture_threshold': 0.75
        },
        tags=['subtle', 'formal', 'reserved']
    ),

    'interview': GesturePreset(
        name='Interview/Discussion',
        description='Professional gestures with moderate intensity',
        config={
            'gesture_intensity': 1.1,
            'smoothing': 0.6,
            'idle_motion_scale': 0.25,
            'emphasis_scale': 1.4,
            'breathing_rate': 0.19,
            'head_nod_threshold': 0.65,
            'hand_gesture_threshold': 0.55
        },
        tags=['professional', 'interview', 'discussion']
    ),

    'storytelling': GesturePreset(
        name='Storytelling/Narrative',
        description='Expressive gestures for storytelling and dramatic narration',
        config={
            'gesture_intensity': 1.5,
            'smoothing': 0.45,
            'idle_motion_scale': 0.35,
            'emphasis_scale': 2.0,
            'breathing_rate': 0.21,
            'head_nod_threshold': 0.55,
            'hand_gesture_threshold': 0.45
        },
        tags=['storytelling', 'narrative', 'expressive']
    ),

    'meditation': GesturePreset(
        name='Meditation/Calm Speech',
        description='Very minimal gestures with focus on breathing',
        config={
            'gesture_intensity': 0.4,
            'smoothing': 0.8,
            'idle_motion_scale': 0.1,
            'emphasis_scale': 0.6,
            'breathing_rate': 0.12,
            'head_nod_threshold': 0.85,
            'hand_gesture_threshold': 0.85
        },
        tags=['calm', 'meditation', 'minimal']
    ),

    'debate': GesturePreset(
        name='Debate/Argument',
        description='Strong, decisive gestures for debates and arguments',
        config={
            'gesture_intensity': 1.6,
            'smoothing': 0.4,
            'idle_motion_scale': 0.2,
            'emphasis_scale': 2.3,
            'breathing_rate': 0.23,
            'head_nod_threshold': 0.5,
            'hand_gesture_threshold': 0.4
        },
        tags=['debate', 'argument', 'strong']
    )
}


class PresetLibrary:
    """
    Manages gesture presets and custom user presets.
    """

    def __init__(self, custom_presets_path: Optional[str] = None):
        """
        Initialize preset library.

        Args:
            custom_presets_path: Path to custom presets JSON file
        """
        self.builtin_presets = BUILTIN_PRESETS.copy()
        self.custom_presets = {}
        self.custom_presets_path = Path(custom_presets_path) if custom_presets_path else None

        if self.custom_presets_path and self.custom_presets_path.exists():
            self.load_custom_presets()

    def load_custom_presets(self):
        """Load custom presets from file."""
        if not self.custom_presets_path:
            return

        try:
            with open(self.custom_presets_path, 'r') as f:
                data = json.load(f)

            for preset_data in data.get('presets', []):
                preset = GesturePreset.from_dict(preset_data)
                self.custom_presets[preset.name.lower().replace(' ', '_')] = preset

            logger.info(f"Loaded {len(self.custom_presets)} custom presets")

        except Exception as e:
            logger.error(f"Failed to load custom presets: {e}")

    def save_custom_presets(self):
        """Save custom presets to file."""
        if not self.custom_presets_path:
            logger.warning("No custom presets path configured")
            return

        try:
            data = {
                'presets': [preset.to_dict() for preset in self.custom_presets.values()]
            }

            with open(self.custom_presets_path, 'w') as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved {len(self.custom_presets)} custom presets")

        except Exception as e:
            logger.error(f"Failed to save custom presets: {e}")

    def get_preset(self, name: str) -> Optional[GesturePreset]:
        """
        Get preset by name.

        Args:
            name: Preset name (case-insensitive)

        Returns:
            GesturePreset or None if not found
        """
        name_key = name.lower().replace(' ', '_')

        # Check builtin first
        if name_key in self.builtin_presets:
            return self.builtin_presets[name_key]

        # Check custom
        if name_key in self.custom_presets:
            return self.custom_presets[name_key]

        return None

    def list_presets(self, tags: Optional[List[str]] = None) -> List[GesturePreset]:
        """
        List all available presets, optionally filtered by tags.

        Args:
            tags: Filter by tags (None = all presets)

        Returns:
            List of presets
        """
        all_presets = list(self.builtin_presets.values()) + list(self.custom_presets.values())

        if tags is None:
            return all_presets

        # Filter by tags
        filtered = []
        for preset in all_presets:
            if any(tag in preset.tags for tag in tags):
                filtered.append(preset)

        return filtered

    def add_custom_preset(self, preset: GesturePreset, save: bool = True):
        """
        Add a custom preset.

        Args:
            preset: Preset to add
            save: Whether to save to disk immediately
        """
        key = preset.name.lower().replace(' ', '_')
        self.custom_presets[key] = preset

        if save:
            self.save_custom_presets()

        logger.info(f"Added custom preset: {preset.name}")

    def remove_custom_preset(self, name: str, save: bool = True):
        """
        Remove a custom preset.

        Args:
            name: Preset name
            save: Whether to save to disk immediately
        """
        key = name.lower().replace(' ', '_')

        if key in self.custom_presets:
            del self.custom_presets[key]

            if save:
                self.save_custom_presets()

            logger.info(f"Removed custom preset: {name}")
        else:
            logger.warning(f"Preset not found: {name}")

    def get_all_tags(self) -> List[str]:
        """
        Get all unique tags from all presets.

        Returns:
            Sorted list of unique tags
        """
        all_tags = set()

        for preset in self.builtin_presets.values():
            all_tags.update(preset.tags)

        for preset in self.custom_presets.values():
            all_tags.update(preset.tags)

        return sorted(all_tags)

    def search_presets(self, query: str) -> List[GesturePreset]:
        """
        Search presets by name, description, or tags.

        Args:
            query: Search query (case-insensitive)

        Returns:
            List of matching presets
        """
        query_lower = query.lower()
        results = []

        all_presets = list(self.builtin_presets.values()) + list(self.custom_presets.values())

        for preset in all_presets:
            if (query_lower in preset.name.lower() or
                query_lower in preset.description.lower() or
                any(query_lower in tag for tag in preset.tags)):
                results.append(preset)

        return results


def create_preset_from_config(
    name: str,
    description: str,
    config: Dict,
    tags: Optional[List[str]] = None
) -> GesturePreset:
    """
    Utility function to create a preset from a config dictionary.

    Args:
        name: Preset name
        description: Description
        config: Configuration dictionary
        tags: Optional tags

    Returns:
        New GesturePreset
    """
    return GesturePreset(name, description, config, tags)


def interpolate_presets(
    preset_a: GesturePreset,
    preset_b: GesturePreset,
    blend_factor: float = 0.5
) -> Dict:
    """
    Interpolate between two presets.

    Args:
        preset_a: First preset
        preset_b: Second preset
        blend_factor: Blend factor (0.0 = all A, 1.0 = all B)

    Returns:
        Interpolated configuration dictionary
    """
    if not 0.0 <= blend_factor <= 1.0:
        raise ValueError("blend_factor must be between 0.0 and 1.0")

    result = {}

    # Get all keys
    all_keys = set(preset_a.config.keys()) | set(preset_b.config.keys())

    for key in all_keys:
        val_a = preset_a.config.get(key, 0)
        val_b = preset_b.config.get(key, 0)

        # Interpolate
        result[key] = val_a * (1.0 - blend_factor) + val_b * blend_factor

    return result


# Global instance
_preset_library = None


def get_preset_library(custom_presets_path: Optional[str] = None) -> PresetLibrary:
    """
    Get or create global preset library.

    Args:
        custom_presets_path: Path to custom presets file

    Returns:
        PresetLibrary instance
    """
    global _preset_library

    if _preset_library is None:
        _preset_library = PresetLibrary(custom_presets_path)

    return _preset_library
