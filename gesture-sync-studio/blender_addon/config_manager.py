"""
Configuration management system for Gesture Sync Studio.

Provides:
- Unified configuration loading/saving
- Validation and schema enforcement
- Configuration profiles
- Environment-specific settings
- Migration support for config versions
"""

import json
import yaml
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class ConfigSchema:
    """
    Configuration schema definition and validation.
    """

    SCHEMA = {
        'version': {
            'type': str,
            'required': True,
            'default': '1.0.0'
        },
        'audio_processing': {
            'type': dict,
            'required': True,
            'schema': {
                'sample_rate': {'type': int, 'default': 22050, 'min': 8000, 'max': 48000},
                'frame_length': {'type': int, 'default': 2048, 'min': 512, 'max': 8192},
                'hop_length': {'type': int, 'default': 512, 'min': 128, 'max': 2048},
                'n_mfcc': {'type': int, 'default': 13, 'min': 1, 'max': 40},
                'enable_cache': {'type': bool, 'default': True},
                'cache_dir': {'type': (str, type(None)), 'default': None}
            }
        },
        'gesture_generation': {
            'type': dict,
            'required': True,
            'schema': {
                'gesture_intensity': {'type': float, 'default': 1.0, 'min': 0.0, 'max': 5.0},
                'smoothing': {'type': float, 'default': 0.5, 'min': 0.0, 'max': 1.0},
                'idle_motion_scale': {'type': float, 'default': 0.3, 'min': 0.0, 'max': 2.0},
                'emphasis_scale': {'type': float, 'default': 1.5, 'min': 0.0, 'max': 5.0},
                'breathing_rate': {'type': float, 'default': 0.2, 'min': 0.0, 'max': 1.0},
                'head_nod_threshold': {'type': float, 'default': 0.7, 'min': 0.0, 'max': 1.0},
                'hand_gesture_threshold': {'type': float, 'default': 0.6, 'min': 0.0, 'max': 1.0}
            }
        },
        'animation': {
            'type': dict,
            'required': True,
            'schema': {
                'fps': {'type': int, 'default': 24, 'min': 1, 'max': 240},
                'default_bones': {
                    'type': list,
                    'default': ['head', 'neck', 'spine', 'spine.001', 'spine.002',
                               'shoulder.L', 'shoulder.R', 'upper_arm.L', 'upper_arm.R',
                               'forearm.L', 'forearm.R', 'hand.L', 'hand.R']
                }
            }
        },
        'ml_inference': {
            'type': dict,
            'required': False,
            'schema': {
                'model_path': {'type': (str, type(None)), 'default': None},
                'model_type': {'type': str, 'default': 'auto', 'choices': ['auto', 'onnx', 'torch']},
                'batch_size': {'type': int, 'default': 32, 'min': 1, 'max': 256},
                'use_gpu': {'type': bool, 'default': False}
            }
        },
        'preprocessing': {
            'type': dict,
            'required': False,
            'schema': {
                'normalize': {'type': bool, 'default': True},
                'reduce_noise': {'type': bool, 'default': True},
                'remove_silence': {'type': bool, 'default': False},
                'enhance_speech': {'type': bool, 'default': True},
                'high_pass_filter': {'type': bool, 'default': True}
            }
        },
        'visualization': {
            'type': dict,
            'required': False,
            'schema': {
                'enabled': {'type': bool, 'default': False},
                'output_dir': {'type': str, 'default': 'visualizations'},
                'generate_plots': {'type': bool, 'default': True},
                'generate_3d': {'type': bool, 'default': False}
            }
        },
        'logging': {
            'type': dict,
            'required': False,
            'schema': {
                'level': {'type': str, 'default': 'INFO',
                         'choices': ['DEBUG', 'INFO', 'WARNING', 'ERROR']},
                'file': {'type': (str, type(None)), 'default': None},
                'format': {'type': str, 'default': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'}
            }
        }
    }

    @classmethod
    def validate(cls, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration against schema.

        Args:
            config: Configuration dictionary

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []

        def validate_value(value, spec, path=''):
            """Validate a single value against its specification."""
            # Type check
            expected_type = spec.get('type')
            if expected_type:
                if isinstance(expected_type, tuple):
                    # Multiple allowed types
                    if not isinstance(value, expected_type):
                        errors.append(f"{path}: Expected one of {expected_type}, got {type(value)}")
                else:
                    if not isinstance(value, expected_type):
                        errors.append(f"{path}: Expected {expected_type}, got {type(value)}")
                        return

            # Range checks for numbers
            if isinstance(value, (int, float)):
                if 'min' in spec and value < spec['min']:
                    errors.append(f"{path}: Value {value} below minimum {spec['min']}")
                if 'max' in spec and value > spec['max']:
                    errors.append(f"{path}: Value {value} above maximum {spec['max']}")

            # Choice validation
            if 'choices' in spec and value not in spec['choices']:
                errors.append(f"{path}: Value '{value}' not in allowed choices {spec['choices']}")

            # Nested schema validation
            if 'schema' in spec and isinstance(value, dict):
                for sub_key, sub_spec in spec['schema'].items():
                    if sub_key in value:
                        validate_value(value[sub_key], sub_spec, f"{path}.{sub_key}" if path else sub_key)
                    elif sub_spec.get('required', False):
                        errors.append(f"{path}.{sub_key} if path else sub_key: Required field missing")

        # Validate top-level schema
        for key, spec in cls.SCHEMA.items():
            if key in config:
                validate_value(config[key], spec, key)
            elif spec.get('required', False):
                errors.append(f"{key}: Required field missing")

        return (len(errors) == 0, errors)

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """
        Get default configuration with all default values.

        Returns:
            Default configuration dictionary
        """
        def get_defaults(schema):
            """Recursively extract defaults from schema."""
            result = {}
            for key, spec in schema.items():
                if 'default' in spec:
                    result[key] = spec['default']
                elif 'schema' in spec:
                    result[key] = get_defaults(spec['schema'])
            return result

        return get_defaults(cls.SCHEMA)


class ConfigurationManager:
    """
    Manages configuration loading, saving, and validation.
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_path: Path to configuration file (JSON or YAML)
        """
        self.config_path = Path(config_path) if config_path else None
        self.config = ConfigSchema.get_default_config()
        self.profiles = {}

        if self.config_path and self.config_path.exists():
            self.load()

    def load(self, filepath: Optional[str] = None):
        """
        Load configuration from file.

        Args:
            filepath: Optional path (uses self.config_path if None)
        """
        if filepath:
            path = Path(filepath)
        elif self.config_path:
            path = self.config_path
        else:
            raise ValueError("No configuration file path specified")

        if not path.exists():
            logger.warning(f"Configuration file not found: {path}")
            return

        try:
            with open(path, 'r') as f:
                if path.suffix in ['.yaml', '.yml']:
                    loaded = yaml.safe_load(f)
                elif path.suffix == '.json':
                    loaded = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {path.suffix}")

            # Merge with defaults
            self.config = self._merge_configs(ConfigSchema.get_default_config(), loaded)

            # Validate
            is_valid, errors = ConfigSchema.validate(self.config)
            if not is_valid:
                logger.error("Configuration validation errors:")
                for error in errors:
                    logger.error(f"  - {error}")
                raise ValueError("Invalid configuration")

            logger.info(f"Loaded configuration from {path}")

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise

    def save(self, filepath: Optional[str] = None, format: str = 'auto'):
        """
        Save configuration to file.

        Args:
            filepath: Optional path (uses self.config_path if None)
            format: Format ('auto', 'json', 'yaml')
        """
        if filepath:
            path = Path(filepath)
        elif self.config_path:
            path = self.config_path
        else:
            raise ValueError("No configuration file path specified")

        # Determine format
        if format == 'auto':
            format = 'yaml' if path.suffix in ['.yaml', '.yml'] else 'json'

        try:
            path.parent.mkdir(parents=True, exist_ok=True)

            with open(path, 'w') as f:
                if format == 'yaml':
                    yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
                elif format == 'json':
                    json.dump(self.config, f, indent=2)
                else:
                    raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Saved configuration to {path}")

        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            raise

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by key path.

        Args:
            key_path: Dot-separated key path (e.g., 'audio_processing.sample_rate')
            default: Default value if not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """
        Set configuration value by key path.

        Args:
            key_path: Dot-separated key path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config

        # Navigate to parent
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        # Set value
        config[keys[-1]] = value

    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with dictionary.

        Args:
            updates: Dictionary of updates
        """
        self.config = self._merge_configs(self.config, updates)

    def _merge_configs(self, base: Dict, updates: Dict) -> Dict:
        """
        Recursively merge configuration dictionaries.

        Args:
            base: Base configuration
            updates: Updates to apply

        Returns:
            Merged configuration
        """
        result = deepcopy(base)

        for key, value in updates.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = deepcopy(value)

        return result

    def create_profile(self, name: str, description: str = ''):
        """
        Create a configuration profile (snapshot).

        Args:
            name: Profile name
            description: Profile description
        """
        self.profiles[name] = {
            'description': description,
            'config': deepcopy(self.config)
        }

        logger.info(f"Created profile: {name}")

    def load_profile(self, name: str):
        """
        Load a configuration profile.

        Args:
            name: Profile name
        """
        if name not in self.profiles:
            raise ValueError(f"Profile not found: {name}")

        self.config = deepcopy(self.profiles[name]['config'])
        logger.info(f"Loaded profile: {name}")

    def list_profiles(self) -> List[Dict[str, str]]:
        """
        List all available profiles.

        Returns:
            List of profile information dictionaries
        """
        return [
            {'name': name, 'description': profile['description']}
            for name, profile in self.profiles.items()
        ]

    def export_profile(self, name: str, filepath: str):
        """
        Export a profile to file.

        Args:
            name: Profile name
            filepath: Output file path
        """
        if name not in self.profiles:
            raise ValueError(f"Profile not found: {name}")

        with open(filepath, 'w') as f:
            json.dump(self.profiles[name], f, indent=2)

        logger.info(f"Exported profile {name} to {filepath}")

    def import_profile(self, filepath: str, name: Optional[str] = None):
        """
        Import a profile from file.

        Args:
            filepath: Input file path
            name: Optional name (uses filename if None)
        """
        with open(filepath, 'r') as f:
            profile = json.load(f)

        if name is None:
            name = Path(filepath).stem

        self.profiles[name] = profile
        logger.info(f"Imported profile: {name}")

    def reset_to_defaults(self):
        """Reset configuration to default values."""
        self.config = ConfigSchema.get_default_config()
        logger.info("Reset configuration to defaults")

    def validate(self) -> Tuple[bool, List[str]]:
        """
        Validate current configuration.

        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        return ConfigSchema.validate(self.config)


# Global instance
_config_manager = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigurationManager:
    """
    Get or create global configuration manager.

    Args:
        config_path: Path to configuration file

    Returns:
        ConfigurationManager instance
    """
    global _config_manager

    if _config_manager is None:
        _config_manager = ConfigurationManager(config_path)

    return _config_manager
