import yaml
import logging
from typing import Dict, Any


# Helper class to convert dict to object for easier access (e.g., config.model.d_model)
class ConfigDict(dict):
    def __getattr__(self, name):
        value = self.get(name)
        if isinstance(value, dict):
            return ConfigDict(value)
        return value

    # Allow attribute modification
    def __setattr__(self, name, value):
        self[name] = value

    # Method to convert back to regular dict (for W&B logging)
    def to_dict(self):
        return dict(self)


def load_config(config_path: str) -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        logging.info(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file {config_path}: {e}")
        raise


def convert_to_dot_notation(config: Dict) -> ConfigDict:
    """Converts a dictionary to allow dot notation access."""
    if not isinstance(config, dict):
        return config

    # Recursive conversion
    def _convert(d):
        if isinstance(d, dict):
            return ConfigDict({k: _convert(v) for k, v in d.items()})
        return d

    return _convert(config)
