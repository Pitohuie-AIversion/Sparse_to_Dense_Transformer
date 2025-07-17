"""Utility functions for loading and validating YAML configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import os

import yaml


_REQUIRED_KEYS: dict[str, Any] = {
    "global": ["device"],
    "data": ["path", "batch_size"],
    "training": ["epochs", "learning_rate", "early_stop_patience"],
    "model": dict,
    "attention_test": {"types": list},
}


def validate_config(cfg: dict[str, Any]) -> None:
    """Basic validation of the parsed configuration.
    
    Validates both top-level sections and nested configuration keys.
    """
    def _validate_section(config: dict, requirements: Union[list, dict], path: str = "") -> None:
        for key, req in requirements.items() if isinstance(requirements, dict) else [(path, requirements)]:
            current_path = f"{path}.{key}" if path else key
            
            if key not in config:
                raise KeyError(f"Missing required section '{current_path}' in config")
            
            if isinstance(req, list):
                for subkey in req:
                    if subkey not in config[key]:
                        raise KeyError(f"Missing key '{current_path}.{subkey}' in config")
            elif isinstance(req, dict):
                _validate_section(config[key], req, current_path)
    
    _validate_section(cfg, _REQUIRED_KEYS)


def load_config(path: Union[str, Path]) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to a YAML file. ``~`` and environment variables are expanded.

    Returns:
        Parsed configuration dictionary.
    """
    # Expand and normalise the path in case a string is provided
    path = Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    validate_config(cfg)
    return cfg
