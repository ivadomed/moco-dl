# -*- coding: utf-8 -*-
"""
Utility script to centralize dataset filename patterns and directory structure.

This module loads configuration values (e.g., filename patterns for moving, fixed,
mask, and raw files) from a YAML file, avoiding hard-coded paths in other scripts.
"""

import os
import yaml

def load_config(config_path="config.yaml"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

config = load_config()
