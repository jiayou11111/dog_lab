"""Dog Lab extension package."""

import os

import toml


DOG_LAB_EXT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
"""Path to the Dog Lab extension source directory."""

DOG_LAB_DATA_DIR = os.path.join(DOG_LAB_EXT_DIR, "data")
"""Path to Dog Lab data assets."""

DOG_LAB_METADATA = toml.load(os.path.join(DOG_LAB_EXT_DIR, "config", "extension.toml"))
"""Extension metadata dictionary parsed from extension.toml."""

__version__ = DOG_LAB_METADATA["package"]["version"]

from .assets import *
from .tasks import *
