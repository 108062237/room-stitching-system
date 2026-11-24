"""Packaged reference settings, independent of process cwd and environment."""
import json
from importlib.resources import files

DEFAULTS = json.loads(files("room_assembly").joinpath("defaults.json").read_text(encoding="utf-8"))

