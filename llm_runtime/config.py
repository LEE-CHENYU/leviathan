"""Config loading helpers for YAML-backed runtime profiles."""

from __future__ import annotations

from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PACKAGE_ROOT / "config"
BACKENDS_DIR = CONFIG_ROOT / "backends"
PROFILES_DIR = CONFIG_ROOT / "profiles"


def load_named_yaml(directory: Path) -> dict[str, dict]:
    """Load `*.yaml` files keyed by file stem."""

    configs: dict[str, dict] = {}
    for path in sorted(directory.glob("*.yaml")):
        data = yaml.safe_load(path.read_text()) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Config {path} must deserialize to a mapping")
        data.setdefault("name", path.stem)
        configs[path.stem] = data
    return configs
