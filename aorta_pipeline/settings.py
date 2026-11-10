from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "settings" / "default.json"


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_settings(path: Path | None = None) -> dict[str, Any]:
    defaults = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    if path is None:
        config = defaults
        config_base = PACKAGE_ROOT
    else:
        path = path.resolve()
        config = _deep_update(deepcopy(defaults), json.loads(path.read_text(encoding="utf-8")))
        config_base = path.parent

    geometry = config["geometry"]
    resolved: dict[str, str] = {}
    for name, raw_path in geometry.items():
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            # Default paths live beside this package. User overrides resolve
            # beside the supplied settings file.
            base = PACKAGE_ROOT if path is None or raw_path == defaults["geometry"].get(name) else config_base
            candidate = base / candidate
        resolved[name] = str(candidate.resolve())
    config["geometry"] = resolved
    return config


def require_files(config: dict[str, Any], names: tuple[str, ...]) -> None:
    for name in names:
        path = Path(config["geometry"][name])
        if not path.exists():
            raise FileNotFoundError(f"Missing geometry file {name!r}: {path}")
