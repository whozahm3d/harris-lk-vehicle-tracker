"""
utils/config_loader.py
======================
Loads, validates, and exposes config.yaml as a dot-accessible
nested object. All other modules import their parameters from here.

Usage:
    from utils.config_loader import ConfigLoader
    cfg = ConfigLoader("config.yaml").get()
    print(cfg.harris.max_corners)
"""

import os
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Dot-accessible wrapper
# ---------------------------------------------------------------------------

class _DotDict:
    """
    Recursively wraps a dictionary so values are accessible via dot notation.

    Args:
        data (dict): Raw dictionary to wrap.

    Example:
        d = _DotDict({"harris": {"max_corners": 200}})
        d.harris.max_corners  # 200
    """

    def __init__(self, data: dict) -> None:
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, _DotDict(value))
            else:
                setattr(self, key, value)

    def __repr__(self) -> str:  # noqa: D105
        return f"_DotDict({self.__dict__})"

    def to_dict(self) -> dict:
        """Return the wrapped data as a plain nested dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, _DotDict):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result


# ---------------------------------------------------------------------------
# Required top-level keys and their expected Python types
# ---------------------------------------------------------------------------

_REQUIRED_SECTIONS: dict[str, type] = {
    "input": dict,
    "memory": dict,
    "harris": dict,
    "lucas_kanade": dict,
    "tracking": dict,
    "preprocessing": dict,
    "output": dict,
    "metrics": dict,
    "cli": dict,
}


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------

class ConfigLoader:
    """
    Loads and validates the project configuration from a YAML file.

    Args:
        config_path (str): Path to config.yaml. Defaults to "config.yaml".

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If a required section is missing or has the wrong type.
        yaml.YAMLError: If the YAML file is malformed.

    Example:
        cfg = ConfigLoader("config.yaml").get()
        win = tuple(cfg.lucas_kanade.win_size)  # (21, 21)
    """

    def __init__(self, config_path: str = "config.yaml") -> None:
        self._path = os.path.abspath(config_path)
        self._raw: dict[str, Any] = self._load()
        self._validate()
        self._config: _DotDict = _DotDict(self._raw)

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def config(self) -> _DotDict:
        """Alias for get() — dot-accessible config object."""
        return self._config

    def get_raw(self) -> dict[str, Any]:
        """
        Return the raw config as a plain Python dictionary.

        Returns:
            dict: Unmodified parsed YAML content.
        """
        return self._raw

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _load(self) -> dict[str, Any]:
        """
        Read and parse the YAML file.

        Returns:
            dict: Parsed YAML content.

        Raises:
            FileNotFoundError: If config file path does not exist.
            yaml.YAMLError: If the file cannot be parsed.
        """
        if not os.path.isfile(self._path):
            raise FileNotFoundError(
                f"[ConfigLoader] config.yaml not found at: {self._path}\n"
                "Make sure config.yaml is in the project root directory."
            )

        with open(self._path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(
                "[ConfigLoader] config.yaml must be a YAML mapping at the top level."
            )

        return data

    def _validate(self) -> None:
        """
        Check that all required top-level sections exist and have correct types.

        Raises:
            ValueError: On any missing key or type mismatch.
        """
        errors: list[str] = []

        for section, expected_type in _REQUIRED_SECTIONS.items():
            if section not in self._raw:
                errors.append(f"  Missing required section: '{section}'")
            elif not isinstance(self._raw[section], expected_type):
                actual = type(self._raw[section]).__name__
                errors.append(
                    f"  Section '{section}' must be {expected_type.__name__}, "
                    f"got {actual}"
                )

        if errors:
            raise ValueError(
                "[ConfigLoader] Invalid config.yaml:\n" + "\n".join(errors)
            )