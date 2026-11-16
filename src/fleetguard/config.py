"""Configuration loading and project paths."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "config.yaml"
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


@dataclass
class Config:
    """Thin wrapper around the YAML dict with resolved paths."""

    raw: dict[str, Any]
    root: Path = PROJECT_ROOT

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    def path(self, name: str) -> Path:
        value = Path(self.raw["paths"][name])
        return value if value.is_absolute() else self.root / value

    @property
    def tracking_uri(self) -> str:
        uri = self.raw["paths"]["mlflow_tracking_uri"]
        if uri.startswith("sqlite:///") and not Path(uri[len("sqlite:///"):]).is_absolute():
            db = (self.root / uri[len("sqlite:///"):]).as_posix()
            return f"sqlite:///{db}"
        return uri

    def subset(self, name: str) -> dict[str, Any]:
        name = name.upper()
        if name not in self.raw["subsets"]:
            raise ValueError(f"Unknown subset {name!r}; expected one of {SUBSETS}")
        return self.raw["subsets"][name]

    def override(self, **sections: dict[str, Any]) -> "Config":
        """Return a copy with some section values replaced (used by tests)."""
        raw = copy.deepcopy(self.raw)
        for section, values in sections.items():
            raw.setdefault(section, {}).update(values)
        return Config(raw=raw, root=self.root)


def load_config(path: str | Path | None = None, root: Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    return Config(raw=raw, root=root or PROJECT_ROOT)
