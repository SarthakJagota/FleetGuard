"""Local, file-based model registry.

models/<subset>/
    registry.json        version list, metrics, status and the current champion
    v1/bundle.joblib     features + RUL model + anomaly detector + triage thresholds
    v1/metadata.json     human-readable summary of the version

MLflow keeps the full experiment history; this registry is what the
serving side (scoring, dashboard, streaming) loads, so it works even when
the MLflow server is not running.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib


class ModelRegistry:
    def __init__(self, root: str | Path, subset: str):
        self.subset = subset.upper()
        self.dir = Path(root) / self.subset
        self.index_path = self.dir / "registry.json"
        self.index = {"subset": self.subset, "champion": None, "versions": []}
        if self.index_path.exists():
            self.index = json.loads(self.index_path.read_text(encoding="utf-8"))

    @property
    def champion(self) -> str | None:
        return self.index["champion"]

    def next_version(self) -> str:
        return f"v{len(self.index['versions']) + 1}"

    def register(self, bundle: dict, metadata: dict) -> str:
        version = bundle["version"]
        vdir = self.dir / version
        vdir.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, vdir / "bundle.joblib", compress=3)
        (vdir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        self.index["versions"].append({
            "version": version,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "candidate",
            **{k: metadata.get(k) for k in ("model_name", "trained_engines", "data_sources", "mlflow_run_id")},
            "metrics": metadata.get("metrics", {}),
        })
        self._save()
        return version

    def promote(self, version: str, reason: str) -> None:
        for v in self.index["versions"]:
            if v["version"] == self.index["champion"]:
                v["status"] = "archived"
            if v["version"] == version:
                v["status"] = "champion"
                v["decision"] = reason
        self.index["champion"] = version
        self._save()

    def reject(self, version: str, reason: str) -> None:
        for v in self.index["versions"]:
            if v["version"] == version:
                v["status"] = "rejected"
                v["decision"] = reason
        self._save()

    def load(self, version: str | None = None) -> dict:
        version = version or self.champion
        if version is None:
            raise FileNotFoundError(f"No champion model for {self.subset}. Run `fleetguard train --subset {self.subset}` first.")
        return joblib.load(self.dir / version / "bundle.joblib")

    def versions(self) -> list[dict]:
        return list(self.index["versions"])

    def _save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(json.dumps(self.index, indent=2, default=str), encoding="utf-8")
