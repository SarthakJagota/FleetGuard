"""MLflow experiment tracking.

Each subset gets one experiment. Every candidate model is a nested run
(params, metrics, feature importance), and the selected model is logged
and registered in the MLflow Model Registry as ``fleetguard-rul-<subset>``
with the alias ``champion`` once it is promoted.

Tracking can be switched off (tests, quick runs) and then becomes a no-op.
"""

from __future__ import annotations

import contextlib
import logging
import os
import warnings
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
log = logging.getLogger(__name__)


class Tracker:
    def __init__(self, tracking_uri: str | None, enabled: bool = True):
        self.enabled = enabled and tracking_uri is not None
        self.mlflow = None
        if self.enabled:
            try:
                import mlflow

                logging.getLogger("mlflow").setLevel(logging.ERROR)
                warnings.filterwarnings("ignore", module="mlflow")
                mlflow.set_tracking_uri(tracking_uri)
                self.mlflow = mlflow
            except ImportError:
                log.warning("mlflow not installed - experiment tracking disabled")
                self.enabled = False

    @contextlib.contextmanager
    def run(self, experiment: str, run_name: str, nested: bool = False, tags: dict | None = None):
        if not self.enabled:
            yield None
            return
        self.mlflow.set_experiment(experiment)
        with self.mlflow.start_run(run_name=run_name, nested=nested, tags=tags or {}) as run:
            yield run

    def log_params(self, params: dict) -> None:
        if self.enabled:
            self.mlflow.log_params({k: str(v)[:500] for k, v in params.items()})

    def log_metrics(self, metrics: dict, step: int | None = None) -> None:
        if self.enabled:
            clean = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and v == v}
            self.mlflow.log_metrics(clean, step=step)

    def log_artifact(self, path: str | Path, folder: str | None = None) -> None:
        if self.enabled:
            self.mlflow.log_artifact(str(path), artifact_path=folder)

    def log_dict(self, data: dict, name: str) -> None:
        if self.enabled:
            self.mlflow.log_dict(data, name)

    def log_model(self, estimator, subset: str, X_example) -> str | None:
        """Log the sklearn-API estimator and register a new model version."""
        if not self.enabled:
            return None
        kwargs = dict(name="rul_model", input_example=X_example.head(3), registered_model_name=f"fleetguard-rul-{subset}")
        module = type(estimator).__module__
        try:
            # Native flavours for boosted trees; cloudpickle for the sklearn models
            # (MLflow's default skops format refuses third-party estimator types).
            if module.startswith("xgboost"):
                info = self.mlflow.xgboost.log_model(estimator, **kwargs)
            elif module.startswith("lightgbm"):
                info = self.mlflow.lightgbm.log_model(estimator, **kwargs)
            else:
                info = self.mlflow.sklearn.log_model(estimator, serialization_format="cloudpickle", **kwargs)
            return str(getattr(info, "registered_model_version", "") or "")
        except Exception as exc:  # registry problems must never break training
            log.warning("MLflow model logging failed: %s", exc)
            return None

    def set_champion(self, subset: str, mlflow_version: str | None) -> None:
        if not (self.enabled and mlflow_version):
            return
        try:
            from mlflow import MlflowClient

            MlflowClient().set_registered_model_alias(f"fleetguard-rul-{subset}", "champion", mlflow_version)
        except Exception as exc:
            log.warning("Could not set champion alias: %s", exc)
