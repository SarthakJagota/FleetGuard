"""Rule-based Triage Agent (no LLM).

Reads the model outputs for one engine and returns a decision with a
severity, an action and *machine-readable reason codes*, so every work
order can be traced back to the exact rule that produced it.

Decision flow (first matching rule wins):

    1. Data gates          -> REVIEW    (invalid prediction, too much repaired data,
                                         too little history while the RUL looks urgent)
                           -> LOW       (too little history and RUL looks healthy:
                                         keep collecting data)
    2. RUL <= critical     -> CRITICAL  (work order P1, ground / immediate maintenance)
    3. RUL <= high, or the
       lower interval bound <= critical, or
       anomaly alarm with RUL <= medium -> HIGH  (work order P2, schedule maintenance)
    4. Interval too wide, or persistent
       anomaly while the RUL lower bound
       is still very healthy           -> REVIEW (models disagree -> human decides)
    5. RUL <= medium or anomaly alarm  -> MEDIUM (inspection request P3)
    6. otherwise                       -> LOW    (continue monitoring, no order)
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from ..data.loader import SENSOR_INFO

SEVERITY_RANK = {"LOW": 0, "MEDIUM": 1, "REVIEW": 1, "HIGH": 2, "CRITICAL": 3}

COMPONENT_CHECKS = {
    "High-pressure compressor": "Borescope the HPC; check for blade fouling/erosion and consider a compressor wash",
    "Low-pressure compressor": "Inspect LPC blades and inlet guide vanes",
    "Fan": "Inspect fan blades and case; verify fan-speed sensor readings",
    "Low-pressure turbine": "Borescope the LPT; review exhaust-gas temperature margin",
    "High-pressure turbine": "Inspect HPT blades and cooling-air passages",
    "Core / HP spool": "Check core-speed trend and HP spool bearing health",
    "Combustor / fuel system": "Inspect fuel nozzles and fuel-metering unit",
    "Bleed system": "Check bleed valves and bleed-air ducting",
    "Bypass duct": "Inspect bypass duct for leaks or damage",
}


@dataclass
class TriageThresholds:
    critical_rul: float = 15
    high_rul: float = 30
    medium_rul: float = 60
    min_history: int = 15
    max_interval_width: float = 90
    max_imputed_fraction: float = 0.2
    disagreement_rul: float = 90

    @classmethod
    def from_config(cls, cfg, high_rul: float | None = None) -> "TriageThresholds":
        t = cfg["triage"]
        return cls(
            critical_rul=t["critical_rul"], high_rul=high_rul if high_rul is not None else t["high_rul"],
            medium_rul=t["medium_rul"], min_history=t["min_history"],
            max_interval_width=t["max_interval_width"], max_imputed_fraction=t["max_imputed_fraction"],
            disagreement_rul=t["disagreement_rul"],
        )


@dataclass
class Decision:
    engine_id: int
    cycle: int
    severity: str
    action: str
    reasons: list[str]
    reason_codes: list[str]
    rul_pred: float
    rul_lower: float
    rul_upper: float
    anomaly_index: float
    anomaly_alarm: bool
    due_within_cycles: int | None
    indicators: list[dict] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)

    @property
    def creates_work_order(self) -> bool:
        return self.severity != "LOW"

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate_high_threshold(pred, true, degraded: int, target_recall: float, bounds) -> dict:
    """Smallest predicted-RUL threshold that catches >= target_recall of engines
    truly within `degraded` cycles of failure (on validation snapshots)."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    pos = true <= degraded
    lo, hi = bounds
    chosen, recall = hi, None
    for t in np.arange(lo, hi + 1):
        r = float(((pred <= t) & pos).sum() / max(pos.sum(), 1))
        if r >= target_recall:
            chosen, recall = float(t), r
            break
    if recall is None:
        recall = float(((pred <= hi) & pos).sum() / max(pos.sum(), 1))
    flagged = pred <= chosen
    return {
        "high_rul": chosen,
        "val_recall": recall,
        "val_precision": float((flagged & pos).sum() / max(flagged.sum(), 1)),
        "target_recall": target_recall,
    }


class TriageAgent:
    def __init__(self, thresholds: TriageThresholds, top_indicators: int = 3):
        self.t = thresholds
        self.top_indicators = top_indicators

    def decide(self, state: dict, deviations: pd.Series | None = None) -> Decision:
        t = self.t
        rul = float(state["rul_pred"])
        lo = float(state.get("rul_lower", rul))
        hi = float(state.get("rul_upper", rul))
        idx = float(state.get("anomaly_index", 0.0))
        alarm = bool(state.get("anomaly_alarm", False))
        n_cycles = int(state.get("n_cycles", t.min_history))
        imputed = float(state.get("imputed_fraction", 0.0))

        reasons: list[tuple[str, str]] = []
        severity = None

        # 1. Data / validity gates ------------------------------------------------
        if not all(math.isfinite(v) for v in (rul, lo, hi, idx)):
            severity = "REVIEW"
            reasons.append(("INVALID_PREDICTION", "Model output is not a finite number"))
        if imputed > t.max_imputed_fraction:
            severity = "REVIEW"
            reasons.append(("DATA_QUALITY", f"{imputed:.0%} of recent readings were repaired (> {t.max_imputed_fraction:.0%})"))
        if n_cycles < t.min_history and severity is None:
            reasons.append(("INSUFFICIENT_HISTORY", f"Only {n_cycles} cycles observed (< {t.min_history}); trends unreliable"))
            if math.isfinite(rul) and rul <= t.high_rul:
                severity = "REVIEW"  # model says urgent but cannot be trusted yet: a human decides
            else:
                severity = "LOW"     # keep collecting data; no order for a young, healthy-looking engine
        if severity == "REVIEW" and math.isfinite(rul) and rul <= t.high_rul:
            reasons.append(("POSSIBLY_URGENT", f"Model RUL {rul:.0f} would otherwise trigger maintenance"))

        # 2-6. Risk rules -----------------------------------------------------------
        if severity is None:
            if rul <= t.critical_rul:
                severity = "CRITICAL"
                reasons.append(("RUL_CRITICAL", f"Predicted RUL {rul:.0f} <= {t.critical_rul:.0f} cycles"))
            elif rul <= t.high_rul or lo <= t.critical_rul or (alarm and rul <= t.medium_rul):
                severity = "HIGH"
                if rul <= t.high_rul:
                    reasons.append(("RUL_HIGH", f"Predicted RUL {rul:.0f} <= {t.high_rul:.0f} cycles"))
                if lo <= t.critical_rul:
                    reasons.append(("RUL_LOWER_BOUND", f"Lower interval bound {lo:.0f} <= {t.critical_rul:.0f} cycles"))
                if alarm and rul <= t.medium_rul:
                    reasons.append(("ANOMALY_AND_RUL", f"Persistent anomaly with RUL {rul:.0f} <= {t.medium_rul:.0f}"))
            elif hi - lo > t.max_interval_width:
                severity = "REVIEW"
                reasons.append(("HIGH_UNCERTAINTY", f"Interval width {hi - lo:.0f} > {t.max_interval_width:.0f} cycles"))
            elif alarm and lo > t.disagreement_rul:
                severity = "REVIEW"
                reasons.append(("MODEL_DISAGREEMENT", f"Persistent sensor anomaly (index {idx:.2f}) but RUL lower bound {lo:.0f} looks healthy"))
            elif rul <= t.medium_rul or alarm:
                severity = "MEDIUM"
                if rul <= t.medium_rul:
                    reasons.append(("RUL_MEDIUM", f"Predicted RUL {rul:.0f} <= {t.medium_rul:.0f} cycles"))
                if alarm:
                    reasons.append(("ANOMALY_ALARM", f"Anomaly index above threshold for {int(state.get('anomaly_run', 0))} consecutive cycles"))
            else:
                severity = "LOW"
                reasons.append(("HEALTHY", f"Predicted RUL {rul:.0f} > {t.medium_rul:.0f} cycles, no persistent anomaly"))

        indicators, checks = self._indicators(deviations)
        return Decision(
            engine_id=int(state["engine_id"]), cycle=int(state["cycle"]), severity=severity,
            action=self._action(severity), reasons=[r[1] for r in reasons], reason_codes=[r[0] for r in reasons],
            rul_pred=round(rul, 1) if math.isfinite(rul) else rul,
            rul_lower=round(lo, 1) if math.isfinite(lo) else lo,
            rul_upper=round(hi, 1) if math.isfinite(hi) else hi,
            anomaly_index=round(idx, 3) if math.isfinite(idx) else idx, anomaly_alarm=alarm,
            due_within_cycles=self._due(severity, lo), indicators=indicators, checks=checks,
        )

    @staticmethod
    def _action(severity: str) -> str:
        return {
            "CRITICAL": "Remove from service and perform maintenance before next operation",
            "HIGH": "Schedule maintenance within the due window",
            "MEDIUM": "Flag for inspection at the next planned check; increase monitoring",
            "REVIEW": "Route to an engineer for manual review before any action",
            "LOW": "Continue normal monitoring",
        }[severity]

    @staticmethod
    def _due(severity: str, lower: float) -> int | None:
        if not math.isfinite(lower):
            return None
        if severity == "CRITICAL":
            return 0
        if severity == "HIGH":
            return max(1, int(lower) - 5)  # finish before the pessimistic bound, with margin
        if severity == "MEDIUM":
            return max(5, int(lower * 0.5))
        return None

    def _indicators(self, deviations: pd.Series | None):
        if deviations is None or deviations.empty:
            return [], []
        top = deviations.reindex(deviations.abs().sort_values(ascending=False).index).head(self.top_indicators)
        indicators, checks = [], []
        for sensor, dev in top.items():
            short, desc, comp = SENSOR_INFO.get(sensor, (sensor, sensor, ""))
            indicators.append({
                "sensor": sensor, "name": short, "description": desc, "component": comp,
                "deviation_sd": round(float(dev), 2), "direction": "rising" if dev > 0 else "falling",
            })
            check = COMPONENT_CHECKS.get(comp)
            if check and check not in checks:
                checks.append(check)
        return indicators, checks
