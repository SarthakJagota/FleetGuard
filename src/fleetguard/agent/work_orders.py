"""Work-order creation, de-duplication, escalation and persistence.

The store is a single JSON file so it is easy to inspect and diff. The
agent never opens a second order for an engine that already has one open:
it updates the existing order, and escalates it if the severity rises.
Orders are only closed by a human (dashboard or CLI), never automatically.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .triage import SEVERITY_RANK, Decision

ORDER_TYPE = {
    "CRITICAL": ("P1", "CORRECTIVE_MAINTENANCE"),
    "HIGH": ("P2", "SCHEDULED_MAINTENANCE"),
    "MEDIUM": ("P3", "INSPECTION"),
    "REVIEW": ("P3", "HUMAN_REVIEW"),
}
OPEN_STATES = {"OPEN", "ACKNOWLEDGED"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class WorkOrderStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.orders: list[dict] = []
        if self.path.exists():
            self.orders = json.loads(self.path.read_text(encoding="utf-8"))

    # --------------------------------------------------------------- queries
    def open_order(self, subset: str, engine_id: int) -> dict | None:
        for wo in reversed(self.orders):
            if wo["subset"] == subset and wo["engine_id"] == engine_id and wo["status"] in OPEN_STATES:
                return wo
        return None

    def get(self, wo_id: str) -> dict | None:
        return next((wo for wo in self.orders if wo["wo_id"] == wo_id), None)

    def _next_id(self) -> str:
        last = max((int(wo["wo_id"].split("-")[1]) for wo in self.orders), default=0)
        return f"WO-{last + 1:05d}"

    # ---------------------------------------------------------------- upsert
    def apply(self, subset: str, decision: Decision, model_version: str) -> tuple[str, dict | None]:
        """Apply one agent decision. Returns (event, order) where event is one of
        CREATED / ESCALATED / UPDATED / NO_ACTION."""
        if not decision.creates_work_order:
            return "NO_ACTION", None
        existing = self.open_order(subset, decision.engine_id)
        snapshot = self._snapshot(decision, model_version)
        if existing is None:
            priority, otype = ORDER_TYPE[decision.severity]
            wo = {
                "wo_id": self._next_id(),
                "subset": subset,
                "engine_id": decision.engine_id,
                "status": "OPEN",
                "severity": decision.severity,
                "priority": priority,
                "type": otype,
                "title": self._title(decision),
                "created_at": _now(),
                "created_at_cycle": decision.cycle,
                "action": decision.action,
                "due_within_cycles": decision.due_within_cycles,
                "requires_human": decision.severity == "REVIEW",
                **snapshot,
                "history": [{"at": _now(), "cycle": decision.cycle, "event": "CREATED", "severity": decision.severity}],
            }
            self.orders.append(wo)
            return "CREATED", wo

        if SEVERITY_RANK[decision.severity] > SEVERITY_RANK[existing["severity"]] or (
            existing["type"] == "HUMAN_REVIEW" and decision.severity in ("CRITICAL", "HIGH")
        ):
            priority, otype = ORDER_TYPE[decision.severity]
            existing.update(
                severity=decision.severity, priority=priority, type=otype, title=self._title(decision),
                action=decision.action, due_within_cycles=decision.due_within_cycles,
                requires_human=decision.severity == "REVIEW", **snapshot,
            )
            existing["history"].append({"at": _now(), "cycle": decision.cycle, "event": "ESCALATED", "severity": decision.severity})
            return "ESCALATED", existing

        # Same or lower severity: refresh the latest evidence, never auto-downgrade.
        if existing.get("last_cycle") != decision.cycle:
            existing.update(**snapshot)
            if decision.severity == existing["severity"] and decision.due_within_cycles is not None:
                existing["due_within_cycles"] = decision.due_within_cycles
            existing["history"].append({"at": _now(), "cycle": decision.cycle, "event": "UPDATED", "severity": decision.severity})
        return "UPDATED", existing

    def set_status(self, wo_id: str, status: str, note: str = "", by: str = "engineer") -> dict:
        status = status.upper()
        if status not in {"OPEN", "ACKNOWLEDGED", "CLOSED", "REJECTED"}:
            raise ValueError(f"Invalid status {status!r}")
        wo = self.get(wo_id)
        if wo is None:
            raise KeyError(wo_id)
        wo["status"] = status
        wo["history"].append({"at": _now(), "event": f"STATUS_{status}", "by": by, "note": note})
        return wo

    # ----------------------------------------------------------------- save
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self.orders, fh, indent=2, default=str)
        os.replace(tmp, self.path)  # atomic: the dashboard never reads a half-written file

    # -------------------------------------------------------------- helpers
    @staticmethod
    def _snapshot(d: Decision, model_version: str) -> dict:
        return {
            "last_cycle": d.cycle,
            "rul_pred": d.rul_pred,
            "rul_interval": [d.rul_lower, d.rul_upper],
            "anomaly_index": d.anomaly_index,
            "anomaly_alarm": d.anomaly_alarm,
            "reasons": d.reasons,
            "reason_codes": d.reason_codes,
            "indicators": d.indicators,
            "recommended_checks": d.checks,
            "model_version": model_version,
            "updated_at": _now(),
        }

    @staticmethod
    def _title(d: Decision) -> str:
        label = {"CRITICAL": "Critical", "HIGH": "High priority", "MEDIUM": "Inspection",
                 "REVIEW": "Manual review"}[d.severity]
        return f"{label}: engine {d.engine_id} - est. {d.rul_pred:.0f} cycles remaining"


def render_markdown(wo: dict) -> str:
    """Human-readable work order (what a technician would receive)."""
    lo, hi = wo["rul_interval"]
    lines = [
        f"# {wo['wo_id']} - {wo['title']}",
        "",
        f"| Field | Value |", "|---|---|",
        f"| Subset / engine | {wo['subset']} / {wo['engine_id']} |",
        f"| Severity / priority | {wo['severity']} / {wo['priority']} |",
        f"| Type | {wo['type']} |",
        f"| Status | {wo['status']} |",
        f"| Estimated RUL | {wo['rul_pred']:.0f} cycles (80% interval {lo:.0f}-{hi:.0f}) |",
        f"| Anomaly index | {wo['anomaly_index']:.2f} ({'persistent alarm' if wo['anomaly_alarm'] else 'no persistent alarm'}) |",
        f"| Due within | {'immediately' if wo['due_within_cycles'] == 0 else str(wo['due_within_cycles']) + ' cycles' if wo['due_within_cycles'] is not None else 'n/a'} |",
        f"| Observed at cycle | {wo['last_cycle']} |",
        f"| Model version | {wo['model_version']} |",
        "",
        f"**Recommended action:** {wo['action']}",
        "",
        "## Why this order was raised",
        *[f"- `{c}` {r}" for c, r in zip(wo["reason_codes"], wo["reasons"])],
    ]
    if wo["indicators"]:
        lines += ["", "## Primary indicators", "| Sensor | Meaning | Component | Deviation |", "|---|---|---|---|"]
        lines += [
            f"| {i['name']} ({i['sensor']}) | {i['description']} | {i['component']} | {i['deviation_sd']:+.1f} SD, {i['direction']} |"
            for i in wo["indicators"]
        ]
    if wo["recommended_checks"]:
        lines += ["", "## Suggested checks", *[f"- {c}" for c in wo["recommended_checks"]]]
    lines += [
        "", "> RUL is a statistical estimate from historical run-to-failure data, not a guarantee.",
        "> Suggested checks are a heuristic mapping from sensor location to component.",
    ]
    return "\n".join(lines) + "\n"
