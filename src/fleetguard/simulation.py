"""Maintenance-policy simulation: is predictive maintenance actually better?

Engines with known run-to-failure trajectories are replayed cycle by cycle
under four policies:

    reactive    run until the engine fails
    fixed       service every N cycles, whatever the engine's condition
    fleetguard  service when the triage agent raises a CRITICAL/HIGH work order
                (after a planning delay: parts, crew, slot)
    oracle      perfect knowledge of the failure time (theoretical best)

Each engine's life ends either in a *planned* service (before failure) or an
*unplanned failure*. Cost is measured per 1,000 operating cycles (a renewal
argument): servicing too early wastes useful life, so the same fleet needs
more services over time. The cost ratio failure:service is an assumption,
so results are reported for several ratios.

Honesty rules:
* Engines are never ones the model was trained on: by default, the
  retraining batch replayed through v1 (which never saw it).
* The fixed interval is the *best possible* interval for the engines the
  model was trained on, so the baseline is as strong as it can be.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .agent.triage import TriageAgent
from .scoring import score_history

log = logging.getLogger(__name__)
MAINTENANCE_SEVERITIES = ("CRITICAL", "HIGH")


@dataclass
class Costs:
    service: float = 1.0          # planned maintenance
    failure: float = 10.0         # unplanned failure (repair + downtime + secondary damage)
    planning_delay: int = 5       # cycles between a work order and the service actually happening


def outcome(maintain_at: float, lifetime: int) -> tuple[bool, int, int]:
    """(failed?, cycles operated, wasted cycles) for one engine life.

    `lifetime` is the last cycle before failure (RUL 0 in C-MAPSS). A service
    must happen strictly before it to prevent the failure.
    """
    if maintain_at is None or not np.isfinite(maintain_at) or maintain_at >= lifetime:
        return True, int(lifetime), 0
    m = max(int(maintain_at), 1)
    return False, m, int(lifetime - m)


def fleetguard_triggers(bundle: dict, raw: pd.DataFrame) -> pd.DataFrame:
    """First cycle at which the agent asks for maintenance, per engine (replaying causally)."""
    hist, feats, _ = score_history(bundle, raw)
    agent = TriageAgent(bundle["thresholds"])
    rows = []
    for engine, g in hist.groupby("engine_id", sort=True):
        trigger, severity = np.nan, None
        for state in g.to_dict("records"):
            d = agent.decide(state)
            if d.severity in MAINTENANCE_SEVERITIES:
                trigger, severity = state["cycle"], d.severity
                break
        rows.append({"engine_id": engine, "lifetime": int(g["cycle"].max()), "trigger_cycle": trigger,
                     "trigger_severity": severity})
    return pd.DataFrame(rows)


def evaluate_policy(maintain_at: pd.Series, lifetimes: pd.Series, costs: Costs) -> dict:
    res = [outcome(m, t) for m, t in zip(maintain_at, lifetimes)]
    failed = np.array([r[0] for r in res])
    cycles = np.array([r[1] for r in res])
    wasted = np.array([r[2] for r in res])
    total_cost = failed.sum() * costs.failure + (~failed).sum() * costs.service
    return {
        "engines": int(len(res)),
        "failures": int(failed.sum()),
        "failures_prevented_pct": float(100 * (~failed).mean()),
        "mean_wasted_cycles": float(wasted[~failed].mean()) if (~failed).any() else 0.0,
        "useful_life_used_pct": float(100 * cycles.sum() / lifetimes.sum()),
        "cost_per_1000_cycles": float(1000 * total_cost / cycles.sum()),
    }, failed, cycles, wasted


def best_fixed_interval(lifetimes: np.ndarray, costs: Costs) -> int:
    """Interval with the lowest cost rate on the given (training) lifetimes."""
    lifetimes = np.asarray(lifetimes)
    best, best_cost = None, np.inf
    for n in range(10, int(lifetimes.max()) + 1):
        failed = lifetimes <= n
        cycles = np.where(failed, lifetimes, n)
        rate = (failed.sum() * costs.failure + (~failed).sum() * costs.service) / cycles.sum()
        if rate < best_cost:
            best, best_cost = n, rate
    return int(best)


def simulate(bundle: dict, raw_eval: pd.DataFrame, train_lifetimes: np.ndarray, costs: Costs,
             ratios=(2, 5, 10, 20, 50)) -> dict:
    """Run all four policies on `raw_eval` (full run-to-failure trajectories)."""
    trig = fleetguard_triggers(bundle, raw_eval)
    life = trig["lifetime"]
    interval = best_fixed_interval(train_lifetimes, costs)
    plans = {
        "reactive": pd.Series(np.nan, index=trig.index),
        "fixed": pd.Series(float(interval), index=trig.index),
        "fleetguard": trig["trigger_cycle"] + costs.planning_delay,
        "oracle": life - 1,
    }
    summary, per_engine = {}, trig.copy()
    for name, plan in plans.items():
        m, failed, cycles, wasted = evaluate_policy(plan, life, costs)
        summary[name] = m
        per_engine[f"{name}_failed"] = failed
        per_engine[f"{name}_wasted"] = wasted

    # Sensitivity: the failure:service cost ratio is an assumption, so sweep it.
    sweep = []
    for r in ratios:
        c = Costs(service=1.0, failure=float(r), planning_delay=costs.planning_delay)
        n = best_fixed_interval(train_lifetimes, c)
        row = {"failure_cost_ratio": r, "fixed_interval": n}
        for name, plan in {**plans, "fixed": pd.Series(float(n), index=trig.index)}.items():
            row[name] = evaluate_policy(plan, life, c)[0]["cost_per_1000_cycles"]
        row["fleetguard_saving_vs_fixed_pct"] = 100 * (1 - row["fleetguard"] / row["fixed"])
        row["fleetguard_saving_vs_reactive_pct"] = 100 * (1 - row["fleetguard"] / row["reactive"])
        sweep.append(row)

    fg, fx, re_ = summary["fleetguard"], summary["fixed"], summary["reactive"]
    headline = {
        "fixed_interval": interval,
        "saving_vs_fixed_pct": 100 * (1 - fg["cost_per_1000_cycles"] / fx["cost_per_1000_cycles"]),
        "saving_vs_reactive_pct": 100 * (1 - fg["cost_per_1000_cycles"] / re_["cost_per_1000_cycles"]),
        "gap_to_oracle_pct": 100 * (fg["cost_per_1000_cycles"] / summary["oracle"]["cost_per_1000_cycles"] - 1),
    }
    return {"policies": summary, "headline": headline, "per_engine": per_engine, "sweep": pd.DataFrame(sweep)}


def save(result: dict, out_dir, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    result["per_engine"].to_csv(out_dir / "policy_engines.csv", index=False)
    result["sweep"].to_csv(out_dir / "cost_ratio_sweep.csv", index=False)
    payload = {**meta, "policies": result["policies"], "headline": result["headline"]}
    (out_dir / "policy_summary.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
