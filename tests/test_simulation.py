"""Maintenance-policy simulation logic."""

import numpy as np
import pandas as pd
import pytest

from fleetguard import simulation
from fleetguard.simulation import Costs, best_fixed_interval, evaluate_policy, outcome


def test_outcome_rules():
    assert outcome(None, 200) == (True, 200, 0)           # reactive: runs to failure
    assert outcome(150, 200) == (False, 150, 50)          # serviced early: 50 cycles wasted
    assert outcome(200, 200) == (True, 200, 0)            # too late: the failure happens first
    assert outcome(np.nan, 120) == (True, 120, 0)


def test_cost_rate_penalises_early_service():
    life = pd.Series([200, 200])
    costs = Costs(service=1, failure=10)
    early = evaluate_policy(pd.Series([50.0, 50.0]), life, costs)[0]
    late = evaluate_policy(pd.Series([190.0, 190.0]), life, costs)[0]
    failing = evaluate_policy(pd.Series([np.nan, np.nan]), life, costs)[0]
    assert early["failures"] == late["failures"] == 0 and failing["failures"] == 2
    assert late["cost_per_1000_cycles"] < early["cost_per_1000_cycles"]   # wasted life costs money
    assert late["useful_life_used_pct"] == pytest.approx(95)
    assert failing["cost_per_1000_cycles"] == pytest.approx(1000 * 20 / 400)


def test_best_fixed_interval_balances_failures_and_waste():
    rng = np.random.default_rng(0)
    lifetimes = rng.normal(200, 20, 500).round()
    cheap_failures = best_fixed_interval(lifetimes, Costs(failure=1.5))
    costly_failures = best_fixed_interval(lifetimes, Costs(failure=50))
    assert costly_failures < cheap_failures          # expensive failures -> service earlier
    assert costly_failures < lifetimes.min() + 20


def test_simulation_end_to_end(fast_bundle, fd001):
    """FleetGuard should prevent most failures and beat run-to-failure on unseen-style engines."""
    train = fd001["train"]
    eval_raw = train[train["engine_id"] > 80]
    train_life = train[train["engine_id"] <= 80].groupby("engine_id")["cycle"].max().to_numpy()
    res = simulation.simulate(fast_bundle, eval_raw, train_life, Costs(), ratios=(5, 10))
    p = res["policies"]
    assert set(p) == {"reactive", "fixed", "fleetguard", "oracle"}
    assert p["reactive"]["failures"] == 20 and p["oracle"]["failures"] == 0
    assert p["fleetguard"]["failures_prevented_pct"] >= 80
    assert p["oracle"]["cost_per_1000_cycles"] <= p["fleetguard"]["cost_per_1000_cycles"] < p["reactive"]["cost_per_1000_cycles"]
    assert len(res["per_engine"]) == 20 and list(res["sweep"]["failure_cost_ratio"]) == [5, 10]
