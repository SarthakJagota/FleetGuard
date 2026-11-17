# FleetGuard design notes

This document explains *why* each part of the system is built the way it is.
The README covers what it does and how to run it.

## 1. Problem framing

A fleet streams sensor readings every operating cycle. We want to:

1. notice degradation early (**anomaly detection**, unsupervised),
2. estimate how many cycles are left (**RUL regression**, supervised, with uncertainty),
3. turn that into an action automatically (**rule-based triage agent → work order**),
4. keep the model healthy over time (**MLOps**: tracking, registry, drift, retraining),
5. leave humans only the exceptions (**REVIEW** orders + close/reject in the dashboard).

NASA C-MAPSS provides run-to-failure trajectories for training and truncated
trajectories (with the true RUL at the cut) for testing, in four subsets:

| Subset | Train / test engines | Operating conditions | Fault modes | What it adds |
|---|---|---|---|---|
| FD001 | 100 / 100 | 1 | 1 (HPC) | baseline |
| FD002 | 260 / 259 | 6 | 1 | regime changes hide the wear trend |
| FD003 | 100 / 100 | 1 | 2 (HPC, fan) | two different degradation signatures |
| FD004 | 249 / 248 | 6 | 2 | both difficulties at once |

All four go through the same code path; the only per-subset setting is the
number of operating conditions (`configs/config.yaml`).

## 2. Data engineering

```
raw rows ─► validate_and_repair ─► regime (KMeans) ─► per-regime z-score ─► spike repair
        ─► EWMA smoothing ─► rolling mean/std/slope (10, 30) + drift from own baseline
```

* **Validation & repair** (`data/cleaning.py`). Schema check, numeric coercion,
  duplicate messages dropped (keep latest), out-of-order rows re-sorted, short
  gaps (≤5 cycles) interpolated *within the engine only*, unrepairable rows
  dropped. Every repaired value is counted per row (`imputed`) so the agent can
  route heavily repaired engines to a human. `inject_faults()` corrupts clean
  data on purpose and the tests prove the pipeline recovers it.
* **Operating regimes**. In FD002/FD004 the raw sensor value is dominated by the
  flight regime (altitude, Mach, throttle), not by wear. KMeans (k=6) on the
  three standardised settings recovers the regime exactly; each sensor is then
  z-scored with its own regime's mean/std. After this, all regimes overlay and
  the slow wear trend becomes visible.
* **Sensor selection**. Sensors that are constant (within every regime) or have
  |Spearman ρ| < 0.05 with RUL are dropped, learned from training data only
  (FD001 drops the 7 flat sensors documented by NASA).
* **Noise**. Isolated spikes (|z| > 6 inside a regime) are replaced by
  interpolation, then an EWMA (α = 0.3) smooths sensor noise.
* **Rolling features**. For each sensor: rolling mean, std and least-squares
  slope over 10 and 30 cycles (slope is computed in closed form from rolling
  sums, so it is vectorised), plus *drift from the engine's own first 10 cycles*
  (engines start with different manufacturing wear) and the cycle count.
* **Causality**. Every feature at cycle *t* uses only cycles ≤ *t*. A unit test
  asserts that featurising a truncated history gives exactly the same row as
  featurising the full history. This is what lets the same code power batch
  scoring, the streaming replay and the per-cycle RUL trajectory in the dashboard.
* **No leakage**. Train/validation splits are by *engine*, never by row, and the
  feature pipeline (regimes, statistics, sensor selection) is fitted on training
  engines only.

## 3. RUL model

* **Target**: piecewise-linear RUL capped at 125 cycles. Early in life there is
  no degradation signal, so asking a model to tell 250 from 200 cycles teaches it
  noise. The cap is standard for C-MAPSS.
* **Imbalance**: with the cap, most rows sit on the 125 plateau and few are near
  failure, which is where accuracy matters. Rows get weight
  `(1 / frequency of their RUL bin) ^ 0.5`, normalised to mean 1. The square root
  keeps the correction mild, as the brief asks.
* **Candidates**: Ridge (explainable baseline), Random Forest, XGBoost, LightGBM.
  Each is a nested MLflow run. The champion is picked on **validation RMSE**
  (validation engines, all cycles), never on the test set. Boosted models use
  early stopping on the validation engines.
* **Why not an LSTM**: gradient-boosted trees on good rolling features reach
  RMSE in the 12-16 range on FD001, which is where many published sequence models
  land. They train in seconds on a laptop, explain cleanly with SHAP, and keep the
  system lightweight. A sequence model is the natural next experiment; the
  `make_model` factory is the extension point.
* **Uncertainty**: cross-conformal intervals, calibrated separately per band of
  predicted RUL (0-30, 30-60, 60-90, 90+). The calibration residuals are
  computed at *truncation snapshots* (random cut points that mimic how the test
  set was built) and come from two sources: 5-fold out-of-fold predictions on
  the training engines (refits of the selected model, grouped by engine), plus
  the validation engines. The intervals are asymmetric and narrow near failure,
  where the signal is strong. Test-set coverage is reported so you can check
  that "80%" really means about 80%.
  *Lesson learned:* the first version calibrated on validation snapshots only,
  about 120 points from engines that early stopping had also seen. Bands fell
  back to a too-narrow global width and test coverage on FD001 was only 63%.
  Calibrating on out-of-fold residuals fixed this.
* **Metrics**: RMSE, MAE, R², the PHM08 **NASA score** (late predictions are
  penalised exponentially more than early ones, because a late prediction means
  a missed failure), bias, % of dangerously late predictions, error by RUL
  bucket, and interval coverage and width. Primary metrics use the true RUL
  capped at 125, the same as the target. Uncapped RMSE is reported alongside it
  for comparison with papers that do not cap.

## 4. Anomaly detection

All three detectors are fitted only on **healthy** rows (true RUL > 125) of the
training engines, using the smoothed regime-normalised sensors:

* **Mahalanobis distance** with a Ledoit-Wolf covariance estimate. This is the
  statistical baseline and it accounts for correlated sensors.
* **Isolation Forest**. Non-parametric, with no distribution assumption.
* **Autoencoder**. A small neural network (sensors → 12 → 4 → 12 → sensors,
  tanh) trained to reconstruct healthy readings through a 4-unit bottleneck.
  The score is the mean squared reconstruction error, so sensor patterns never
  seen in healthy engines rebuild badly. It can learn non-linear relationships
  between sensors that Mahalanobis (linear correlations only) cannot. It is
  built with scikit-learn's `MLPRegressor`, to avoid a deep-learning framework
  dependency for a model this small; a test shows it catches a "broken sensor
  relationship" where every sensor on its own is still in its normal range.

The alarm threshold is the 99th percentile of scores on *healthy validation
engines*, which fixes the false-alarm rate on healthy data instead of guessing a
`contamination` parameter. An alarm needs **3 consecutive** anomalous cycles
(persistence), which suppresses single-cycle noise. The three detectors are
compared on "is this row within 30 cycles of failure?" using PR-AUC, because the
classes are imbalanced, and the best one is kept per subset. Lead time (the RUL
at the first persistent alarm) is also reported.

An anomaly is not a failure prediction. It means the engine no longer looks
healthy. The agent uses the anomaly signal only together with RUL, or to send
an engine for human review when the two disagree.

## 5. Risk assessment (`models/risk.py`)

Two numbers summarise each engine for people who will not read an RUL interval:

* **Failure probability, P(fail within 30 cycles).** A logistic regression on
  three inputs the system already computes: predicted RUL / 125, log(1 + predicted
  RUL), and the anomaly index. It is fitted on the same held-out calibration
  snapshots as the conformal intervals (out-of-fold training predictions plus
  validation engines), so it learns how often the model's predictions are
  *actually* followed by failure. On the test fleet it is checked with ROC-AUC,
  the Brier score and the expected calibration error, and the dashboard shows a
  reliability curve (predicted % against observed %).
  *Why not derive it from the conformal interval?* The interval has only two
  quantiles per band. A direct probabilistic model is smoother, uses the anomaly
  signal too, and can be checked for calibration.
* **Health score (0-100).** A transparent formula, not a model:
  `100 × (predicted RUL / 125) × (0.75 + 0.25 × anomaly part)`, where the
  anomaly part is 1 below the alarm threshold and falls to 0 at three times the
  threshold. Abnormal sensor behaviour can only *lower* the score (by up to
  25%). The first version added the two parts instead, which gave engines a few
  cycles from failure a floor of about 25 points ("Poor" instead of "Critical").
  This was caught when inspecting the first scored fleet. The bands are Good ≥ 80, Fair ≥ 50, Poor ≥ 25 and Critical below
  that. It is validated by its rank correlation with the true RUL. It stays a
  formula on purpose: a maintenance manager can recompute it by hand, and it
  will never disagree with the RUL in a way nobody can explain.

The agent uses the probability as an extra HIGH trigger (≥ 50%). It catches
engines whose RUL point estimate looks fine but whose combination of RUL and
anomaly behaviour historically ends in failure soon.

## 6. Triage agent (`agent/triage.py`)

The agent is plain Python with no LLM or agent framework. The first matching
rule wins:

| # | Condition | Severity | Output |
|---|---|---|---|
| 1 | prediction not finite, or > 20% of recent readings repaired | REVIEW | human-review order |
| 1b | < 15 cycles of history: RUL ≤ high → REVIEW, otherwise LOW | REVIEW / LOW | |
| 2 | RUL ≤ 15 | CRITICAL | P1 corrective order, due now |
| 3 | RUL ≤ *high* **or** lower bound ≤ 15 **or** P(fail ≤ 30) ≥ 50% **or** (persistent anomaly and RUL ≤ 60) | HIGH | P2 scheduled order, due before lower bound − 5 |
| 4 | interval wider than 90 cycles | REVIEW | human-review order |
| 4b | persistent anomaly but RUL lower bound > 90 (models disagree) | REVIEW | human-review order |
| 5 | RUL ≤ 60 or persistent anomaly | MEDIUM | P3 inspection request |
| 6 | otherwise | LOW | keep monitoring, no order |

* **Calibrated thresholds**. *high* is not hand-picked. At training time it is
  set to the smallest value (within [25, 50]) that catches ≥ 95% of validation
  snapshots truly within 30 cycles of failure. It compensates for any optimistic
  bias of the model, and the chosen value and its validation precision are
  recorded in the model metadata.
* **Uncertainty-aware**. HIGH also fires when the *lower* interval bound is
  critical, even if the point estimate is not.
* **SHAP in every work order** (`explain.py`). For engines that get an order, the
  RUL prediction is explained with SHAP (TreeExplainer for the boosted and forest
  models, exact linear contributions for Ridge). The contributions of all ~8
  features of each sensor are summed, so the order lists the 3 *sensors* that
  pulled the predicted RUL down the most, in cycles (for example "phi −9 cycles").
  Contributions under half a cycle are treated as noise. SHAP is computed in one
  batched call, only for engines that get an order, so a whole fleet still
  scores in seconds. Two kinds of evidence sit side by side: *sensor deviations*
  (how far the readings are from the healthy baseline) and *model drivers*
  (what the model based its estimate on). Neither is a proven root cause, and
  the order says so.
* **Explainable**. Every decision carries machine-readable reason codes
  (`RUL_HIGH`, `ANOMALY_AND_RUL`, `MODEL_DISAGREEMENT`, ...). It also lists the
  three sensors deviating most from the healthy baseline, with their physical
  meaning and component, and suggested checks (a heuristic mapping from sensor
  location to component).
* **Work-order lifecycle** (`agent/work_orders.py`). There is one open order per
  engine. A repeat decision updates that order instead of creating a duplicate.
  A higher severity escalates it in place. An order is never downgraded
  automatically, and only a human closes or rejects it. Every change is appended
  to the order's history.
* **Storage** (`storage.py`). Work orders live in a SQLite database
  (`outputs/fleetguard.db`). Key fields (fleet, engine, status, severity, RUL,
  health, probability) are real, indexed columns, so they can be queried with
  SQL; the full order is kept as JSON next to them. The history is a separate
  `work_order_events` table. Every scoring run is recorded in `scoring_runs`,
  with each engine's prediction in `predictions`, so an engine's evolution
  across runs and model versions can be queried. Only *changed* orders are
  written, in one transaction, so the dashboard closing an order and the
  pipeline updating another never overwrite each other (tested). WAL mode
  lets the dashboard read while the pipeline writes. SQLite needs no server
  and ships with Python, so it fits a single-site deployment; moving to
  PostgreSQL would only touch this module. An older `work_orders.json` is
  imported automatically the first time.
* **Evaluated like a model**. On the test fleet we report recall on engines
  ≤ 30 cycles from failure, false-alarm rate on engines with > 90 cycles left,
  missed critical engines, and work-order precision.

## 7. MLOps

* **MLflow** (SQLite backend `mlflow.db`). Each training run is one MLflow run
  per subset and version, with params, metrics, evaluation plots and tables.
  Every candidate model is a nested run. The chosen estimator is registered in
  the MLflow Model Registry as `fleetguard-rul-<subset>`, and the alias
  `champion` moves when a version is promoted.
* **Local registry** (`models/<subset>/registry.json`). This is what the serving
  side loads. Each version is a self-contained bundle (feature pipeline + model +
  intervals + anomaly detector + triage thresholds + drift reference sample), so
  scoring never depends on the MLflow server.
* **Retraining flow**. New run-to-failure files dropped into `data/incoming/`
  start this sequence: PSI drift check of the new batch against the champion's
  reference → files archived and recorded in a data manifest → challenger
  trained on all data → challenger compared with the champion on the same fixed
  hold-out → promoted or rejected, with the reason stored. In the demo, v1 is
  trained on 70% of engines and the other 30% arrive later as a new batch.
* **Monitoring**. Each scoring run writes `monitoring.json`: severity counts,
  alarm rate, data-quality report, and the PSI of every input feature against
  training. The dashboard shows these together with the version history.

## 8. Simulation

**Live fleet replay.** The test fleet's telemetry is replayed tick by tick
(`fleetguard stream`). The clock is **end-aligned**: every engine's last
reading lands on the final tick ("now"), as in a real fleet where all engines
report up to the present but at different ages. Engines with short histories
therefore join the fleet later; they don't go silent early. (NASA cut each
test engine's data at a random point, so a start-aligned clock made engines
disappear one by one, which no real fleet does.) At every tick the agent
decides for each engine, and the work-order store creates, updates or
escalates orders exactly as in production. The final tick matches the Fleet
overview page. Each engine's state per tick is saved, so the
dashboard can play the fleet forward and show the work-order feed. True RUL is
attached only for display; the agent never sees it.

**Maintenance-policy simulation** (`simulation.py`). This answers the business
question behind the case study: *is predictive maintenance actually better than
what fleets do today?* Engines with known failure times are replayed cycle by
cycle under four policies:

| Policy | Service happens |
|---|---|
| Reactive | never: the engine runs until it fails |
| Fixed interval | every N cycles, whatever the condition |
| FleetGuard | when the agent raises a CRITICAL/HIGH order, plus a 5-cycle planning delay |
| Oracle | one cycle before the true failure (theoretical best) |

* **Cost per 1,000 operating cycles** (a renewal argument). Each engine life
  ends in a planned service (cost 1) or an unplanned failure (cost 10). Dividing
  by cycles operated means servicing too early is penalised automatically: the
  fleet would need more services over time.
* **No leakage.** The engines are the retraining batch, scored by **model v1**,
  which was trained before that batch arrived. If no batch exists, the
  champion's validation engines are used and the report says it is slightly
  optimistic.
* **A strong baseline.** The fixed interval is the one with the lowest cost on
  the engines v1 was trained on, which is the best a fixed schedule could do,
  not a strawman.
* **Assumptions are swept.** The failure:service cost ratio is unknown, so
  results are reported for 2×, 5×, 10×, 20× and 50×. FleetGuard beats the best
  fixed interval at every ratio, on every subset.
* **Result.** FleetGuard prevented every failure on all four subsets, used
  about 77-80% of each engine's life (fixed interval: 53-62%), and cost 25-36%
  less than the best fixed interval. It sits about 25% above the oracle, which
  is the room left for better models.

## 9. Serving and deployment

* **REST API** (`api.py`, FastAPI). It reuses exactly the same feature, model,
  agent and work-order code as the pipeline and dashboard, so the three can
  never disagree. `POST /predict` takes cycle-by-cycle readings for one or more
  engines and returns, for each engine's latest cycle, the RUL with its 80%
  interval, the health score, the failure probability, the severity with reason
  codes, sensor deviations and SHAP drivers. It can optionally create or update
  work orders. The model is picked automatically from the data (number of
  operating regimes), and a mismatching manual choice is flagged. Request and
  response schemas are Pydantic models, so `/docs` is generated automatically
  and invalid input gets a clear 422 error. Model bundles are cached and
  reloaded automatically after a promotion. A warm request with a 486-cycle
  history takes about 0.2 s, SHAP included.
* **Docker.** One image runs every role, chosen by its command: API, Streamlit
  dashboard, MLflow UI and the training pipeline. `docker-compose.yml` starts
  the first three, and the pipeline is an opt-in `train` profile. Models,
  outputs and MLflow state are bind-mounted from the project folder, so the
  stack serves whatever was trained locally. The image pins the exact library
  versions the models were trained with, which matters because model bundles
  are pickles; the container's predictions were verified to be identical to the
  host's. MLflow keeps its SQLite database inside `mlruns/` because Docker
  cannot reliably bind-mount a single file that may not exist yet. The image
  runs as a non-root user, has a health check, and is about 640 MB.
* **CI** builds the image, trains a tiny model inside it, starts the API and
  sends a real prediction request on every push.

## 10. Known limitations and next steps

* C-MAPSS is simulated. The physical component mapping in work orders is a
  heuristic, not a diagnosis.
* Promotion compares models on the official test set used as a fixed "golden"
  hold-out. In production this would be a separate, frozen evaluation set.
* The validation engines are used for early stopping, model selection and
  conformal calibration. Test-set coverage is the honest check of the intervals.
* Next experiments: a 1D-CNN or LSTM on raw windows, quantile regression
  instead of conformal intervals, a cost-based threshold optimisation (cost of
  a missed failure versus an early removal), and a real message queue in place
  of the replay.
