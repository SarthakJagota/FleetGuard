"""'Predict on new data' page: upload histories, add a reading, what-if simulator.

The model is chosen automatically by default:
* uploaded data   -> count the operating regimes in the file; one regime -> FD003
                     (else FD001), several regimes -> FD004 (else FD002)
* NASA test engine -> the model of the fleet the engine belongs to (exact match)
The user can override the choice; a mismatching override shows a warning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from fleetguard import inference
from fleetguard.data.loader import SENSOR_INFO, SETTING_COLS, sensor_label

AUTO = "Auto (recommended)"


@dataclass
class Model:
    subset: str
    version: str
    bundle: dict
    reason: str
    mismatch: bool = False

    @property
    def label(self) -> str:
        return f"{self.subset} model ({self.version}, {self.bundle['model'].name})"


def render(ctx) -> None:
    """ctx provides: subset, available, load(subset)->(bundle, version), fleet_raw, out_dir,
    style, severity_badge, series, muted."""
    st.title("Predict on new data")
    c1, c2 = st.columns([1, 2])
    choice = c1.selectbox("Model", [AUTO] + list(ctx.available), key="predict_model",
                          help="Auto picks the model from the data. Override only if you know better.")
    c2.info(
        "FleetGuard predicts from an engine's **history** (rolling averages, 10-30 cycle trends, drift from "
        "its own first cycles), so give it cycle-by-cycle readings. With fewer than 15 cycles the agent marks the "
        "result as unreliable.", icon="ℹ️")
    tab_up, tab_add, tab_wi = st.tabs(["📤 Upload engine history", "➕ Add a new reading", "🎛️ What-if simulator"])
    with tab_up:
        _upload_tab(ctx, choice)
    with tab_add:
        _add_reading_tab(ctx, choice)
    with tab_wi:
        _what_if_tab(ctx, choice)


# ---------------------------------------------------------------- model choice
def _resolve(ctx, choice: str, data: pd.DataFrame, home_subset: str | None) -> Model:
    """Decide which trained model scores `data`.

    home_subset is set when the engine comes from a benchmark fleet: its own model is the exact match.
    """
    multi = inference.is_multi_regime(data)
    if choice == AUTO:
        if home_subset and home_subset in ctx.available:
            subset, reason = home_subset, f"engine belongs to the {home_subset} fleet -> its own model"
        else:
            subset, reason = inference.suggest_subset(data, ctx.available)
    else:
        subset, reason = choice, "selected manually"
    bundle, version = ctx.load(subset)
    model_multi = bundle["features"].n_conditions > 1
    m = Model(subset, version, bundle, reason, mismatch=(multi != model_multi))
    if m.mismatch:
        m.reason += (f" - but the data has {inference.count_regimes(data)} operating regime(s) and this model expects "
                     f"{'several' if model_multi else 'one'}")
    return m


def _model_banner(m: Model) -> None:
    text = f"**Using the {m.label}** · {m.reason}"
    if m.mismatch:
        st.warning(text + ". Predictions may be unreliable; choose Auto or a matching model.", icon="⚠️")
    else:
        st.success(text, icon="🤖")


# ---------------------------------------------------------------- helpers
@st.cache_data(show_spinner=False)
def _read(data: bytes, name: str) -> pd.DataFrame:
    return inference.read_upload(data, name)


@st.cache_data(show_spinner=False)
def _score_upload(data: bytes, name: str, subset: str, version: str, _bundle):
    fleet, hist, report = inference.predict_upload(_bundle, _read(data, name))
    return fleet, hist, report


def _rul_chart(ctx, m: Model, hist: pd.DataFrame, title: str, extra: list | None = None, height: int = 360) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["cycle"], y=hist["rul_upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hist["cycle"], y=hist["rul_lower"], fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(42,120,214,0.18)", name="80% interval"))
    fig.add_trace(go.Scatter(x=hist["cycle"], y=hist["rul_pred"], line=dict(color=ctx.series[0], width=2), name="Predicted RUL"))
    for trace in extra or []:
        fig.add_trace(trace)
    th = m.bundle["thresholds"]
    for v, name in [(th.critical_rul, "critical"), (th.high_rul, "high")]:
        fig.add_hline(y=v, line_dash="dot", line_color=ctx.muted, annotation_text=name)
    fig.update_layout(title=title, xaxis_title="cycle", yaxis_title="cycles remaining", hovermode="x unified",
                      legend=dict(orientation="h", y=-0.2))
    return ctx.style(fig, height)


def _decision_card(ctx, d: dict, before: dict | None = None) -> None:
    c = st.columns(4)
    delta = None if before is None else f"{d['rul_pred'] - before['rul_pred']:+.1f} cycles"
    c[0].metric("Triage", ctx.severity_badge(d["severity"]),
                None if before is None or before["severity"] == d["severity"] else f"was {before['severity']}",
                delta_color="off")
    c[1].metric("Predicted RUL", f"{d['rul_pred']:.0f} cycles", delta)
    c[2].metric("80% interval", f"{d['rul_lower']:.0f} – {d['rul_upper']:.0f}")
    c[3].metric("Anomaly index", f"{d['anomaly_index']:.2f}", "persistent alarm" if d["anomaly_alarm"] else None,
                delta_color="inverse")
    c = st.columns(4)
    if pd.notna(d.get("health_score")):
        hd = None if before is None or pd.isna(before.get("health_score")) else f"{d['health_score'] - before['health_score']:+.0f}"
        c[0].metric("Health score", f"{d['health_score']:.0f} / 100 ({d.get('health_band', '')})", hd)
    if pd.notna(d.get("failure_prob")):
        pd_ = None if before is None or pd.isna(before.get("failure_prob")) else f"{(d['failure_prob'] - before['failure_prob']) * 100:+.0f} pts"
        c[1].metric("P(fail ≤30 cycles)", f"{d['failure_prob']:.0%}", pd_, delta_color="inverse")
    st.markdown(f"**Action:** {d['action']}" + (f" · due within **{int(d['due_within_cycles'])}** cycles"
                                                 if pd.notna(d.get("due_within_cycles")) else ""))
    st.markdown(f"**Rules fired:** `{d['reason_codes']}`"
                + (f"  \n**Sensor deviations:** {d['indicators']}" if d.get("indicators") else "")
                + (f"  \n**Model drivers (SHAP, cycles):** {d['model_drivers']}" if d.get("model_drivers") else ""))


def _engine_source(ctx, key: str) -> tuple[pd.DataFrame, str, str | None]:
    """Pick an engine from the benchmark fleet or the last upload. Returns (history, key, home subset)."""
    sources = [f"NASA test fleet {ctx.subset}"]
    if st.session_state.get("upload_raw") is not None:
        sources.append(f"Uploaded file ({st.session_state['upload_name']})")
    src = st.radio("Engine source", sources, horizontal=True, key=f"{key}_src")
    benchmark = src.startswith("NASA")
    raw = ctx.fleet_raw if benchmark else st.session_state["upload_raw"]
    engine = st.selectbox("Engine", sorted(raw["engine_id"].unique().tolist()), key=f"{key}_engine")
    hist = raw[raw["engine_id"] == engine].sort_values("cycle").reset_index(drop=True)
    tag = f"{ctx.subset}-{engine}" if benchmark else f"upload-{st.session_state['upload_name']}-{engine}"
    return hist, tag, (ctx.subset if benchmark else None)


# ---------------------------------------------------------------- tab 1: upload
def _upload_tab(ctx, choice: str) -> None:
    st.markdown("Upload cycle-by-cycle readings for one or more engines. Columns: `engine_id, cycle, setting_1-3, "
                "sensor_1-21` (aliases such as `unit_number`, `s4` are accepted), or a raw NASA `.txt` file.")
    c1, c2 = st.columns(2)
    c1.download_button("Download CSV template", inference.template().to_csv(index=False).encode(),
                       "fleetguard_template.csv", "text/csv")
    sample = ctx.fleet_raw[ctx.fleet_raw["engine_id"] <= 5].drop(columns=["rul"], errors="ignore")
    c2.download_button(f"Download sample ({ctx.subset}, 5 real engines)", sample.to_csv(index=False).encode(),
                       f"sample_{ctx.subset}.csv", "text/csv")

    up = st.file_uploader("Engine history file", type=["csv", "txt"])
    if up is None:
        return
    data = up.getvalue()
    try:
        raw = _read(data, up.name)
        m = _resolve(ctx, choice, raw, home_subset=None)
        fleet, hist, report = _score_upload(data, up.name, m.subset, m.version, m.bundle)
    except Exception as exc:
        st.error(f"Could not process the file: {exc}")
        return
    st.session_state["upload_raw"], st.session_state["upload_name"] = raw, up.name
    _model_banner(m)

    q = report
    st.markdown(f"**Data quality:** {q['rows_in']} rows in → {q['rows_out']} used · {q['engines']} engines · "
                f"{q['dropped_duplicates']} duplicates dropped · {q['imputed_values']} missing values filled · "
                f"{q['outliers_replaced']} spikes repaired · {q['reordered_engines']} engines re-ordered")

    counts = fleet["severity"].value_counts()
    cols = st.columns(5)
    for c, sev in zip(cols, ["CRITICAL", "HIGH", "REVIEW", "MEDIUM", "LOW"]):
        c.metric(ctx.severity_badge(sev), int(counts.get(sev, 0)))

    show = fleet.copy()
    show["severity"] = show["severity"].map(ctx.severity_badge)
    st.dataframe(show[["engine_id", "cycle", "severity", "health_score", "failure_prob", "rul_pred", "rul_lower", "rul_upper", "anomaly_index",
                       "due_within_cycles", "model_drivers", "indicators", "reason_codes"]].round(1),
                 hide_index=True, width="stretch")

    b1, b2 = st.columns(2)
    out = fleet.assign(model=f"{m.subset}-{m.version}")
    b1.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(),
                       f"predictions_{Path(up.name).stem}.csv", "text/csv")
    label = f"UPLOAD-{Path(up.name).stem}"
    if b2.button(f"Send to work-order system as fleet '{label}'"):
        store = ctx.store()
        sent, _, _ = inference.predict_upload(m.bundle, raw, store, fleet_label=label)
        store.save()
        store.db.record_scoring(label, f"{m.subset}-{m.version}", sent)
        st.success(f"Work orders created/updated for fleet {label} - see the Work orders page.")

    engine = st.selectbox("Inspect an engine", fleet["engine_id"].tolist(), key="up_engine")
    _decision_card(ctx, fleet[fleet["engine_id"] == engine].iloc[0].to_dict())
    st.plotly_chart(_rul_chart(ctx, m, hist[hist["engine_id"] == engine], f"Engine {engine}: RUL estimate over its history"),
                    width="stretch")


# ---------------------------------------------------------------- tab 2: add a reading
def _add_reading_tab(ctx, choice: str) -> None:
    st.markdown("Simulates a live sensor feed: take an engine's history, enter the readings of its **next cycle** "
                "and see how the prediction and the agent's decision change. Readings accumulate, so you can add "
                "several cycles in a row.")
    base, tag, home = _engine_source(ctx, "add")
    m = _resolve(ctx, choice, base, home)
    _model_banner(m)
    store_key = f"added::{tag}"
    added: list[dict] = st.session_state.setdefault(store_key, [])

    history = base
    for r in added:
        history = inference.append_reading(history, r)
    before = inference.latest_prediction(m.bundle, base)["decision"]
    current = inference.latest_prediction(m.bundle, history)

    last = history.iloc[-1]
    model_sensors = m.bundle["features"].sensors_
    with st.form(f"form_{store_key}"):
        st.markdown(f"**Readings for cycle {int(last['cycle']) + 1}** (pre-filled with cycle {int(last['cycle'])}; "
                    "to mimic wear, raise T50 / Ps30 / T30 slightly and lower phi)")
        values = {}
        cols = st.columns(4)
        for i, s in enumerate(model_sensors):
            short, desc, _ = SENSOR_INFO[s]
            values[s] = cols[i % 4].number_input(f"{short} ({s})", value=float(last[s]), format="%.4f", help=desc,
                                                 key=f"{store_key}_{s}_{len(added)}")
        with st.expander("Operating settings and sensors this model does not use"):
            cols2 = st.columns(4)
            others = SETTING_COLS + [s for s in SENSOR_INFO if s not in model_sensors]
            for i, s in enumerate(others):
                values[s] = cols2[i % 4].number_input(s, value=float(last[s]), format="%.4f",
                                                      key=f"{store_key}_{s}_{len(added)}")
        submitted = st.form_submit_button("Add reading and predict", type="primary")
    if submitted:
        added.append(values)
        st.rerun()

    c1, c2 = st.columns([3, 1])
    c1.markdown(f"**{len(added)} reading(s) added** to {len(base)} cycles of history.")
    if added and c2.button("Reset added readings", key=f"reset_{store_key}"):
        st.session_state[store_key] = []
        st.rerun()

    _decision_card(ctx, current["decision"], before if added else None)
    hist = current["history"]
    extra = []
    if added:
        new_rows = hist[hist["cycle"] > base["cycle"].max()]
        extra.append(go.Scatter(x=new_rows["cycle"], y=new_rows["rul_pred"], mode="markers",
                                marker=dict(color=ctx.series[1], size=10, line=dict(width=2, color="white")),
                                name="added reading(s)"))
    st.plotly_chart(_rul_chart(ctx, m, hist, "RUL estimate, including the added readings", extra), width="stretch")


# ---------------------------------------------------------------- tab 3: what-if
_PRESETS = {
    "HPC degradation": {"sensor_3": 3.0, "sensor_4": 3.0, "sensor_11": 3.0, "sensor_7": -3.0, "sensor_12": -3.0},
    "Fan degradation": {"sensor_8": 3.0, "sensor_13": 3.0, "sensor_15": 3.0, "sensor_2": 2.0},
    "Healthy (reset)": {},
}


def _what_if_tab(ctx, choice: str) -> None:
    st.markdown("Make the engine's recent readings drift and watch the RUL, the anomaly score and the agent react. "
                "Shifts are in **standard deviations within the engine's operating regime** and ramp up linearly "
                "over the last N cycles (wear is gradual).")
    base, tag, home = _engine_source(ctx, "wi")
    m = _resolve(ctx, choice, base, home)
    _model_banner(m)
    fp = m.bundle["features"]
    ranked = sorted(fp.sensors_, key=lambda s: -abs(fp.sensor_corr_.get(s, 0)))
    pk = f"wi::{m.subset}::{tag}"

    def apply_preset(name: str) -> None:
        for s in ranked:
            st.session_state[f"{pk}_{s}"] = float(_PRESETS[name].get(s, 0.0))

    pcols = st.columns(len(_PRESETS))
    for c, name in zip(pcols, _PRESETS):
        c.button(name, key=f"{pk}_preset_{name}", on_click=apply_preset, args=(name,), width="stretch")

    last_n = st.slider("Ramp the change over the last N cycles", 5, max(6, min(150, len(base))),
                       max(5, min(40, len(base))), key=f"{pk}_n")
    shifts = {}
    cols = st.columns(3)
    for i, s in enumerate(ranked):
        st.session_state.setdefault(f"{pk}_{s}", 0.0)  # value lives in session state so presets can set it
        shifts[s] = cols[i % 3].slider(sensor_label(s), -5.0, 5.0, step=0.5, key=f"{pk}_{s}")

    modified = inference.what_if(m.bundle, base, shifts, last_n)
    res = inference.compare(m.bundle, base, modified)
    _decision_card(ctx, res["after"], res["before"])

    left, right = st.columns(2)
    with left:
        bh = res["before_hist"]
        extra = [go.Scatter(x=bh["cycle"], y=bh["rul_pred"], line=dict(color=ctx.muted, width=2, dash="dash"),
                            name="Original prediction")]
        st.plotly_chart(_rul_chart(ctx, m, res["after_hist"], "RUL: scenario vs original", extra), width="stretch")
    with right:
        changed = [s for s, v in shifts.items() if v] or ranked[:1]
        s = max(changed, key=lambda x: abs(shifts.get(x, 0)))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=base["cycle"], y=base[s], line=dict(color=ctx.muted, width=2), name="original"))
        fig.add_trace(go.Scatter(x=modified["cycle"], y=modified[s], line=dict(color=ctx.series[1], width=2), name="scenario"))
        fig.update_layout(title=f"{sensor_label(s)}: raw reading", xaxis_title="cycle", hovermode="x unified",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(ctx.style(fig, 360), width="stretch")
    st.caption("This shows how the model and agent respond to sensor changes. It is a sensitivity tool, "
               "not a physical simulation of the engine.")
