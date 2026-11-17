"""FleetGuard monitoring dashboard.

Run:  streamlit run dashboard/app.py

Reads the files written by the pipeline (outputs/, models/) so it opens
instantly; heavier actions (re-score, retrain) call the same pipeline code.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dashboard"))

from fleetguard.agent.work_orders import WorkOrderStore, render_markdown  # noqa: E402
from fleetguard.config import SUBSETS, load_config  # noqa: E402
from fleetguard.data.loader import SENSOR_INFO, add_test_rul, load_subset, sensor_label  # noqa: E402
from fleetguard.mlops.registry import ModelRegistry  # noqa: E402

st.set_page_config(page_title="FleetGuard", page_icon="🛠️", layout="wide")

CFG = load_config()
OUT = CFG.path("outputs")
MODELS = CFG.path("models")

# Status palette (reserved for state) - always paired with a text label.
SEVERITY_COLORS = {"CRITICAL": "#d03b3b", "HIGH": "#ec835a", "MEDIUM": "#fab219", "REVIEW": "#6250d6", "LOW": "#0ca30c"}
SEVERITY_ORDER = ["CRITICAL", "HIGH", "REVIEW", "MEDIUM", "LOW"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
MUTED = "#898781"


def mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data
def read_csv(path: str, _m: float) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def read_json(path: str, _m: float):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def csv(path: Path) -> pd.DataFrame | None:
    return read_csv(str(path), mtime(path)) if path.exists() else None


def js(path: Path):
    return read_json(str(path), mtime(path)) if path.exists() else None


def wo_markdown(wo: dict) -> str:
    """Work order as Markdown, with headings scaled down to fit inside a dashboard page."""
    md = render_markdown(wo)
    return "### " + md[2:].replace("\n## ", "\n#### ") if md.startswith("# ") else md


def work_order_store() -> WorkOrderStore:
    """Work orders live in SQLite; an older work_orders.json is imported once."""
    return WorkOrderStore(CFG.path("database"), import_json=OUT / "work_orders.json")


@st.cache_resource
def bundle_for(subset: str, version: str):
    return ModelRegistry(MODELS, subset).load(version)


@st.cache_data
def fleet_raw(subset: str) -> pd.DataFrame:
    d = load_subset(CFG.path("raw_data"), subset)
    return add_test_rul(d["test"], d["test_rul"])


def available_subsets() -> list[str]:
    return [s for s in SUBSETS if (MODELS / s / "registry.json").exists() and (OUT / s / "fleet_latest.csv").exists()]


def style(fig: go.Figure, height: int = 360) -> go.Figure:
    # Margins leave room for axis titles and tick labels (10 px clipped them).
    fig.update_layout(height=height, margin=dict(l=70, r=20, t=50, b=60), legend_title_text="")
    fig.update_xaxes(showgrid=False, automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def severity_badge(sev: str) -> str:
    icon = {"CRITICAL": "⛔", "HIGH": "🔶", "MEDIUM": "🔍", "REVIEW": "👤", "LOW": "✅"}[sev]
    return f"{icon} {sev}"


# ------------------------------------------------------------------ sidebar
subsets = available_subsets()
st.sidebar.title("🛠️ FleetGuard")
st.sidebar.caption("Closed-loop predictive maintenance · NASA C-MAPSS")
if not subsets:
    st.title("FleetGuard")
    st.warning("No trained models found. Run the pipeline first:")
    st.code("python -m fleetguard run-all", language="bash")
    st.stop()

subset = st.sidebar.selectbox("Fleet (dataset subset)", subsets, help=(
    "FD001: 1 operating condition, 1 fault · FD002: 6 conditions, 1 fault · "
    "FD003: 1 condition, 2 faults · FD004: 6 conditions, 2 faults"))
page = st.sidebar.radio("Page", ["Fleet overview", "Engine detail", "Predict on new data", "Simulation",
                                 "Work orders", "Model performance", "MLOps & monitoring"])

registry = ModelRegistry(MODELS, subset)
champion = registry.champion
fleet = csv(OUT / subset / "fleet_latest.csv")
monitoring = js(OUT / subset / "monitoring.json") or {}
st.sidebar.divider()
st.sidebar.metric("Champion model", f"{champion}")
st.sidebar.caption(f"Scored with {monitoring.get('model_version', '?')} at {monitoring.get('scored_at', '?')}")

# ================================================================== OVERVIEW
if page == "Fleet overview":
    st.title(f"Fleet overview · {subset}")
    counts = fleet["severity"].value_counts()
    cols = st.columns(6)
    cols[0].metric("Engines", len(fleet))
    for c, sev in zip(cols[1:], SEVERITY_ORDER):
        c.metric(severity_badge(sev), int(counts.get(sev, 0)))

    left, right = st.columns([1, 1.4])
    with left:
        sev_df = pd.DataFrame({"severity": SEVERITY_ORDER, "engines": [int(counts.get(s, 0)) for s in SEVERITY_ORDER]})
        fig = px.bar(sev_df, x="engines", y="severity", orientation="h", color="severity", text="engines",
                     color_discrete_map=SEVERITY_COLORS, title="Engines by triage severity")
        fig.update_traces(textposition="outside", marker_cornerradius=4)
        fig.update_layout(showlegend=False, yaxis=dict(categoryorder="array", categoryarray=SEVERITY_ORDER[::-1], title=""))
        st.plotly_chart(style(fig), width="stretch")
    with right:
        fig = px.histogram(fleet, x="rul_pred", nbins=25, title="Predicted remaining useful life across the fleet",
                           color_discrete_sequence=[SERIES[0]], labels={"rul_pred": "Predicted RUL (cycles)"})
        th = bundle_for(subset, champion)["thresholds"]
        for v, name in [(th.critical_rul, "critical"), (th.high_rul, "high"), (th.medium_rul, "medium")]:
            fig.add_vline(x=v, line_dash="dot", line_color=MUTED, annotation_text=name, annotation_position="top")
        fig.update_traces(marker_line_width=2, marker_line_color="rgba(0,0,0,0)")
        fig.update_layout(yaxis_title="engines", bargap=0.08)
        st.plotly_chart(style(fig), width="stretch")

    st.subheader("Fleet status (sorted by urgency)")
    sel = st.multiselect("Severity", SEVERITY_ORDER, default=SEVERITY_ORDER)
    show = fleet[fleet["severity"].isin(sel)].copy()
    show["severity"] = show["severity"].map(severity_badge)
    show["RUL (80% interval)"] = show.apply(lambda r: f"{r.rul_pred:.0f}  [{r.rul_lower:.0f}–{r.rul_upper:.0f}]", axis=1)
    st.dataframe(
        show[[c for c in ["engine_id", "severity", "health_score", "failure_prob", "RUL (80% interval)", "true_rul",
                          "anomaly_index", "due_within_cycles", "model_drivers", "indicators", "reason_codes", "wo_id"]
               if c in show.columns]],
        hide_index=True, width="stretch",
        column_config={
            "engine_id": "Engine", "true_rul": st.column_config.NumberColumn("True RUL", help="Known only because this is benchmark data"),
            "health_score": st.column_config.ProgressColumn("Health", min_value=0, max_value=100, format="%.0f",
                                                            help="0-100: remaining life, reduced by up to 25% for abnormal sensor behaviour"),
            "failure_prob": st.column_config.NumberColumn("P(fail ≤30)", format="percent",
                                                          help="Calibrated probability of failing within 30 cycles"),
            "anomaly_index": st.column_config.ProgressColumn("Anomaly index", min_value=0, max_value=3, format="%.2f",
                                                             help="Score / alarm threshold. >1 = outside healthy behaviour"),
            "due_within_cycles": "Due (cycles)", "indicators": "Sensor deviations", "reason_codes": "Rules fired", "wo_id": "Work order",
            "model_drivers": st.column_config.TextColumn("Model drivers (SHAP)",
                                                         help="Sensors that lowered the predicted RUL, in cycles (work-order engines only)"),
        },
    )
    st.caption("RUL is an estimate from historical run-to-failure data; the interval is a calibrated 80% conformal interval.")

    if st.button("Re-score fleet and run the triage agent now"):
        from fleetguard.pipeline import FleetGuard

        with st.spinner("Scoring fleet..."):
            FleetGuard(CFG, track=False).score(subset)
        st.cache_data.clear()
        st.rerun()

# ================================================================== ENGINE
elif page == "Engine detail":
    st.title(f"Engine detail · {subset}")
    hist_all = csv(OUT / subset / "fleet_history.csv.gz")
    engine = st.selectbox("Engine", fleet["engine_id"].tolist(),
                          format_func=lambda e: f"Engine {e} — {fleet.set_index('engine_id').loc[e, 'severity']}")
    row = fleet.set_index("engine_id").loc[engine]
    hist = hist_all[hist_all["engine_id"] == engine]
    bundle = bundle_for(subset, champion)
    th = bundle["thresholds"]

    c = st.columns(4)
    c[0].metric("Triage", severity_badge(row["severity"]))
    c[1].metric("Predicted RUL", f"{row['rul_pred']:.0f} cycles")
    c[2].metric("80% interval", f"{row['rul_lower']:.0f} – {row['rul_upper']:.0f} cycles")
    c[3].metric("True RUL (benchmark)", f"{int(row['true_rul'])} cycles", help="Known only because this is benchmark data")
    c = st.columns(4)
    if pd.notna(row.get("health_score")):
        c[0].metric("Health score", f"{row['health_score']:.0f} / 100 · {row.get('health_band')}")
    if pd.notna(row.get("failure_prob")):
        c[1].metric("P(fail within 30 cycles)", f"{row['failure_prob']:.0%}")
    c[2].metric("Anomaly index", f"{row['anomaly_index']:.2f}", help="Score ÷ alarm threshold; above 1 = abnormal")
    c[3].metric("Persistent anomaly alarm", "yes" if row["anomaly_alarm"] else "no")

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["cycle"], y=hist["rul_upper"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=hist["cycle"], y=hist["rul_lower"], fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(42,120,214,0.18)", name="80% interval"))
        fig.add_trace(go.Scatter(x=hist["cycle"], y=hist["rul_pred"], line=dict(color=SERIES[0], width=2), name="Predicted RUL"))
        fig.add_trace(go.Scatter(x=hist["cycle"], y=hist["rul"].clip(upper=CFG["data"]["rul_cap"]),
                                 line=dict(color=SERIES[1], width=2, dash="dash"), name="True RUL (capped)"))
        for v, name in [(th.critical_rul, "critical"), (th.high_rul, "high")]:
            fig.add_hline(y=v, line_dash="dot", line_color=MUTED, annotation_text=name)
        fig.update_layout(title="RUL estimate over the engine's life", xaxis_title="cycle", yaxis_title="cycles remaining",
                          hovermode="x unified", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(style(fig, 380), width="stretch")
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hist["cycle"], y=hist["anomaly_index"], line=dict(color=SERIES[0], width=2), name="Anomaly index"))
        alarms = hist[hist["anomaly_alarm"]]
        fig.add_trace(go.Scatter(x=alarms["cycle"], y=alarms["anomaly_index"], mode="markers",
                                 marker=dict(color=SEVERITY_COLORS["CRITICAL"], size=8, symbol="x"), name="Persistent alarm"))
        fig.add_hline(y=1, line_dash="dot", line_color=MUTED, annotation_text="threshold")
        fig.update_layout(title=f"Degradation / anomaly score ({bundle['anomaly'].method})", xaxis_title="cycle",
                          yaxis_title="score ÷ threshold", hovermode="x unified", legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(style(fig, 380), width="stretch")

    if "health_score" in hist.columns:
        left, right = st.columns(2)
        with left:
            fig = go.Figure(go.Scatter(x=hist["cycle"], y=hist["health_score"], line=dict(color=SERIES[0], width=2),
                                       name="Health score"))
            for v, name in [(80, "Good"), (50, "Fair"), (25, "Poor")]:
                fig.add_hline(y=v, line_dash="dot", line_color=MUTED, annotation_text=name)
            fig.update_layout(title="Health score (0-100) over the engine's life", xaxis_title="cycle",
                              yaxis=dict(title="health", range=[0, 102]), hovermode="x unified")
            st.plotly_chart(style(fig, 320), width="stretch")
        with right:
            if hist["failure_prob"].notna().any():
                fig = go.Figure(go.Scatter(x=hist["cycle"], y=hist["failure_prob"], line=dict(color=SERIES[1], width=2),
                                           name="P(fail ≤30)"))
                fig.add_hline(y=getattr(th, "high_failure_prob", 0.5), line_dash="dot", line_color=MUTED, annotation_text="HIGH rule")
                fig.update_layout(title="Probability of failing within 30 cycles", xaxis_title="cycle",
                                  yaxis=dict(title="probability", range=[0, 1.02], tickformat=".0%"), hovermode="x unified")
                st.plotly_chart(style(fig, 320), width="stretch")

    st.subheader("Sensor trends")
    raw = fleet_raw(subset)
    eng_raw = raw[raw["engine_id"] == engine]
    feats = bundle["features"].transform(eng_raw)
    sensors = bundle["features"].sensors_
    default = [i.split(" ")[0] for i in str(row["indicators"]).split(", ") if i and i != "nan"]
    name_to_sensor = {SENSOR_INFO[s][0]: s for s in sensors}
    picked_default = [name_to_sensor[n] for n in default if n in name_to_sensor] or sensors[:3]
    picked = st.multiselect("Sensors", sensors, default=picked_default[:3], format_func=sensor_label, max_selections=3)
    view = st.radio("Show", ["Condition-normalised & smoothed (what the model sees)", "Raw readings"], horizontal=True)
    if picked:
        fig = go.Figure()
        for i, s in enumerate(picked):
            y = feats[f"{s}_ewm"] if view.startswith("Condition") else eng_raw[s]
            fig.add_trace(go.Scatter(x=feats["cycle"], y=y, name=sensor_label(s), line=dict(color=SERIES[i], width=2)))
        fig.update_layout(xaxis_title="cycle", yaxis_title="z-score vs. regime" if view.startswith("Condition") else "reading",
                          hovermode="x unified", legend=dict(orientation="h", y=-0.25))
        st.plotly_chart(style(fig, 360), width="stretch")
        if not view.startswith("Condition") and bundle["features"].n_conditions > 1:
            st.caption("Raw readings jump between the 6 flight regimes; normalising per regime reveals the wear trend.")

    st.subheader("Agent decision")
    store = work_order_store()
    wo = store.get(row["wo_id"]) if isinstance(row.get("wo_id"), str) else None
    if wo:
        st.markdown(wo_markdown(wo))
    else:
        st.success(f"{severity_badge(row['severity'])} — {row['action']}  \nRules fired: `{row['reason_codes']}`")

    with st.expander("Why did the model predict this? (SHAP, latest cycle)"):
        try:
            import shap

            model = bundle["model"]
            X = feats[model.feature_names].tail(1)
            if hasattr(model.estimator, "feature_importances_"):
                sv = shap.TreeExplainer(model.estimator)(X).values[0]
            else:
                sv = shap.LinearExplainer(model.estimator.named_steps["ridge"],
                                          model.estimator.named_steps["standardscaler"].transform(feats[model.feature_names]))(
                    model.estimator.named_steps["standardscaler"].transform(X)).values[0]
            contrib = pd.DataFrame({"feature": model.feature_names, "shap": sv})
            contrib["sensor"] = contrib["feature"].str.extract(r"(sensor_\d+)")[0].fillna("cycle")
            by_sensor = contrib.groupby("sensor")["shap"].sum().sort_values()
            by_sensor.index = [sensor_label(s) if s != "cycle" else "cycle (age)" for s in by_sensor.index]
            df = by_sensor.reset_index().rename(columns={"index": "sensor"})
            df["effect"] = np.where(df["shap"] < 0, "lowers RUL", "raises RUL")
            fig = px.bar(df, x="shap", y="sensor", orientation="h", color="effect",
                         color_discrete_map={"lowers RUL": SERIES[1], "raises RUL": SERIES[0]},
                         title="Contribution to predicted RUL, summed over each sensor's features")
            fig.update_traces(marker_cornerradius=4)
            st.plotly_chart(style(fig, 420), width="stretch")
            st.caption("SHAP values explain the model's output, not physical causation.")
        except Exception as exc:  # explanation is optional
            st.info(f"SHAP explanation unavailable: {exc}")

# ================================================================== WORK ORDERS
elif page == "Work orders":
    st.title("Work orders")
    st.caption("Created automatically by the triage agent. Humans acknowledge, close or reject them; "
               "`HUMAN_REVIEW` orders are the exceptions the agent could not decide on its own.")
    store = work_order_store()
    if not store.orders:
        st.info("No work orders yet.")
        st.stop()
    df = pd.DataFrame(store.orders)
    f1, f2, f3 = st.columns(3)
    subs = f1.multiselect("Subset", sorted(df["subset"].unique()), default=[subset])
    stat = f2.multiselect("Status", ["OPEN", "ACKNOWLEDGED", "CLOSED", "REJECTED"], default=["OPEN", "ACKNOWLEDGED"])
    sevs = f3.multiselect("Severity", SEVERITY_ORDER[:-1], default=SEVERITY_ORDER[:-1])
    view = df[df["subset"].isin(subs) & df["status"].isin(stat) & df["severity"].isin(sevs)].copy()
    view["_o"] = view["severity"].map({s: i for i, s in enumerate(SEVERITY_ORDER)})
    view = view.sort_values(["_o", "rul_pred"])
    c = st.columns(4)
    c[0].metric("Shown", len(view))
    c[1].metric("P1 (critical)", int((view["priority"] == "P1").sum()))
    c[2].metric("Need human review", int(view["requires_human"].sum()))
    c[3].metric("Escalated at least once", int(view["history"].apply(lambda h: any(e["event"] == "ESCALATED" for e in h)).sum()))
    view["severity"] = view["severity"].map(severity_badge)
    st.dataframe(view[["wo_id", "subset", "engine_id", "severity", "priority", "type", "status", "rul_pred",
                       "due_within_cycles", "title"]], hide_index=True, width="stretch",
                 column_config={"rul_pred": st.column_config.NumberColumn("RUL", format="%.0f"), "due_within_cycles": "Due (cycles)"})

    if len(view):
        wo_id = st.selectbox("Open a work order", view["wo_id"].tolist())
        wo = store.get(wo_id)
        left, right = st.columns([2, 1])
        with left:
            st.markdown(wo_markdown(wo))
        with right:
            st.markdown("#### Engineer actions")
            note = st.text_input("Note", placeholder="e.g. borescope done, compressor washed")
            for status, label in [("ACKNOWLEDGED", "Acknowledge"), ("CLOSED", "Close (work done)"),
                                  ("REJECTED", "Reject (not needed)")]:
                if st.button(label, key=status, width="stretch"):
                    store.set_status(wo_id, status, note, by="dashboard")
                    store.save()
                    st.rerun()
            st.markdown("#### History")
            st.dataframe(pd.DataFrame(wo["history"]), hide_index=True, width="stretch")

# ================================================================== PERFORMANCE
elif page == "Model performance":
    st.title(f"Model performance · {subset}")
    meta = js(OUT / subset / "evaluation" / "metrics.json")
    m = meta["metrics"]
    st.caption(f"Champion {meta['version']}: **{meta['model_name']}** · {meta['trained_engines']} training engines · "
               f"evaluated on the official NASA test set (last cycle of each engine). RUL targets capped at {CFG['data']['rul_cap']}.")
    c = st.columns(6)
    c[0].metric("RMSE", f"{m['test_rmse']:.2f}")
    c[1].metric("MAE", f"{m['test_mae']:.2f}")
    c[2].metric("R²", f"{m['test_r2']:.3f}")
    c[3].metric("NASA score", f"{m['test_nasa_score']:.0f}", help="Asymmetric: late predictions penalised more. Lower is better.")
    c[4].metric("80% interval coverage", f"{m['test_interval_coverage']:.0%}", help=f"Avg width {m['test_interval_width']:.0f} cycles")
    c[5].metric("Uncapped RMSE", f"{m['test_uncapped_rmse']:.2f}", help="Against raw true RUL; engines above the cap inflate this")

    tp = csv(OUT / subset / "evaluation" / "test_predictions.csv")
    tp["true_capped"] = tp["true_rul"].clip(upper=CFG["data"]["rul_cap"])
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for sev in SEVERITY_ORDER:
            d = tp[tp["severity"] == sev]
            if len(d):
                fig.add_trace(go.Scatter(
                    x=d["true_capped"], y=d["rul_pred"], mode="markers", name=sev,
                    marker=dict(color=SEVERITY_COLORS[sev], size=8, line=dict(width=1, color="white")),
                    error_y=dict(type="data", symmetric=False, array=d["rul_upper"] - d["rul_pred"],
                                 arrayminus=d["rul_pred"] - d["rul_lower"], thickness=0.8, width=0, color="rgba(137,135,129,0.5)"),
                    customdata=d["engine_id"], hovertemplate="engine %{customdata}<br>true %{x}<br>pred %{y:.1f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[0, 125], y=[0, 125], mode="lines", line=dict(color=MUTED, dash="dash", width=1), name="perfect"))
        fig.update_layout(title="Predicted vs true RUL (with 80% intervals, coloured by triage)",
                          xaxis_title="true RUL (capped)", yaxis_title="predicted RUL")
        st.plotly_chart(style(fig, 420), width="stretch")
    with right:
        eb = csv(OUT / subset / "evaluation" / "error_buckets.csv")
        fig = px.bar(eb, x="bucket", y="mae", text=eb["mae"].round(1), title="Mean absolute error by true-RUL range",
                     color_discrete_sequence=[SERIES[0]], labels={"bucket": "true RUL (cycles)", "mae": "MAE (cycles)"},
                     hover_data={"n": True, "bias": ":.1f"})
        fig.update_traces(marker_cornerradius=4, textposition="outside")
        st.plotly_chart(style(fig, 420), width="stretch")
        st.caption("Errors near failure matter most for maintenance and are the smallest - "
                   "far from failure the degradation signal is weak (the 'healthy plateau').")

    st.subheader("Model comparison (validation engines = selection, test = report)")
    mc = csv(OUT / subset / "evaluation" / "model_comparison.csv")
    st.dataframe(mc[["model", "description", "val_rmse", "val_snapshot_rmse", "test_rmse", "test_mae", "test_nasa_score",
                     "best_iteration", "fit_seconds"]].round(2), hide_index=True, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Anomaly detection")
        rows = []
        for det in ("mahalanobis", "iforest", "autoencoder"):
            if f"anomaly_{det}_pr_auc" not in m:
                continue  # model trained before this detector existed
            rows.append({"detector": det + (" ✓ selected" if det == meta["selected_detector"] else ""),
                         "PR-AUC": m.get(f"anomaly_{det}_pr_auc"), "ROC-AUC": m.get(f"anomaly_{det}_roc_auc"),
                         "recall": m.get(f"anomaly_{det}_recall"), "precision": m.get(f"anomaly_{det}_precision"),
                         "median lead (cycles)": m.get(f"anomaly_{det}_median_lead_cycles")})
        st.dataframe(pd.DataFrame(rows).round(3), hide_index=True, width="stretch")
        st.caption(f"Task: flag rows within {CFG['anomaly']['degraded_rul']} cycles of failure (positive rate "
                   f"{m.get('anomaly_mahalanobis_positive_rate', 0):.0%}, imbalanced → PR-AUC, not accuracy). "
                   "Lead time = RUL when the first persistent alarm fired on validation engines.")
    with right:
        st.subheader("Triage agent on the test fleet")
        c = st.columns(2)
        c[0].metric("Recall near failure (≤30)", f"{m['agent_recall_near_failure']:.0%}")
        c[1].metric("False-alarm rate (>90 left)", f"{m['agent_false_alarm_rate']:.0%}")
        c = st.columns(2)
        c[0].metric("Missed critical (≤15 left)", m["agent_missed_critical"])
        c[1].metric("Work-order precision (≤60 left)", f"{m['agent_work_order_precision']:.0%}")
        cal = meta.get("triage_calibration") or {}
        if cal:
            st.caption(f"`high_rul` threshold calibrated on validation engines to {cal['high_rul']:.0f} cycles "
                       f"(target recall {cal['target_recall']:.0%}, achieved {cal['val_recall']:.0%}).")

    calib = csv(OUT / subset / "evaluation" / "failure_prob_calibration.csv")
    if calib is not None and "test_failure_prob_brier" in m:
        st.subheader("Risk assessment: failure probability & health score")
        left, right = st.columns([1, 1.3])
        with left:
            c = st.columns(2)
            c[0].metric("ROC-AUC", f"{m.get('test_failure_prob_auc', float('nan')):.3f}",
                        help="Ranks engines that fail within 30 cycles above those that don't")
            c[1].metric("Brier score", f"{m['test_failure_prob_brier']:.3f}",
                        help=f"Mean squared error of the probability (lower is better). Base rate "
                             f"{m['test_failure_prob_base_rate']:.0%}")
            c = st.columns(2)
            c[0].metric("Calibration error (ECE)", f"{m['test_failure_prob_ece']:.3f}",
                        help="Average gap between predicted probability and observed frequency")
            c[1].metric("Health ↔ true RUL", f"ρ = {m['test_health_spearman_rul']:.2f}",
                        help="Spearman rank correlation of the health score with the true remaining life")
            rm = meta.get("risk_model", {})
            st.caption(f"Logistic model on predicted RUL + anomaly index, fitted on {rm.get('horizon', 30)}-cycle "
                       "labels of held-out calibration snapshots. Health = RUL share × (0.75 + 0.25 × normal-behaviour share).")
        with right:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=MUTED, dash="dash", width=1),
                                     name="perfectly calibrated"))
            # Markers only: some bins hold 1-2 engines, so a connecting line would suggest a pattern that isn't there.
            fig.add_trace(go.Scatter(x=calib["predicted"], y=calib["observed"], mode="markers",
                                     marker=dict(size=calib["n"].clip(lower=4) ** 0.5 * 3, color=SERIES[0],
                                                 line=dict(width=1, color="white")), name="FleetGuard",
                                     customdata=calib["n"], hovertemplate="predicted %{x:.0%}<br>observed %{y:.0%}<br>n=%{customdata}<extra></extra>"))
            fig.update_layout(title="Reliability curve on the test fleet (marker size = engines)",
                              xaxis=dict(title="predicted P(fail ≤30)", tickformat=".0%", range=[0, 1]),
                              yaxis=dict(title="observed fraction failing ≤30", tickformat=".0%", range=[0, 1.02]))
            st.plotly_chart(style(fig, 360), width="stretch")

    fi = csv(OUT / subset / "evaluation" / "feature_importance.csv")
    def short_feature(f: str) -> str:  # sensor_11_mean30 -> Ps30_mean30
        parts = f.split("_")
        sensor = "_".join(parts[:2])
        return f"{SENSOR_INFO[sensor][0]}_{'_'.join(parts[2:])}" if sensor in SENSOR_INFO else f

    fi["label"] = fi["feature"].apply(short_feature)
    fig = px.bar(fi.head(20).iloc[::-1], x="importance", y="label", orientation="h", title="Top 20 features",
                 color_discrete_sequence=[SERIES[0]], labels={"label": ""})
    fig.update_traces(marker_cornerradius=4)
    st.plotly_chart(style(fig, 520), width="stretch")

    summary = js(OUT / "summary.json")
    if summary:
        st.subheader("All subsets")
        rows = [{"subset": s, "champion": r["champion"], **{k: round(v, 3) for k, v in r["metrics"].items()}} for s, r in summary.items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

# ================================================================== PREDICT
elif page == "Predict on new data":
    from types import SimpleNamespace

    import predict_page

    def load_champion(s: str):
        version = ModelRegistry(MODELS, s).champion
        return bundle_for(s, version), version

    predict_page.render(SimpleNamespace(
        subset=subset, available=subsets, load=load_champion, fleet_raw=fleet_raw(subset),
        out_dir=OUT, store=work_order_store, style=style, severity_badge=severity_badge, series=SERIES, muted=MUTED,
    ))

# ================================================================== SIMULATION
elif page == "Simulation":
    from types import SimpleNamespace

    import simulation_page

    simulation_page.render(SimpleNamespace(
        subset=subset, out_dir=OUT / subset, style=style, severity_badge=severity_badge,
        colors=SEVERITY_COLORS, series=SERIES, muted=MUTED,
    ))

# ================================================================== MLOPS
elif page == "MLOps & monitoring":
    st.title(f"MLOps & monitoring · {subset}")
    versions = pd.DataFrame(registry.versions())
    versions["test_rmse"] = versions["metrics"].apply(lambda m: m.get("test_rmse"))
    versions["test_nasa_score"] = versions["metrics"].apply(lambda m: m.get("test_nasa_score"))
    versions["trained_engines"] = versions["trained_engines"].astype(int)
    st.subheader("Model registry")
    st.dataframe(versions[["version", "status", "model_name", "trained_engines", "test_rmse", "test_nasa_score",
                           "created_at", "decision"]].round(2), hide_index=True, width="stretch")
    st.caption("Champion/challenger: a retrained model is promoted only if its test RMSE is not worse than the champion's. "
               "Full experiment history (every candidate model, params, plots) is in MLflow: "
               "`mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`, then open http://127.0.0.1:5000 (or `docker compose up`)")

    left, right = st.columns(2)
    with left:
        fig = px.line(versions, x="version", y="test_rmse", markers=True, title="Test RMSE by model version",
                      color_discrete_sequence=[SERIES[0]], labels={"test_rmse": "test RMSE (cycles)", "version": "model version"})
        fig.update_traces(line_width=2, marker_size=9)
        st.plotly_chart(style(fig, 320), width="stretch")
    with right:
        drift_new = csv(OUT / subset / "drift_new_data.csv")
        st.markdown("**Drift check on the last incoming batch** (PSI vs training reference)")
        if drift_new is not None:
            st.dataframe(drift_new.head(8), hide_index=True, width="stretch")
        else:
            st.info("No incoming batch processed yet.")

    st.subheader("Live fleet input drift")
    drift = csv(OUT / subset / "drift_fleet.csv")
    if drift is not None:
        d = drift.head(15).copy()
        d["label"] = d["feature"].apply(lambda f: sensor_label(f.replace("_ewm", "")) if f.startswith("sensor") else f)
        status_colors = {"stable": SEVERITY_COLORS["LOW"], "moderate": SEVERITY_COLORS["MEDIUM"], "drift": SEVERITY_COLORS["CRITICAL"]}
        fig = px.bar(d.iloc[::-1], x="psi", y="label", color="status", orientation="h", color_discrete_map=status_colors,
                     title="Population Stability Index: scored fleet vs training data", labels={"label": ""})
        fig.add_vline(x=0.1, line_dash="dot", line_color=MUTED, annotation_text="0.1")
        fig.add_vline(x=CFG["retraining"]["psi_alert"], line_dash="dot", line_color=MUTED, annotation_text="alert")
        fig.update_traces(marker_cornerradius=4)
        st.plotly_chart(style(fig, 440), width="stretch")
        st.caption("Some shift is expected here: test engines are truncated earlier in life than the run-to-failure "
                   "training engines, so smoothed sensor levels sit closer to healthy.")

    c = st.columns(2)
    with c[0]:
        st.markdown("**Data quality of the last scoring run**")
        st.json(monitoring.get("data_quality", {}), expanded=False)
    with c[1]:
        st.markdown("**Streaming replay** (fleet telemetry replayed cycle by cycle)")
        ev = csv(OUT / subset / "stream_events.csv")
        if ev is not None and len(ev):
            agg = ev.groupby(["tick", "severity"]).size().reset_index(name="events")
            fig = px.bar(agg, x="tick", y="events", color="severity", color_discrete_map=SEVERITY_COLORS,
                         category_orders={"severity": SEVERITY_ORDER}, title="Work orders created / escalated over time")
            fig.update_layout(bargap=0.15, xaxis_title="fleet time (cycle)")
            st.plotly_chart(style(fig, 320), width="stretch")

    st.subheader("Database")
    store = work_order_store()
    counts = store.db.counts()
    c = st.columns(4)
    for col, (table, n) in zip(c, counts.items()):
        col.metric(table.replace("_", " "), f"{n:,}")
    runs = store.db.scoring_runs(subset)
    if len(runs):
        runs["severity_counts"] = runs["severity_counts"].apply(lambda j: ", ".join(f"{k} {v}" for k, v in json.loads(j).items()))
        st.dataframe(runs.head(10), hide_index=True, width="stretch")
    st.caption(f"SQLite file `{CFG.path('database').relative_to(ROOT).as_posix()}`: work orders with their full history, "
               "and every scoring run with each engine's prediction, so trends can be queried with SQL.")

    st.subheader("Retraining")
    incoming = sorted(p.name for p in CFG.path("incoming_data").glob(f"{subset}_*"))
    st.write(f"Files waiting in `data/incoming/`: {incoming or 'none'}")
    st.caption("Drop a new run-to-failure file named `<SUBSET>_<anything>.csv` (columns engine_id, cycle, setting_1-3, sensor_1-21) "
               "or `.txt` (NASA format) into data/incoming, then retrain.")
    if st.button("Retrain now (drift check → train challenger → compare → promote/reject → re-score)", disabled=not incoming):
        from fleetguard.pipeline import FleetGuard

        with st.spinner("Retraining - this takes a few minutes..."):
            fg = FleetGuard(CFG)
            result = fg.retrain(subset)
            fg.score(subset)
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success(result["decision"])
