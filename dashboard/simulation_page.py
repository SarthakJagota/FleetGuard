"""'Simulation' page: live fleet replay and maintenance-policy comparison."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

SEVERITIES = ["CRITICAL", "HIGH", "REVIEW", "MEDIUM", "LOW"]
POLICY_LABELS = {"reactive": "Reactive (run to failure)", "fixed": "Fixed interval", "fleetguard": "FleetGuard",
                 "oracle": "Oracle (perfect knowledge)"}


def render(ctx) -> None:
    """ctx provides: subset, out_dir, style, severity_badge, colors, series, muted."""
    st.title(f"Simulation · {ctx.subset}")
    tab_live, tab_policy = st.tabs(["🎬 Live fleet replay", "💰 Maintenance policies"])
    with tab_live:
        _live_replay(ctx)
    with tab_policy:
        _policies(ctx)


# ================================================================ live replay
@st.cache_data(show_spinner=False)
def _load_replay(states_path: str, events_path: str, _mtime: float):
    states = pd.read_csv(states_path)
    events = pd.read_csv(events_path) if events_path else pd.DataFrame()
    ticks = sorted(states["tick"].unique())
    engines = sorted(states["engine_id"].unique())
    # Severity counts at every tick (each engine keeps its last known state).
    counts = []
    latest = {}
    for t, grp in states.groupby("tick", sort=True):
        for row in grp.itertuples():
            latest[row.engine_id] = row.severity
        c = pd.Series(list(latest.values())).value_counts()
        counts.append({"tick": t, **{s: int(c.get(s, 0)) for s in SEVERITIES}})
    return states, events, ticks, engines, pd.DataFrame(counts)


def _snapshot(states: pd.DataFrame, tick: int) -> pd.DataFrame:
    """Latest state of every engine that has joined the fleet by `tick`."""
    return states[states["tick"] <= tick].groupby("engine_id").tail(1).copy()


def _grid(ctx, snap: pd.DataFrame, engines: list, tick: int) -> go.Figure:
    per_row = 20 if len(engines) <= 140 else 26
    pos = {e: (i % per_row, -(i // per_row)) for i, e in enumerate(engines)}
    fig = go.Figure()
    waiting = [e for e in engines if e not in set(snap["engine_id"])]
    if waiting:  # engines that have not started reporting yet: empty outlines
        xy = np.array([pos[e] for e in waiting])
        fig.add_trace(go.Scatter(
            x=xy[:, 0], y=xy[:, 1], mode="markers", name="not yet reporting", hoverinfo="text",
            hovertext=[f"engine {e}: joins the fleet later" for e in waiting],
            marker=dict(symbol="square-open", size=20, color="rgba(137,135,129,0.45)", line=dict(width=1.5)),
        ))
    for sev in SEVERITIES:
        d = snap[snap["severity"] == sev]
        if d.empty:
            continue
        xy = np.array([pos[e] for e in d["engine_id"]])
        hover = [
            f"engine {r.engine_id} · age {r.cycle} cycles<br>{sev} · predicted RUL {r.rul_pred:.0f} (true {r.true_rul})"
            f"<br>health {r.health_score:.0f} · P(fail ≤30) {r.failure_prob:.0%}"
            for r in d.itertuples()
        ]
        fig.add_trace(go.Scatter(
            x=xy[:, 0], y=xy[:, 1], mode="markers", name=sev, hovertext=hover, hoverinfo="text",
            marker=dict(symbol="square", size=22, color=ctx.colors[sev], line=dict(width=2, color="rgba(255,255,255,0.9)")),
        ))
    rows = (len(engines) - 1) // per_row + 1
    fig.update_layout(
        title=f"Fleet at time {tick}: {len(snap)} of {len(engines)} engines reporting", height=max(260, 46 * rows + 110),
        xaxis=dict(visible=False, range=[-1, per_row]), yaxis=dict(visible=False, range=[-rows, 1]),
        legend=dict(orientation="h", y=-0.05), margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _counts_chart(ctx, counts: pd.DataFrame, tick: int) -> go.Figure:
    fig = go.Figure()
    for sev in ["LOW", "MEDIUM", "REVIEW", "HIGH", "CRITICAL"]:
        fig.add_trace(go.Scatter(x=counts["tick"], y=counts[sev], name=sev, stackgroup="one", mode="lines",
                                 line=dict(width=0.5, color=ctx.colors[sev])))
    fig.add_vline(x=tick, line_dash="dot", line_color=ctx.muted)
    fig.update_layout(title="Engines reporting, by severity, over fleet time", xaxis_title="fleet time (last tick = now)",
                      yaxis_title="engines", hovermode="x unified", legend=dict(orientation="h", y=-0.3))
    fig = ctx.style(fig, 320)
    fig.update_layout(margin=dict(l=60, b=20))  # room for the y-axis labels next to a bottom legend
    return fig


def _render_tick(ctx, slots, states, events, engines, counts, tick):
    snap = _snapshot(states, tick)
    c = snap["severity"].value_counts()
    with slots["metrics"].container():
        cols = st.columns(6)
        cols[0].metric("Fleet time", f"{tick} / {states['tick'].max()}")
        if tick == states["tick"].max():
            cols[0].caption("now")
        for col, sev in zip(cols[1:], SEVERITIES):
            col.metric(ctx.severity_badge(sev), int(c.get(sev, 0)))
    slots["grid"].plotly_chart(_grid(ctx, snap, engines, tick), width="stretch", key=f"grid_{tick}_{time.time()}")
    with slots["feed"].container():
        st.markdown("**Work-order feed**")
        recent = events[events["tick"] <= tick].tail(10).iloc[::-1] if len(events) else events
        if recent.empty:
            st.caption("No work orders yet.")
        for r in recent.itertuples():
            icon = "🆕" if r.event == "CREATED" else "⬆️"
            st.markdown(f"{icon} t={r.tick} · **{r.wo_id}** engine {r.engine_id} → "
                        f"{ctx.severity_badge(r.severity)} (RUL {r.rul_pred:.0f})")
    slots["counts"].plotly_chart(_counts_chart(ctx, counts, tick), width="stretch", key=f"counts_{tick}_{time.time()}")


def _live_replay(ctx) -> None:
    states_path = ctx.out_dir / "stream_states.csv.gz"
    events_path = ctx.out_dir / "stream_events.csv"
    if not states_path.exists():
        st.info(f"No replay yet. Run `python -m fleetguard stream --subset {ctx.subset}`.")
        return
    states, events, ticks, engines, counts = _load_replay(
        str(states_path), str(events_path) if events_path.exists() else "", states_path.stat().st_mtime)
    st.markdown(
        f"The {len(engines)} test engines' telemetry is replayed as if it arrived live. Like a real fleet, every engine "
        "reports up to **now** (the last tick) but engines have different ages, so engines with a short history join "
        "the fleet later. At every tick the triage agent decides for each engine, and work orders are created or "
        "escalated exactly as in production (one open order per engine, never auto-downgraded). The last tick matches "
        "the Fleet overview page.")

    key = f"replay_tick_{ctx.subset}"
    st.session_state.setdefault(key, ticks[min(len(ticks) - 1, (2 * len(ticks)) // 3)])
    c1, c2, c3 = st.columns([5, 1, 1])
    tick = c1.select_slider("Fleet time (last tick = now)", options=ticks, key=key)
    speed = c3.selectbox("Speed", [0.15, 0.4, 0.8], index=1, format_func=lambda s: {0.15: "fast", 0.4: "normal", 0.8: "slow"}[s])
    play = c2.button("▶ Play", width="stretch", help="Animate from the current time to the end")

    slots = {"metrics": st.empty()}
    left, right = st.columns([2.2, 1])
    with left:
        slots["grid"] = st.empty()
    with right:
        slots["feed"] = st.empty()
    slots["counts"] = st.empty()

    if play:
        for t in [x for x in ticks if x >= tick]:
            _render_tick(ctx, slots, states, events, engines, counts, t)
            time.sleep(speed)
        st.caption("Replay finished. Move the slider to explore any moment.")
    else:
        _render_tick(ctx, slots, states, events, engines, counts, tick)


# ================================================================ policies
def _policies(ctx) -> None:
    sim_dir = ctx.out_dir / "simulation"
    path = sim_dir / "policy_summary.json"
    if not path.exists():
        st.info(f"No simulation yet. Run `python -m fleetguard simulate-policies --subset {ctx.subset}`.")
        return
    summary = json.loads(path.read_text(encoding="utf-8"))
    pol, h, costs = summary["policies"], summary["headline"], summary["costs"]
    st.markdown(
        "**Is predictive maintenance actually better?** Engines with known failure times are replayed cycle by "
        "cycle under four maintenance policies. Servicing too early wastes useful life (more services over time), "
        "servicing too late means an unplanned failure.")
    st.caption(f"Evaluated on: {summary['evaluation_source']}. Costs (relative): planned service = {costs['service']:g}, "
               f"unplanned failure = {costs['failure']:g}, planning delay = {costs['planning_delay']} cycles. "
               f"The fixed interval ({h['fixed_interval']} cycles) is the best possible one for the training engines.")

    c = st.columns(4)
    c[0].metric("Saving vs best fixed interval", f"{h['saving_vs_fixed_pct']:.0f}%")
    c[1].metric("Saving vs run-to-failure", f"{h['saving_vs_reactive_pct']:.0f}%")
    c[2].metric("Failures prevented", f"{pol['fleetguard']['failures_prevented_pct']:.0f}%")
    c[2].caption(f"{pol['fleetguard']['failures']} of {pol['fleetguard']['engines']} engines failed")
    c[3].metric("Useful life used", f"{pol['fleetguard']['useful_life_used_pct']:.0f}%")
    c[3].caption(f"vs {pol['fixed']['useful_life_used_pct']:.0f}% with the fixed interval")

    left, right = st.columns(2)
    with left:
        st.plotly_chart(cost_figure(ctx, pol), width="stretch")
    with right:
        st.plotly_chart(sweep_figure(ctx, pd.read_csv(sim_dir / "cost_ratio_sweep.csv")), width="stretch")
        st.caption("The cost ratio is an assumption, so the result is checked across a wide range.")

    table = pd.DataFrame([{**{"policy": POLICY_LABELS[p]}, **pol[p]} for p in POLICY_ORDER])
    table = table.rename(columns={"failures_prevented_pct": "failures prevented %", "mean_wasted_cycles": "avg wasted cycles",
                                  "useful_life_used_pct": "useful life used %", "cost_per_1000_cycles": "cost / 1k cycles"})
    st.dataframe(table.round(1), hide_index=True, width="stretch")

    eng = pd.read_csv(sim_dir / "policy_engines.csv")
    st.plotly_chart(engines_figure(ctx, eng, h["fixed_interval"], costs["planning_delay"]), width="stretch")
    st.caption("A dot left of its ✕ is a prevented failure; the gap is useful life given up. The fixed interval "
               "services every engine at the same cycle, wasting most life on long-lived engines.")


POLICY_ORDER = ["reactive", "fixed", "fleetguard", "oracle"]


def _policy_colors(ctx) -> dict:
    return {"reactive": ctx.muted, "fixed": ctx.series[1], "fleetguard": ctx.series[0], "oracle": ctx.series[2]}


def cost_figure(ctx, pol: dict) -> go.Figure:
    colors = _policy_colors(ctx)
    vals = [pol[p]["cost_per_1000_cycles"] for p in POLICY_ORDER]
    short = {"reactive": "Run to failure", "fixed": "Fixed interval", "fleetguard": "FleetGuard", "oracle": "Oracle"}
    fig = go.Figure(go.Bar(x=[short[p] for p in POLICY_ORDER], y=vals,
                           marker_color=[colors[p] for p in POLICY_ORDER],
                           text=[f"{v:.1f}" for v in vals], textposition="outside", marker_cornerradius=4))
    fig.update_layout(title="Maintenance cost per 1,000 operating cycles (lower is better)",
                      yaxis=dict(title="relative cost", range=[0, max(vals) * 1.15]), showlegend=False)
    fig = ctx.style(fig, 400)
    fig.update_layout(margin=dict(l=60, b=50))
    return fig


def sweep_figure(ctx, sweep: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep["failure_cost_ratio"], y=sweep["fleetguard_saving_vs_fixed_pct"], mode="lines+markers",
                             name="vs best fixed interval", line=dict(color=ctx.series[1], width=2), marker=dict(size=9)))
    fig.add_trace(go.Scatter(x=sweep["failure_cost_ratio"], y=sweep["fleetguard_saving_vs_reactive_pct"], mode="lines+markers",
                             name="vs run-to-failure", line=dict(color=ctx.muted, width=2), marker=dict(size=9)))
    fig.add_hline(y=0, line_color=ctx.muted, line_width=1)
    fig.update_layout(title="FleetGuard saving for different failure:service cost ratios",
                      xaxis=dict(title="cost of a failure ÷ cost of a planned service", type="log",
                                 tickvals=sweep["failure_cost_ratio"]),
                      yaxis=dict(title="saving (%)", ticksuffix="%", range=[0, 100]),
                      legend=dict(orientation="h", y=-0.3), hovermode="x unified")
    fig = ctx.style(fig, 380)
    fig.update_layout(margin=dict(l=60, b=20))
    return fig


def engines_figure(ctx, eng: pd.DataFrame, fixed_interval: int, delay: int) -> go.Figure:
    eng = eng.sort_values("lifetime").reset_index(drop=True)
    service = eng["trigger_cycle"] + delay
    y = [f"engine {e}" for e in eng["engine_id"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eng["lifetime"], y=y, mode="markers", name="actual failure",
                             marker=dict(symbol="x", size=9, color=ctx.colors["CRITICAL"])))
    fig.add_trace(go.Scatter(x=service, y=y, mode="markers", name="FleetGuard service",
                             marker=dict(size=9, color=ctx.series[0], line=dict(width=1, color="white"))))
    fig.add_vline(x=fixed_interval, line_dash="dash", line_color=ctx.series[1],
                  annotation_text=f"fixed interval ({fixed_interval})")
    fig.update_layout(title="When each engine is serviced vs when it would have failed",
                      xaxis_title="cycle", yaxis=dict(showticklabels=len(eng) <= 40, autorange="reversed"),
                      hovermode="y unified", legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom"))
    fig = ctx.style(fig, max(380, 14 * len(eng)))
    fig.update_layout(margin=dict(l=80, t=70, b=40))
    return fig
