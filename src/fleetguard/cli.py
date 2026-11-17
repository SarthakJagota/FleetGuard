"""Command-line entry point.

    fleetguard run-all                      full demo on all four subsets
    fleetguard run-all --subsets FD001      ... or just one
    fleetguard train   --subset FD002       train + register (+ promote if better)
    fleetguard retrain --subset FD002       retrain when data/incoming has new batches
    fleetguard score   --subset FD002       score fleet, raise work orders
    fleetguard stream  --subset FD002       replay telemetry cycle by cycle
    fleetguard work-orders --status OPEN    list work orders
    fleetguard close WO-00012 --note "..."  close / reject a work order
    fleetguard serve --port 8000            REST API, docs at http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import SUBSETS, load_config
from .pipeline import FleetGuard


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="fleetguard", description="Closed-loop predictive maintenance on NASA C-MAPSS")
    p.add_argument("--config", default=None, help="path to config.yaml")
    p.add_argument("--no-mlflow", action="store_true", help="disable MLflow tracking")
    p.add_argument("--fast", action="store_true", help="tiny models (smoke test)")
    p.add_argument("--models", nargs="+", help="override candidate models")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    ra = sub.add_parser("run-all", help="train v1 -> new data -> retrain v2 -> score -> stream")
    ra.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    ra.add_argument("--keep", action="store_true", help="do not reset previous models/outputs")
    ra.add_argument("--step", type=int, default=10, help="stream replay tick size in cycles")

    tr = sub.add_parser("train")
    tr.add_argument("--subset", required=True)
    tr.add_argument("--initial-fraction", type=float, default=1.0,
                    help="train on this fraction of engines; the rest can later 'arrive' as new data")

    sa = sub.add_parser("simulate-arrival", help="release held-back engines into data/incoming/")
    sa.add_argument("--subset", required=True)

    rt = sub.add_parser("retrain")
    rt.add_argument("--subset", required=True)
    rt.add_argument("--force", action="store_true")

    sc = sub.add_parser("score")
    sc.add_argument("--subset", required=True)

    st = sub.add_parser("stream")
    st.add_argument("--subset", required=True)
    st.add_argument("--step", type=int, default=10)

    sv = sub.add_parser("serve", help="start the REST API (docs at /docs)")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)

    sp = sub.add_parser("simulate-policies", help="reactive vs fixed-interval vs FleetGuard maintenance cost")
    sp.add_argument("--subset", required=True)

    wl = sub.add_parser("work-orders", help="list work orders")
    wl.add_argument("--subset")
    wl.add_argument("--status")

    cl = sub.add_parser("close", help="close or reject a work order")
    cl.add_argument("wo_id")
    cl.add_argument("--status", default="CLOSED", choices=["CLOSED", "REJECTED", "ACKNOWLEDGED"])
    cl.add_argument("--note", default="")

    args = p.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    if args.cmd == "serve":
        import os

        import uvicorn

        if args.config:
            os.environ["FLEETGUARD_CONFIG"] = args.config
        uvicorn.run("fleetguard.api:app", host=args.host, port=args.port)
        return 0

    cfg = load_config(args.config)
    fg = FleetGuard(cfg, track=not args.no_mlflow, fast=args.fast, candidates=args.models)

    if args.cmd == "run-all":
        subsets = [s.upper() for s in args.subsets]
        summary = fg.run_all(subsets, fresh=not args.keep, stream_step=args.step)
        _print_summary(summary)
    elif args.cmd == "train":
        print(json.dumps(_brief(fg.train(args.subset, args.initial_fraction)), indent=2))
    elif args.cmd == "simulate-arrival":
        print(fg.simulate_arrival(args.subset))
    elif args.cmd == "retrain":
        print(json.dumps(_brief(fg.retrain(args.subset, force=args.force)), indent=2))
    elif args.cmd == "score":
        fleet = fg.score(args.subset)
        cols = ["engine_id", "severity", "rul_pred", "rul_lower", "rul_upper", "anomaly_index", "reason_codes"]
        print(fleet[cols].head(20).to_string(index=False))
    elif args.cmd == "stream":
        ev = fg.stream(args.subset, step=args.step)
        print(ev.groupby(["event", "severity"]).size().to_string())
    elif args.cmd == "simulate-policies":
        res = fg.simulate_policies(args.subset)
        print(f"Evaluated on: {res['evaluation_source']}")
        print(f"{'policy':<12}{'failures':>9}{'prevented':>11}{'wasted cyc':>11}{'life used':>10}{'cost/1k cyc':>12}")
        for name, m in res["policies"].items():
            print(f"{name:<12}{m['failures']:>9}{m['failures_prevented_pct']:>10.0f}%{m['mean_wasted_cycles']:>11.1f}"
                  f"{m['useful_life_used_pct']:>9.0f}%{m['cost_per_1000_cycles']:>12.2f}")
        h = res["headline"]
        print(f"FleetGuard saves {h['saving_vs_fixed_pct']:.0f}% vs the best fixed interval ({h['fixed_interval']} cycles) "
              f"and {h['saving_vs_reactive_pct']:.0f}% vs reactive; {h['gap_to_oracle_pct']:.0f}% above the oracle.")
    elif args.cmd in ("work-orders", "close"):
        store = fg.work_orders()
        if args.cmd == "close":
            wo = store.set_status(args.wo_id, args.status, args.note)
            store.save()
            print(f"{wo['wo_id']} -> {wo['status']}")
        else:
            for wo in store.orders:
                if (not args.subset or wo["subset"] == args.subset.upper()) and (not args.status or wo["status"] == args.status.upper()):
                    print(f"{wo['wo_id']}  {wo['subset']}  engine {wo['engine_id']:>3}  {wo['severity']:<8} {wo['status']:<12} {wo['title']}")
    return 0


def _brief(result: dict) -> dict:
    m = result.get("metrics", {})
    keys = ("test_rmse", "test_mae", "test_nasa_score", "test_interval_coverage", "agent_recall_near_failure")
    return {**{k: v for k, v in result.items() if k != "metrics"}, **{k: round(m[k], 3) for k in keys if k in m}}


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 92)
    print(f"{'subset':<7}{'champion':<10}{'RMSE':>7}{'MAE':>7}{'NASA':>9}{'cover80':>9}{'recall':>8}{'falseAl':>9}  v1->v2")
    print("-" * 92)
    for s, r in summary.items():
        m = r["metrics"]
        print(f"{s:<7}{r['champion']:<10}{m['test_rmse']:>7.2f}{m['test_mae']:>7.2f}{m['test_nasa_score']:>9.0f}"
              f"{m['test_interval_coverage']:>9.2f}{m['agent_recall_near_failure']:>8.2f}{m['agent_false_alarm_rate']:>9.2f}"
              f"  {r['v1']['test_rmse']:.2f} -> {r['v2']['test_rmse']:.2f} ({r['v2']['decision'].split(':')[0]})")
    print("=" * 92)


if __name__ == "__main__":
    sys.exit(main())
