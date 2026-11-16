"""SQLite persistence for work orders and scoring history.

Tables
    work_orders        one row per order; key fields as columns (for querying and
                       indexing), the full order as JSON in `payload`
    work_order_events  the order's history (created, escalated, updated, status changes)
    scoring_runs       one row per fleet scoring run (which model, when, how many engines)
    predictions        every engine's result in every scoring run, so trends over
                       time can be queried (e.g. how an engine's health evolved)

SQLite is a single file with no server and ships with Python. WAL mode lets the
dashboard read while the pipeline writes, which is all a single-site deployment
needs. Moving to PostgreSQL later would only touch this module.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS work_orders (
    wo_id         TEXT PRIMARY KEY,
    fleet         TEXT    NOT NULL,
    engine_id     INTEGER NOT NULL,
    status        TEXT    NOT NULL,
    severity      TEXT    NOT NULL,
    priority      TEXT,
    type          TEXT,
    title         TEXT,
    rul_pred      REAL,
    failure_prob  REAL,
    health_score  REAL,
    model_version TEXT,
    created_at    TEXT,
    updated_at    TEXT,
    payload       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_wo_engine ON work_orders(fleet, engine_id, status);

CREATE TABLE IF NOT EXISTS work_order_events (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    wo_id    TEXT    NOT NULL REFERENCES work_orders(wo_id) ON DELETE CASCADE,
    seq      INTEGER NOT NULL,
    at       TEXT,
    cycle    INTEGER,
    event    TEXT    NOT NULL,
    severity TEXT,
    actor    TEXT,
    note     TEXT
);
CREATE INDEX IF NOT EXISTS ix_events_wo ON work_order_events(wo_id, seq);

CREATE TABLE IF NOT EXISTS scoring_runs (
    run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fleet           TEXT NOT NULL,
    model_version   TEXT,
    scored_at       TEXT NOT NULL,
    engines         INTEGER,
    severity_counts TEXT
);

CREATE TABLE IF NOT EXISTS predictions (
    run_id        INTEGER NOT NULL REFERENCES scoring_runs(run_id) ON DELETE CASCADE,
    fleet         TEXT    NOT NULL,
    engine_id     INTEGER NOT NULL,
    cycle         INTEGER,
    severity      TEXT,
    rul_pred      REAL,
    rul_lower     REAL,
    rul_upper     REAL,
    failure_prob  REAL,
    health_score  REAL,
    anomaly_index REAL,
    true_rul      INTEGER,
    wo_id         TEXT,
    PRIMARY KEY (run_id, engine_id)
);
CREATE INDEX IF NOT EXISTS ix_pred_engine ON predictions(fleet, engine_id);
"""

_ORDER_COLS = ["wo_id", "fleet", "engine_id", "status", "severity", "priority", "type", "title", "rul_pred",
               "failure_prob", "health_score", "model_version", "created_at", "updated_at", "payload"]
_PRED_COLS = ["engine_id", "cycle", "severity", "rul_pred", "rul_lower", "rul_upper", "failure_prob",
              "health_score", "anomaly_index", "true_rul", "wo_id"]


def _num(v):
    """SQLite-friendly scalar (None for NaN, plain Python numbers)."""
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(v, "item"):
        return v.item()
    return v


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA journal_mode = WAL")
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    # ------------------------------------------------------------ work orders
    def load_orders(self) -> list[dict]:
        with self.connect() as con:
            rows = con.execute("SELECT wo_id, payload FROM work_orders ORDER BY wo_id").fetchall()
            events = con.execute(
                "SELECT wo_id, at, cycle, event, severity, actor, note FROM work_order_events ORDER BY wo_id, seq"
            ).fetchall()
        history: dict[str, list] = {}
        for e in events:
            item = {k: e[k] for k in ("at", "cycle", "event", "severity") if e[k] is not None}
            if e["actor"] is not None:
                item["by"] = e["actor"]
            if e["note"] is not None:
                item["note"] = e["note"]
            history.setdefault(e["wo_id"], []).append(item)
        orders = []
        for r in rows:
            wo = json.loads(r["payload"])
            wo["history"] = history.get(r["wo_id"], [])
            orders.append(wo)
        return orders

    def save_orders(self, orders: list[dict]) -> None:
        """Upsert orders and replace their event history, in one transaction."""
        if not orders:
            return
        with self.connect() as con:
            for wo in orders:
                payload = {k: v for k, v in wo.items() if k != "history"}
                row = {
                    **{c: _num(wo.get(c)) for c in _ORDER_COLS if c not in ("fleet", "payload")},
                    "fleet": wo["subset"],
                    "payload": json.dumps(payload, default=str),
                }
                con.execute(
                    f"INSERT INTO work_orders ({', '.join(_ORDER_COLS)}) VALUES ({', '.join('?' * len(_ORDER_COLS))}) "
                    f"ON CONFLICT(wo_id) DO UPDATE SET "
                    + ", ".join(f"{c} = excluded.{c}" for c in _ORDER_COLS if c != "wo_id"),
                    [row[c] for c in _ORDER_COLS],
                )
                con.execute("DELETE FROM work_order_events WHERE wo_id = ?", (wo["wo_id"],))
                con.executemany(
                    "INSERT INTO work_order_events (wo_id, seq, at, cycle, event, severity, actor, note) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [(wo["wo_id"], i, e.get("at"), _num(e.get("cycle")), e["event"], e.get("severity"),
                      e.get("by"), e.get("note")) for i, e in enumerate(wo.get("history", []))],
                )

    def delete_fleets(self, fleets) -> None:
        fleets = list(fleets)
        marks = ", ".join("?" * len(fleets))
        with self.connect() as con:
            con.execute(f"DELETE FROM work_orders WHERE fleet IN ({marks})", fleets)
            con.execute(f"DELETE FROM scoring_runs WHERE fleet IN ({marks})", fleets)

    # ------------------------------------------------------ scoring history
    def record_scoring(self, fleet: str, model_version: str, fleet_table: pd.DataFrame) -> int:
        counts = fleet_table["severity"].value_counts().to_dict()
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self.connect() as con:
            cur = con.execute(
                "INSERT INTO scoring_runs (fleet, model_version, scored_at, engines, severity_counts) VALUES (?, ?, ?, ?, ?)",
                (fleet, model_version, now, len(fleet_table), json.dumps(counts)),
            )
            run_id = cur.lastrowid
            con.executemany(
                f"INSERT INTO predictions (run_id, fleet, {', '.join(_PRED_COLS)}) "
                f"VALUES (?, ?, {', '.join('?' * len(_PRED_COLS))})",
                [(run_id, fleet, *[_num(r.get(c)) for c in _PRED_COLS]) for r in fleet_table.to_dict("records")],
            )
        return run_id

    def query(self, sql: str, params=()) -> pd.DataFrame:
        with self.connect() as con:
            return pd.read_sql_query(sql, con, params=params)

    def scoring_runs(self, fleet: str | None = None) -> pd.DataFrame:
        where, params = ("WHERE fleet = ?", (fleet,)) if fleet else ("", ())
        return self.query(f"SELECT * FROM scoring_runs {where} ORDER BY run_id DESC", params)

    def engine_history(self, fleet: str, engine_id: int) -> pd.DataFrame:
        return self.query(
            "SELECT r.run_id, r.scored_at, r.model_version, p.* FROM predictions p "
            "JOIN scoring_runs r USING (run_id) WHERE p.fleet = ? AND p.engine_id = ? ORDER BY r.run_id",
            (fleet, int(engine_id)),
        ).loc[:, lambda d: ~d.columns.duplicated()]

    def counts(self) -> dict:
        with self.connect() as con:
            return {t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                    for t in ("work_orders", "work_order_events", "scoring_runs", "predictions")}
