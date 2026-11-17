"""
VehicleGPT Tiered Storage
=========================
SQLite-backed multi-tier retention store:
  Tier 1: raw_samples (5-second downsampled points, rolling 15 days)
  Tier 2: daily_summaries (1 row per day with comprehensive signal stats & metrics, 12+ months)
  Tier 3: events (warnings, DTCs, trips with +/- 60s signal context, kept forever)
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;

CREATE TABLE IF NOT EXISTS raw_samples (
    timestamp REAL NOT NULL,
    signal_name TEXT NOT NULL,
    value REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_raw_sig_time ON raw_samples(signal_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_raw_time ON raw_samples(timestamp);

CREATE TABLE IF NOT EXISTS daily_summaries (
    date TEXT PRIMARY KEY,
    distance_km REAL DEFAULT 0.0,
    energy_consumption_kwh REAL DEFAULT 0.0,
    fuel_consumption_l REAL DEFAULT 0.0,
    city_efficiency REAL DEFAULT 0.0,
    highway_efficiency REAL DEFAULT 0.0,
    city_share REAL DEFAULT 0.0,
    highway_share REAL DEFAULT 0.0,
    avg_outside_temp_c REAL DEFAULT 20.0,
    max_battery_temp_c REAL DEFAULT 35.0,
    hvac_energy_kwh REAL DEFAULT 0.0,
    adas_disengagements INTEGER DEFAULT 0,
    warning_count INTEGER DEFAULT 0,
    trip_count INTEGER DEFAULT 0,
    signal_stats TEXT NOT NULL,
    digest TEXT,
    created_at REAL NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS daily_summaries_fts USING fts5(
    date UNINDEXED,
    digest
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    event_type TEXT NOT NULL,
    title TEXT NOT NULL,
    code TEXT,
    detail TEXT,
    severity TEXT DEFAULT 'info',
    context_snapshot TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);

CREATE TABLE IF NOT EXISTS compaction_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp REAL NOT NULL,
    date_compacted TEXT NOT NULL,
    rows_compacted INTEGER NOT NULL,
    duration_ms REAL NOT NULL,
    storage_bytes_before INTEGER NOT NULL,
    storage_bytes_after INTEGER NOT NULL,
    status TEXT NOT NULL
);
"""


class TieredStore:
    def __init__(self, db_path: str = "vehiclegpt_retention.db") -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._lock:
            conn = self._get_connection()
            try:
                conn.executescript(SCHEMA_SQL)
                conn.commit()
            finally:
                conn.close()

    def insert_raw_samples(self, samples: List[Tuple[float, str, float]]) -> None:
        if not samples:
            return
        with self._lock:
            conn = self._get_connection()
            try:
                conn.executemany(
                    "INSERT INTO raw_samples (timestamp, signal_name, value) VALUES (?, ?, ?)",
                    samples
                )
                conn.commit()
            finally:
                conn.close()

    def insert_event(
        self,
        event_type: str,
        title: str,
        code: Optional[str] = None,
        detail: Optional[str] = None,
        severity: str = "info",
        context_snapshot: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None
    ) -> int:
        ts = timestamp if timestamp is not None else time.time()
        context_json = json.dumps(context_snapshot) if context_snapshot else None
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    INSERT INTO events (timestamp, event_type, title, code, detail, severity, context_snapshot)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (ts, event_type, title, code, detail, severity, context_json)
                )
                conn.commit()
                return cur.lastrowid or 0
            finally:
                conn.close()

    def get_events(
        self,
        start_time: float = 0.0,
        end_time: Optional[float] = None,
        event_type: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        end_ts = end_time if end_time is not None else time.time() + 86400
        query = "SELECT * FROM events WHERE timestamp >= ? AND timestamp <= ?"
        params: List[Any] = [start_time, end_ts]
        if event_type:
            query += " AND event_type = ?"
            params.append(event_type)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute(query, params)
                rows = cur.fetchall()
                results = []
                for r in rows:
                    snap = json.loads(r["context_snapshot"]) if r["context_snapshot"] else None
                    results.append({
                        "id": r["id"],
                        "timestamp": r["timestamp"],
                        "event_type": r["event_type"],
                        "title": r["title"],
                        "code": r["code"],
                        "detail": r["detail"],
                        "severity": r["severity"],
                        "context_snapshot": snap,
                    })
                return results
            finally:
                conn.close()

    def get_raw_samples(
        self,
        signal_name: str,
        start_time: float,
        end_time: float,
        limit: int = 2000
    ) -> List[Tuple[float, float]]:
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT timestamp, value FROM raw_samples
                    WHERE signal_name = ? AND timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp ASC LIMIT ?
                    """,
                    (signal_name, start_time, end_time, limit)
                )
                return [(r[0], r[1]) for r in cur.fetchall()]
            finally:
                conn.close()

    def get_signal_context(self, center_ts: float, window_s: float = 60.0) -> Dict[str, Any]:
        """+/- window of every signal around an event, stored with the freeze-frame."""
        start_time = center_ts - window_s
        end_time = center_ts + window_s
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT signal_name, timestamp, value FROM raw_samples
                    WHERE timestamp >= ? AND timestamp <= ?
                    ORDER BY timestamp ASC
                    """,
                    (start_time, end_time)
                )
                by_sig: Dict[str, List[Dict[str, float]]] = {}
                for r in cur.fetchall():
                    by_sig.setdefault(r[0], []).append({"t": float(r[1]), "v": float(r[2])})
                return {
                    "signal_context_window": "+/- 60 seconds",
                    "center_timestamp": center_ts,
                    "window_s": window_s,
                    "signals": by_sig,
                }
            finally:
                conn.close()

    def get_event_by_id(self, event_id: int) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("SELECT * FROM events WHERE id = ?", (event_id,))
                r = cur.fetchone()
                if r is None:
                    return None
                snap = json.loads(r["context_snapshot"]) if r["context_snapshot"] else None
                return {
                    "id": r["id"],
                    "timestamp": r["timestamp"],
                    "event_type": r["event_type"],
                    "title": r["title"],
                    "code": r["code"],
                    "detail": r["detail"],
                    "severity": r["severity"],
                    "context_snapshot": snap,
                }
            finally:
                conn.close()

    def get_daily_summaries(
        self,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    SELECT * FROM daily_summaries
                    WHERE date >= ? AND date <= ?
                    ORDER BY date ASC
                    """,
                    (start_date, end_date)
                )
                rows = cur.fetchall()
                results = []
                for r in rows:
                    stats = json.loads(r["signal_stats"]) if r["signal_stats"] else {}
                    results.append({
                        "date": r["date"],
                        "distance_km": r["distance_km"],
                        "energy_consumption_kwh": r["energy_consumption_kwh"],
                        "fuel_consumption_l": r["fuel_consumption_l"],
                        "city_efficiency": r["city_efficiency"],
                        "highway_efficiency": r["highway_efficiency"],
                        "city_share": r["city_share"],
                        "highway_share": r["highway_share"],
                        "avg_outside_temp_c": r["avg_outside_temp_c"],
                        "max_battery_temp_c": r["max_battery_temp_c"],
                        "hvac_energy_kwh": r["hvac_energy_kwh"],
                        "adas_disengagements": r["adas_disengagements"],
                        "warning_count": r["warning_count"],
                        "trip_count": r["trip_count"],
                        "signal_stats": stats,
                        "digest": r["digest"],
                        "created_at": r["created_at"],
                    })
                return results
            finally:
                conn.close()

    def search_digests(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search daily digest texts via FTS5 or fallback LIKE search."""
        clean_q = "".join(c for c in query if c.isalnum() or c.isspace()).strip()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                if clean_q:
                    try:
                        cur.execute(
                            "SELECT date, digest FROM daily_summaries_fts WHERE daily_summaries_fts MATCH ? LIMIT ?",
                            (clean_q, limit)
                        )
                        rows = cur.fetchall()
                        if rows:
                            return [{"date": r[0], "digest": r[1]} for r in rows]
                    except Exception:
                        pass
                # Fallback to LIKE
                like_term = f"%{query}%"
                cur.execute(
                    "SELECT date, digest FROM daily_summaries WHERE digest LIKE ? ORDER BY date DESC LIMIT ?",
                    (like_term, limit)
                )
                return [{"date": r[0], "digest": r[1]} for r in cur.fetchall()]
            finally:
                conn.close()

    def prune_expired_summaries(self, retention_months: int = 12) -> int:
        """Prunes daily summaries older than retention_months (12+ months)."""
        cutoff_date = (datetime.date.today() - datetime.timedelta(days=int(retention_months * 30.5))).isoformat()
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM daily_summaries WHERE date < ?", (cutoff_date,))
                to_prune = cur.fetchone()[0]
                if to_prune > 0:
                    cur.execute("DELETE FROM daily_summaries WHERE date < ?", (cutoff_date,))
                    try:
                        cur.execute("DELETE FROM daily_summaries_fts WHERE date < ?", (cutoff_date,))
                    except Exception:
                        pass
                    conn.commit()
                    logger.info("Pruned %d daily summaries older than %s", to_prune, cutoff_date)
                return to_prune
            finally:
                conn.close()

    def get_storage_stats(self) -> Dict[str, Any]:
        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM raw_samples")
                raw_count = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM daily_summaries")
                summary_count = cur.fetchone()[0]

                cur.execute("SELECT COUNT(*) FROM events")
                event_count = cur.fetchone()[0]

                cur.execute("SELECT * FROM compaction_log ORDER BY id DESC LIMIT 1")
                last_compaction = cur.fetchone()

                cur.execute("SELECT MIN(timestamp), MAX(timestamp) FROM raw_samples")
                raw_time_range = cur.fetchone()

                cur.execute("SELECT MIN(date), MAX(date) FROM daily_summaries")
                summary_date_range = cur.fetchone()

                file_size_bytes = 0
                if os.path.exists(self.db_path):
                    file_size_bytes = os.path.getsize(self.db_path)

                # Storage per tier estimation
                tier_1_raw_kb = round((raw_count * 36) / 1024.0, 1)
                tier_2_daily_kb = round((summary_count * 2048) / 1024.0, 1)
                tier_3_events_kb = round((event_count * 1024) / 1024.0, 1)

                # Compute total searchable days
                all_dates = set()
                cur.execute("SELECT date FROM daily_summaries")
                for r in cur.fetchall():
                    if r[0]:
                        all_dates.add(r[0])
                if raw_time_range and raw_time_range[0] and raw_time_range[1]:
                    d_cur = datetime.date.fromtimestamp(raw_time_range[0])
                    d_end = datetime.date.fromtimestamp(raw_time_range[1])
                    while d_cur <= d_end:
                        all_dates.add(d_cur.isoformat())
                        d_cur += datetime.timedelta(days=1)
                days_searchable = max(1, len(all_dates))

                return {
                    "raw_samples_count": raw_count,
                    "daily_summaries_count": summary_count,
                    "events_count": event_count,
                    "database_size_bytes": file_size_bytes,
                    "database_size_kb": round(file_size_bytes / 1024.0, 1),
                    "database_size_mb": round(file_size_bytes / (1024.0 * 1024.0), 2),
                    "storage_per_tier": {
                        "tier_1_raw_samples_kb": tier_1_raw_kb,
                        "tier_2_daily_summaries_kb": tier_2_daily_kb,
                        "tier_3_events_kb": tier_3_events_kb,
                        "raw_samples_kb": tier_1_raw_kb,
                        "daily_summaries_kb": tier_2_daily_kb,
                        "events_kb": tier_3_events_kb,
                    },
                    "days_searchable": days_searchable,
                    "raw_oldest_ts": raw_time_range[0] if raw_time_range else None,
                    "raw_newest_ts": raw_time_range[1] if raw_time_range else None,
                    "summary_earliest_date": summary_date_range[0] if summary_date_range else None,
                    "summary_latest_date": summary_date_range[1] if summary_date_range else None,
                    "last_compaction": dict(last_compaction) if last_compaction else None,
                }
            finally:
                conn.close()
