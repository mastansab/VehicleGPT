"""
Acceptance tests for VehicleGPT tiered data retention.

Covers the full acceptance checklist:
 1. Fast-forward 30 days → range/efficiency queries answered from daily summaries.
 2. Battery temp 3 weeks ago → answered from daily summaries with attribution.
 3. Tyre-leak check still works on data > 15 days old.
 4. Single-signal lookups on old data return daily-summary tier.
 5. Day-20 battery overheating event preserved with +/- 60s freeze-frame.
 6. Compaction is idempotent (second run finds nothing to compact).
 7. Crash resilience (transaction not committed → raw data intact).
 8. Bounded storage after 90 simulated days.
 9. Modules import cleanly with no errors.
"""
from __future__ import annotations

import datetime
import json
import os
import sqlite3
import tempfile
import time

import pytest

from peykan.server.storage.tiered_store import TieredStore
from peykan.server.storage.compaction import Compactor, RetentionScheduler
from peykan.server.storage.query_layer import VehicleDataAccess


# ──────────────────────── fixtures ────────────────────────


@pytest.fixture()
def tmp_db(tmp_path):
    """Create a fresh TieredStore backed by a temp DB."""
    db_file = str(tmp_path / "test_retention.db")
    return TieredStore(db_path=db_file)


@pytest.fixture()
def data_access(tmp_db):
    return VehicleDataAccess(tmp_db, raw_retention_days=15)


@pytest.fixture()
def populated(data_access):
    """Fast-forward 30 days into the data access fixture."""
    result = data_access.fast_forward_30_days()
    return data_access, result


# ──────────────────── 1. Range / efficiency query ─────────────────────


def test_fast_forward_range_query(populated):
    da, ff_result = populated
    assert ff_result["status"] == "completed"
    assert ff_result["days_simulated"] == 30
    assert ff_result["days_compacted"] > 0

    answer = da.query_historical_question("why has my range reduced over the last two weeks?")
    assert answer is not None
    assert "daily summaries" in answer["attribution"]
    assert answer["tier"] == "daily_summary"
    assert len(answer.get("answer", "")) > 0


# ────────────────── 2. Battery temp 3 weeks ago ──────────────────────


def test_battery_temp_three_weeks_ago(populated):
    da, _ = populated
    answer = da.query_historical_question("what was my battery temperature three weeks ago?")
    assert answer is not None
    assert "daily summaries" in answer["attribution"]
    assert "daily_summary" in answer["tier"]
    assert len(answer["answer"]) > 0


# ──────────── 3. Tyre-leak check works on old data ───────────────────


def test_tyre_leak_on_old_data(populated):
    da, _ = populated
    result = da.get_tyre_leak_analysis(days=30)
    assert result is not None
    assert "daily summaries" in result["tier_used"]
    assert result.get("attribution") is not None
    assert result.get("days_analyzed") == 30


# ────────── 4. Single-signal lookup returns daily tier ───────────────


def test_signal_history_old_data(populated):
    da, _ = populated
    start_time = time.time() - (30 * 86400)
    history = da.get_signal_history("ENGINE_TEMP", start_time)
    assert "daily" in history["tier"] or history["tier"] == "hybrid (raw + daily summaries)"
    assert history["detail_available"] is False
    assert "daily summaries" in history["attribution"]
    assert len(history["points"]) > 0

    pt = next(p for p in history["points"] if "mean" in p or p.get("source") == "daily_summary")
    assert "min" in pt
    assert "max" in pt
    assert "mean" in pt

    older_start = time.time() - (25 * 86400)
    older_end = time.time() - (20 * 86400)
    archived = da.get_signal_history("ENGINE_TEMP", older_start, older_end)
    assert archived["tier"] == "daily_summary"


# ────── 5. Day-20 overheating event with freeze-frame preserved ──────


def test_day20_overheating_event(populated):
    da, _ = populated
    events = da.get_fault_diagnosis(days=30)
    assert len(events) > 0

    overheat = [e for e in events if e.get("code") == "P0217"]
    assert len(overheat) >= 1, "Day-20 P0217 overheat event must be preserved"

    ev = overheat[0]
    assert ev["event_type"] == "warning"
    assert ev["severity"] == "critical"

    # Verify freeze-frame context snapshot
    snap = ev.get("context_snapshot")
    assert snap is not None, "Event must have +/- 60s context snapshot"
    assert snap.get("fault_code") == "P0217"
    assert snap.get("signal_context_window") == "+/- 60 seconds"
    assert len(snap.get("samples", [])) >= 5


# ────────────── 6. Compaction idempotency ────────────────────────────


def test_compaction_idempotent(populated):
    da, _ = populated
    compactor = da.compactor

    # Second compaction should find no dates to compact
    results = compactor.run_compaction(raw_retention_days=15)
    for r in results:
        assert r.get("status") in ("skipped_no_rows", "success")
    # In fact, after the first fast-forward already compacted, there should
    # be no uncompacted dates at all.
    uncompacted = compactor.get_uncompacted_dates(raw_retention_days=15)
    assert len(uncompacted) == 0, (
        f"Expected zero uncompacted dates after re-run, got: {uncompacted}"
    )


# ──────────────── 7. Crash resilience ────────────────────────────────


def test_crash_resilience(tmp_db):
    """If a transaction isn't committed, raw data survives."""
    store = tmp_db
    # Insert some raw data for a date that's 20 days ago
    target_date = datetime.date.today() - datetime.timedelta(days=20)
    midday = datetime.datetime.combine(target_date, datetime.time(hour=12))
    ts = midday.timestamp()

    samples = [(ts + i * 5, "ENGINE_TEMP", 85.0 + i) for i in range(10)]
    store.insert_raw_samples(samples)

    # Manually open a connection, begin a transaction, write the summary,
    # but do NOT commit (simulating a crash)
    conn = store._get_connection()
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        """
        INSERT OR REPLACE INTO daily_summaries
            (date, signal_stats, created_at)
        VALUES (?, ?, ?)
        """,
        (target_date.isoformat(), json.dumps({}), time.time())
    )
    # Simulate crash: close without commit → implicit rollback
    conn.close()

    # Raw data must still be intact
    raw = store.get_raw_samples("ENGINE_TEMP", ts - 1, ts + 100)
    assert len(raw) == 10, (
        f"Expected 10 raw samples after aborted transaction, got {len(raw)}"
    )

    # And the daily summary should NOT have been written
    summaries = store.get_daily_summaries(
        target_date.isoformat(), target_date.isoformat()
    )
    assert len(summaries) == 0, (
        "Uncommitted daily summary should not persist after crash"
    )


# ────────────── 8. Bounded storage over 90 days ─────────────────────


def test_bounded_storage_90_days(tmp_db):
    """After 3 × fast-forward (90 simulated days), storage stays bounded."""
    store = tmp_db
    da = VehicleDataAccess(store, raw_retention_days=15)

    sizes = []
    for _ in range(3):
        da.fast_forward_30_days()
        stats = store.get_storage_stats()
        sizes.append(stats["database_size_bytes"])

    # The DB should not grow linearly with each 30-day batch —
    # compaction keeps the raw tier bounded.
    assert sizes[-1] < sizes[-2] * 3, (
        f"Storage should be bounded; got sizes: {sizes}"
    )

    # Raw sample count should be bounded to ~15 days worth of data at most
    final_stats = store.get_storage_stats()
    # Each day generates 24 samples × 6 signals = 144 per day
    # For 15 days that's ~2160. Allow generous headroom.
    assert final_stats["raw_samples_count"] < 10000, (
        f"Raw samples should be bounded; got {final_stats['raw_samples_count']}"
    )


# ──────────────── 9. Clean imports ───────────────────────────────────


def test_imports():
    """All storage modules import cleanly."""
    from peykan.server.storage import (
        TieredStore,
        Compactor,
        RetentionScheduler,
        VehicleDataAccess,
    )
    assert TieredStore is not None
    assert Compactor is not None
    assert RetentionScheduler is not None
    assert VehicleDataAccess is not None


# ──────────── 10. Trend analysis delegates to tier correctly ─────────


def test_trend_analysis_old_data(populated):
    da, _ = populated
    result = da.get_trend_analysis("ENGINE_TEMP", days=30)
    assert "daily" in str(result["tier"])
    assert result.get("slope") is not None
    assert result.get("direction") in ("rising", "falling", "stable")
    assert "daily summaries" in result["attribution"]


# ──────────── 11. Anomaly analysis works on summaries ────────────────


def test_anomaly_analysis_old_data(populated):
    da, _ = populated
    result = da.get_anomaly_analysis("ENGINE_TEMP", days=30)
    assert "daily" in str(result["tier"])
    assert "anomalies_detected" in result


# ─────── 12. Predictive warnings work across tiers ──────────────────


def test_predictive_warnings(populated):
    da, _ = populated
    warnings = da.get_predictive_warnings(days=30)
    assert isinstance(warnings, list)


# ─────── 13. RetentionScheduler status ──────────────────────────────


def test_scheduler_status(tmp_db):
    compactor = Compactor(tmp_db)
    sched = RetentionScheduler(compactor, raw_days=15, summary_months=12, interval_seconds=3600)
    status = sched.get_status()
    assert "is_running" in status
    assert "raw_days" in status
    assert "summary_months" in status
    assert "next_run" in status
    assert status["raw_days"] == 15
    assert status["summary_months"] == 12


# ─────── 14. Storage stats include new fields ───────────────────────


def test_storage_stats_fields(populated):
    da, _ = populated
    stats = da.store.get_storage_stats()
    assert "days_searchable" in stats
    assert stats["days_searchable"] > 0
    assert "storage_per_tier" in stats
    assert "tier_1_raw_samples_kb" in stats["storage_per_tier"]
    assert "tier_2_daily_summaries_kb" in stats["storage_per_tier"]
    assert "tier_3_events_kb" in stats["storage_per_tier"]
    assert "last_compaction" in stats
    assert stats["last_compaction"] is not None


# ─────── 15. Digest search ──────────────────────────────────────────


def test_digest_search(populated):
    da, _ = populated
    # Digests are generated during compaction
    results = da.store.search_digests("summary")
    assert isinstance(results, list)
    # Each result should have date and digest
    if results:
        assert "date" in results[0]
        assert "digest" in results[0]


def test_no_hourly_rollup_table(populated):
    da, _ = populated
    conn = da.store._get_connection()
    try:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    finally:
        conn.close()
    assert "hourly_rollups" not in names
    assert "hourly_summaries" not in names


def test_trips_and_charge_sessions_kept(populated):
    da, _ = populated
    trips = da.store.get_events(event_type="trip", limit=50)
    charges = da.store.get_events(event_type="charge", limit=50)
    assert len(trips) >= 1
    assert len(charges) >= 1


def test_timeline_markers(populated):
    da, _ = populated
    tl = da.get_timeline(30)
    assert tl["raw_retention_days"] == 15
    assert any(m["kind"] == "daily" and m["summarized"] for m in tl["markers"])
    assert any(m.get("code") == "P0217" for m in tl["markers"])


def test_single_signal_lookup_via_query_layer(populated):
    da, _ = populated
    hist = da.get_signal_history("ENGINE_TEMP", time.time() - 21 * 86400, time.time() - 20 * 86400)
    assert hist["tier"] == "daily_summary"
    assert hist["detail_available"] is False

