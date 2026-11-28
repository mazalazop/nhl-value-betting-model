"""Offline guards for production-only CANONICAL source restoration."""
import csv
import datetime as dt
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "postgame_restore_production_ledger", ROOT / "scripts/postgame_restore_production_ledger.py")
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)


def record(bet_id="old", date="2026-11-19", price="1.50"):
    return dict(date_match=date, market="player_points", stat="points", threshold="1",
                recommended_flag="1", captured_at=date + "T12:00:00Z",
                start_time_utc=date + "T20:00:00Z", odds_decimal=price,
                model_probability="0.6", bet_id=bet_id, id_match="2026021234",
                id_joueur="8471234", player_name="Test Skater", team="NJD",
                bookmaker="Unibet")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_context_only_newest_cannot_hide_actual_production():
    runs = [
        {"databaseId": 102, "createdAt": "2026-11-28T04:00:00Z", "conclusion": "success"},
        {"databaseId": 101, "createdAt": "2026-11-27T20:00:00Z", "conclusion": "success"},
        {"databaseId": 100, "createdAt": "2026-11-26T20:00:00Z", "conclusion": "failure"},
    ]
    def jobs(number):
        return ([{"name": "Intraday context only — no model or bookmaker execution",
                  "conclusion": "success"},
                 {"name": restore.PRODUCTION_JOB, "conclusion": "skipped"}] if number == 102
                else [{"name": restore.PRODUCTION_JOB, "conclusion": "success"}])
    def artifacts(number):
        return [{"id": number, "name": restore.ARTIFACT_PREFIX + "history",
                 "created_at": "2026-11-27T22:00:00Z", "expired": False}]
    chosen, artifact = restore.choose_production_run(runs, jobs, artifacts)
    assert chosen == 101 and artifact["id"] == 101


def test_success_without_executed_model_is_not_an_eligible_source():
    runs = [{"databaseId": 10, "createdAt": "2026-11-28T02:00:00Z",
             "conclusion": "success"}]
    with pytest.raises(RuntimeError, match="Context-only"):
        restore.choose_production_run(
            runs, lambda _: [{"name": restore.PRODUCTION_JOB, "conclusion": "skipped"}],
            lambda _: [{"name": restore.ARTIFACT_PREFIX + "x", "expired": False}])


def test_expired_production_artifact_is_not_accepted():
    runs = [{"databaseId": 20, "createdAt": "2026-11-28T02:00:00Z",
             "conclusion": "success"}]
    with pytest.raises(RuntimeError, match="No successful"):
        restore.choose_production_run(
            runs, lambda _: [{"name": restore.PRODUCTION_JOB, "conclusion": "success"}],
            lambda _: [{"name": restore.ARTIFACT_PREFIX + "x", "expired": True}])


def test_recent_verified_snapshot_tracks_missing_dates_without_inventing_losses(tmp_path):
    source = tmp_path / "source.csv"
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    rows = [record(), record("latest", "2026-11-27")]
    write_csv(source, rows)
    # The previously certified pre-match snapshot must remain in the new ledger.
    write_csv(snapshots / "2026-11-19-canonical.csv", [{
        "bet_id": "old",
    }])
    health = restore.inspect_ledger(source, snapshots, dt.date(2026, 11, 27))
    assert health["eligible_pregame_picks"] == 2
    assert health["last_eligible_date"] == "2026-11-27"
    assert health["lag_days_vs_as_of"] == 0
    assert "2026-11-20" in health["recent_dates_without_eligible_pregame_snapshot"]
    assert health["status"] == "FRESH"
    assert "NOT proven picks or losses" in health["gap_interpretation"]


def test_stale_success_does_not_pass_green_audit(tmp_path):
    source = tmp_path / "source.csv"
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    write_csv(source, [record()])
    with pytest.raises(RuntimeError, match="Stale production ledger"):
        restore.inspect_ledger(source, snapshots, dt.date(2026, 11, 28))


def test_history_regression_cannot_erase_previously_certified_identity(tmp_path):
    source = tmp_path / "source.csv"
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    write_csv(source, [record("new", "2026-11-27")])
    write_csv(snapshots / "2026-11-19-canonical.csv", [{"bet_id": "old"}])
    with pytest.raises(RuntimeError, match="omits 1"):
        restore.inspect_ledger(source, snapshots, dt.date(2026, 11, 27))


def test_undocumented_old_prices_stay_excluded_from_economic_cohort(tmp_path):
    source = tmp_path / "source.csv"
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    write_csv(source, [record("low", "2026-11-26", "1.35"),
                       record("eligible", "2026-11-27", "1.50")])
    health = restore.inspect_ledger(source, snapshots, dt.date(2026, 11, 27))
    assert health["eligible_pregame_picks"] == 1
    assert health["excluded_rows_by_reason"]["bad_price_or_probability"] == 1
