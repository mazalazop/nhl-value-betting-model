"""Prevent leakage and spurious >80% claims in research-only POINT precision."""
import csv
import datetime as dt
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts" / "point_precision_walkforward.py"
SPEC = importlib.util.spec_from_file_location("point_precision_walkforward", MODULE)
precision = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = precision
SPEC.loader.exec_module(precision)


def pick(day="2026-10-17", name="first", win="1", probability=".70",
         price="1.50", status="SETTLED"):
    return {
        "date": day, "market": "POINT", "archive_group": "CANONICAL",
        "archive_provenance": "PREGAME_CANONICAL_LEDGER",
        "player": name, "team": "NJD", "bet_id": name,
        "id_match": "2026021234", "id_joueur": "8483999",
        "captured_at": day + "T12:00:00Z",
        "start_time_utc": day + "T23:00:00Z",
        "model_probability": probability, "odds": price,
        "official_win": win if status == "SETTLED" else "",
        "status": status, "shots": "10", "attempts_pbp": "12",
    }


def write_file(d, day, records):
    with (d / (day + "-canonical-picks.csv")).open(
            "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def test_filters_out_comparators_and_keeps_canonical_only(tmp_path):
    examples = [
        pick(name="a"),
        {**pick(name="b"), "archive_group": "CLASSEUR"},
        {**pick(name="c"), "archive_provenance": "RECONSTRUCTED"},
        {**pick(name="d"), "captured_at": "2026-10-18T00:00:00Z"},
        pick(name="e", price="1.39"),
    ]
    write_file(tmp_path, "2026-10-17", examples)
    rows, rejected = precision.read_canonical(tmp_path)
    assert [r["bet_id"] for r in rows] == ["a"]
    assert sum(rejected.values()) == 4


def test_abstain_without_sufficient_prior_evidence():
    records = [{"date": "2026-10-17", "bet_id": "a", "p": .8, "odds": 1.4,
                "outcome": 1, "top_one": True}]
    results = precision.walk_forward(records, "all_pregame")
    assert results["approved_releases"]["settled"] == 0
    assert results["evaluation_days"] == 0


def test_walk_forward_never_uses_same_day_outcomes():
    records = []
    for i in range(20):
        d = (dt.date(2026, 10, 3) + dt.timedelta(days=i)).isoformat()
        for j in range(5):
            records.append({"date": d, "bet_id": f"{d}-{j}", "p": .7,
                            "odds": 1.5, "outcome": 1 if i < 15 else 0,
                            "top_one": j == 0})
    # Prior to day 15, successes from first 15 days are enough to pass Wilson.
    a = precision.walk_forward(records, "all_pregame", min_prior=50,
                               min_days=10)
    d15 = (dt.date(2026, 10, 3) + dt.timedelta(days=15)).isoformat()
    observation = next(d for d in a["decisions"] if d["date"] == d15)
    assert observation["prior_n"] == 75
    assert observation["approved_before_game"] is True
    assert observation["released_count"] == 5
    # Editing day 15 outcomes cannot affect decisions on day 15.
    for r in records:
        if r["date"] == d15:
            r["outcome"] = 1
    b = precision.walk_forward(records, "all_pregame", min_prior=50,
                               min_days=10)
    other = next(d for d in b["decisions"] if d["date"] == d15)
    assert observation == other


def test_no_claim_above_80_with_low_sample():
    records = []
    for i in range(20):
        d = (dt.date(2026, 10, 3) + dt.timedelta(days=i)).isoformat()
        records.append({"date": d, "bet_id": str(i), "p": .71,
                        "odds": 1.5, "outcome": 1 if i < 18 else 0,
                        "top_one": True})
    report = precision.build_report(records, {})
    assert report["validated_above_80pct_rule_names"] == []
    assert report["rules"]["point_probability_ge_070"]["walk_forward_conservative_gate"]["approved_releases"]["settled"] == 0


def test_absences_are_unresolved_not_failures(tmp_path):
    rows = [pick(name="a", win="1"),
            pick(name="b", win="", status="PLAYER_ID_NOT_IN_TEAM_BOXSCORE")]
    write_file(tmp_path, "2026-10-17", rows)
    found, ignored = precision.read_canonical(tmp_path)
    stats = precision.describe(found)
    assert stats["selected"] == 2
    assert stats["settled"] == 1
    assert stats["unresolved"] == 1
    assert stats["hit_rate"] == 1.0


def test_top_one_ranks_before_seeing_outcomes(tmp_path):
    examples = [pick(name="low", win="1", probability=".60"),
                pick(name="high", win="0", probability=".70")]
    write_file(tmp_path, "2026-10-17", examples)
    found, _ = precision.read_canonical(tmp_path)
    top = [r for r in found if precision.in_rule(r, "top_one_by_probability_per_day")]
    assert len(top) == 1 and top[0]["bet_id"] == "high"
    assert precision.describe(top)["wins"] == 0


def test_duplicate_bet_id_rejected(tmp_path):
    examples = [pick(name="a"), pick(name="a")]
    write_file(tmp_path, "2026-10-17", examples)
    with pytest.raises(ValueError, match="Duplicate frozen canonical pick"):
        precision.read_canonical(tmp_path)


def test_wilson_100_percent_ten_is_not_certified_eighty():
    bound = precision.wilson(10, 10)
    assert bound["low"] < 0.8
    assert precision.wilson(275, 469)["high"] < 0.7


def test_economic_edge_is_descriptive_and_cannot_enable_betting():
    records = [
        {"date": "2026-10-17", "bet_id": "one", "p": .7,
         "odds": 1.5, "outcome": 0, "top_one": True},
        {"date": "2026-10-18", "bet_id": "two", "p": .6,
         "odds": 1.4, "outcome": 1, "top_one": True},
        {"date": "2026-10-19", "bet_id": "three", "p": .8,
         "odds": 1.5, "outcome": None, "top_one": True},
    ]
    report = precision.build_report(records, {})
    diagnostics = report["model_edge_economic_diagnostic_retrospective_only"]
    assert report["betting_promotion_authorized"] is False
    assert diagnostics["all_frozen_pregame"]["settled"] == 2
    assert diagnostics["model_ev_nonnegative"]["selected"] == 2
    assert diagnostics["model_ev_nonnegative"]["unresolved"] == 1
    assert diagnostics["model_ev_nonnegative"]["unit_stake_simulated_roi"] == -1.0
    assert diagnostics["model_ev_negative"]["wins"] == 1
    assert report["validated_above_80pct_rule_names"] == []
