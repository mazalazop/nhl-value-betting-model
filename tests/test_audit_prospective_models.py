"""Offline provenance, conflict and conservative promotion tests for prospective models."""
import importlib.util
import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "audit_prospective_models", ROOT / "scripts" / "audit_prospective_models.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def new_db(path):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE batches (batch_id TEXT PRIMARY KEY,kind TEXT NOT NULL,payload TEXT NOT NULL,archived_at_utc TEXT NOT NULL)")
    return con


def put(con, ident, kind, rows, at):
    con.execute("INSERT INTO batches VALUES (?,?,?,?)", (ident, kind, json.dumps(rows), at))
    con.commit()


def pred(target="GOAL", p=.35, time="2026-10-17T12:00:00Z", price=.2):
    return dict(target=target, game_id=2026020001, player_id=8477777,
                game_date="2026-10-17", probability=p,
                frozen_probability=price, player_smoothed_probability=.15,
                collected_at_utc=time, event_start_utc="2026-10-17T19:00:00Z",
                model_status="research_only_operational",
                production_approved=False)


def out(goals=1, points=2):
    return dict(game_id=2026020001, player_id=8477777,
                actual_goals=goals, actual_points=points,
                game_status="FINAL", evidence_status="final_stats_observed")


def test_official_outcome_separates_point_from_goal(tmp_path):
    p = tmp_path / "archive.sqlite3"; c = new_db(p)
    put(c, "a:predictions", "pregame", [pred(), pred(target="POINT", p=.8)],
        "2026-10-17T12:00:01Z")
    put(c, "o", "outcome", [out(goals=0, points=1)], "2026-10-18T02:00:00Z")
    c.close()
    records, issues = m.collect(p)
    assert records[("GOAL", 2026020001, 8477777)]["label"] == 0
    assert records[("POINT", 2026020001, 8477777)]["label"] == 1
    summary = m.audit(p, {"promotion_allowed": True, "min_rows": 1,
                          "min_dates": 1, "min_positive_outcomes": 0,
                          "min_settlement_coverage": 0})
    assert summary["goal_promotion_authorized"] is False
    assert summary["markets"]["GOAL"]["probabilities"]["model"]["n"] == 1
    assert summary["markets"]["GOAL"]["paired_comparisons"]["frozen_probability"]["paired_n"] == 1


def test_pre_game_date_guards_and_no_future_probability_leak(tmp_path):
    p=tmp_path/"a.sqlite3";c=new_db(p)
    put(c, "bad:predictions", "pregame",
        [pred(time="2026-10-17T20:00:00Z")], "2026-10-17T20:00:01Z")
    put(c, "real:predictions", "pregame", [pred(p=.2)],
        "2026-10-17T12:00:01Z")
    put(c, "update:predictions", "pregame",
        [pred(p=.99,time="2026-10-17T18:00:00Z")], "2026-10-17T18:00:01Z")
    put(c, "o", "outcome", [out()], "2026-10-18T02:00:00Z")
    c.close()
    rows, flags = m.collect(p)
    assert len(rows)==1
    assert rows[("GOAL",2026020001,8477777)]["probability"] == .2
    assert flags["not_provably_pregame"] == 1


def test_conflicting_official_snapshots_never_counted(tmp_path):
    p=tmp_path/"a.sqlite3";c=new_db(p)
    put(c, "a:predictions", "pregame", [pred()], "2026-10-17T12:00:01Z")
    put(c, "a", "outcome", [out(1,2)], "2026-10-18T02:00:00Z")
    put(c, "b", "outcome", [out(0,1)], "2026-10-18T03:00:00Z")
    c.close()
    rows, flags = m.collect(p)
    assert flags["conflicting_official_identity_count"] == 1
    assert rows[("GOAL",2026020001,8477777)]["label"] is None
    d=m.audit(p)
    assert d["markets"]["GOAL"]["settled"] == 0
    assert "conflicting_final_outcomes_require_independent_review" in d["goal_promotion_blockers"]


def test_outcome_reported_before_game_does_not_settle(tmp_path):
    p=tmp_path/"a.sqlite3";c=new_db(p)
    put(c, "a:predictions", "pregame", [pred()], "2026-10-17T12:00:01Z")
    put(c, "o", "outcome", [out()], "2026-10-17T18:00:00Z")
    c.close()
    rows, flags = m.collect(p)
    assert rows[("GOAL",2026020001,8477777)]["label"] is None
    assert flags["outcome_before_puckdrop"] == 1


def test_no_player_final_means_unknown_not_defeat(tmp_path):
    p=tmp_path/"a.sqlite3";c=new_db(p)
    put(c, "a:predictions", "pregame", [pred()], "2026-10-17T12:00:01Z")
    c.close()
    x=m.audit(p)
    assert x["markets"]["GOAL"]["unresolved"] == 1
    assert x["markets"]["GOAL"]["settled"] == 0
    assert x["goal_promotion_authorized"] is False


def test_brier_logloss_and_block_bootstrap_are_deterministic():
    rows=[]
    for i in range(21):
        label=i%3==0
        rows.append(dict(date=f"2026-10-{i+3:02d}", label=int(label),
                         probability=.25, frozen_probability=.2,
                         player_smoothed_probability=.15))
    result=m.scores(rows,"probability")
    assert result["n"]==21 and result["hit_rate"]==round(7/21,6)
    a=m.paired_block_ci(rows,"frozen_probability")
    b=m.paired_block_ci(rows,"frozen_probability")
    assert a==b
    assert a["paired_n"]==21
    assert a["ci_family_adjusted95"][0]<=a["ci_family_adjusted95"][1]


def test_empty_archive_explicitly_nonpromoted(tmp_path):
    p=tmp_path/"a.sqlite3";c=new_db(p);c.close()
    x=m.audit(p,{"promotion_allowed":True})
    assert x["goal_promotion_authorized"] is False
    assert "insufficient_settled_participations" in x["goal_promotion_blockers"]


def test_outcome_sources_must_be_plausible(tmp_path):
    p=tmp_path/"a.sqlite3";c=new_db(p)
    put(c, "a:predictions", "pregame", [pred()], "2026-10-17T12:00:01Z")
    put(c, "o", "outcome", [out(goals=3,points=2)], "2026-10-18T02:00:00Z")
    c.close()
    x=m.audit(p)
    assert x["markets"]["GOAL"]["settled"] == 0
    assert x["issues"]["bad_final_statistics"] == 1
