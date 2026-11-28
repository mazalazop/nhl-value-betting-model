#!/usr/bin/env python3
"""Prospective-style precision audit for HENACHEL POINT, research only.

No retrospective PP1 / xG / shot observations enter a pregame decision.
Every decision for day D is based on *settled* canonical observations from
days strictly before D; the original prices and probabilities stay frozen.

A lower 95% Wilson bound above 80% is a demanding evidence criterion, not a
guarantee that the next bet wins. Without enough evidence the gate abstains.
This tool never changes POINT/GOAL models or the production pick register.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

START = dt.date(2026, 10, 3)
MIN_ODDS = 1.40
TARGET_PRECISION = 0.80
MIN_PRIOR_SETTLED = 60
MIN_PRIOR_DATES = 14

# FIXED, PREDECLARED hypotheses. No after-the-fact threshold tuning.
RULES = (
    "all_pregame",
    "point_probability_ge_065",
    "point_probability_ge_070",
    "point_probability_ge_065_and_odds_le_150",
    "top_one_by_probability_per_day",
)


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) else None
    except (ValueError, TypeError):
        return None


def timestamp(value):
    try:
        t = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return t.astimezone(dt.timezone.utc) if t.tzinfo else None
    except (ValueError, TypeError):
        return None


def wilson(wins, n, z=1.96):
    if not n:
        return None
    p = wins / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    rad = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"low": round(max(0, mid - rad), 6),
            "high": round(min(1, mid + rad), 6)}


def read_canonical(result_dir):
    records, rejected, seen = [], Counter(), set()
    for path in sorted(Path(result_dir).glob("*-canonical-picks.csv")):
        try:
            file_day = dt.date.fromisoformat(path.name[:10])
        except ValueError:
            rejected["bad_filename_date"] += 1
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                if row.get("archive_group") != "CANONICAL" or row.get("market") != "POINT":
                    rejected["other_group_or_market"] += 1
                    continue
                try:
                    d = dt.date.fromisoformat(row.get("date", ""))
                except ValueError:
                    rejected["bad_date"] += 1
                    continue
                if d != file_day or d < START:
                    rejected["date_mismatch_or_outside_scope"] += 1
                    continue
                if row.get("archive_provenance") != "PREGAME_CANONICAL_LEDGER":
                    rejected["not_frozen_canonical"] += 1
                    continue
                cap = timestamp(row.get("captured_at"))
                start = timestamp(row.get("start_time_utc"))
                if not cap or not start or not dt.timedelta(0) < start - cap <= dt.timedelta(days=7):
                    rejected["not_provably_pregame"] += 1
                    continue
                p = number(row.get("model_probability"))
                odds = number(row.get("odds"))
                if p is None or not 0 <= p <= 1 or odds is None or odds < MIN_ODDS:
                    rejected["ineligible_probability_or_odds"] += 1
                    continue
                identity = (d, row.get("bet_id"), "POINT")
                if not all(row.get(k) for k in ("bet_id", "id_match", "id_joueur")):
                    rejected["missing_identity"] += 1
                    continue
                if identity in seen:
                    raise ValueError(f"Duplicate frozen canonical pick: {identity}")
                seen.add(identity)
                status = row.get("status")
                if status == "SETTLED" and row.get("official_win") not in {"0", "1"}:
                    raise ValueError(f"Settled without valid outcome: {identity}")
                outcome = int(row["official_win"]) if status == "SETTLED" else None
                records.append({"date": d.isoformat(), "bet_id": row["bet_id"],
                                "id_match": row["id_match"], "id_joueur": row["id_joueur"],
                                "p": p, "odds": odds, "outcome": outcome,
                                "source": path.name})
    # Deterministic daily top-one selection uses only frozen pregame p and ID.
    by_day = defaultdict(list)
    for row in records:
        by_day[row["date"]].append(row)
    for batch in by_day.values():
        top = sorted(batch, key=lambda x: (-x["p"], x["bet_id"]))[0]
        top["top_one"] = True
    return sorted(records, key=lambda x: (x["date"], x["bet_id"])), dict(rejected)


def in_rule(r, rule):
    if rule == "all_pregame":
        return True
    if rule == "point_probability_ge_065":
        return r["p"] >= .65
    if rule == "point_probability_ge_070":
        return r["p"] >= .70
    if rule == "point_probability_ge_065_and_odds_le_150":
        return r["p"] >= .65 and r["odds"] <= 1.50
    if rule == "top_one_by_probability_per_day":
        return r.get("top_one", False)
    raise ValueError(f"Unknown rule {rule}")


def describe(rows):
    n = len(rows)
    settled = [r for r in rows if r["outcome"] is not None]
    k = len(settled)
    w = sum(r["outcome"] for r in settled)
    roi = sum((r["outcome"] * r["odds"] - 1) for r in settled) / k if k else None
    brier = sum((r["outcome"] - r["p"]) ** 2 for r in settled) / k if k else None
    return {"selected": n, "settled": k, "unresolved": n - k, "wins": w,
            "hit_rate": round(w / k, 6) if k else None,
            "wilson95": wilson(w, k),
            "unit_stake_simulated_roi": round(roi, 6) if roi is not None else None,
            "brier": round(brier, 6) if brier is not None else None}


def walk_forward(records, rule, min_prior=MIN_PRIOR_SETTLED,
                 min_days=MIN_PRIOR_DATES, target=TARGET_PRECISION):
    """For each day D: decide from historical outcomes strictly before D."""
    batches = defaultdict(list)
    for r in records:
        batches[r["date"]].append(r)
    dates = sorted(batches)
    accepted = []
    decisions = []
    for index, day in enumerate(dates):
        previous_dates = dates[:index]
        if len(previous_dates) < min_days:
            continue
        # Exactly zero information from the evaluation day's result is used.
        prior = [r for d in previous_dates for r in batches[d] if in_rule(r, rule)]
        validated = [r for r in prior if r["outcome"] is not None]
        w = sum(r["outcome"] for r in validated)
        bound = wilson(w, len(validated))
        approved = (len(validated) >= min_prior and bound is not None
                    and bound["low"] > target)
        current = [r for r in batches[day] if in_rule(r, rule)]
        if approved:
            accepted.extend(current)
        decisions.append({"date": day, "prior_n": len(validated),
                          "prior_wins": w, "prior_wilson_lower95": bound["low"] if bound else None,
                          "approved_before_game": approved,
                          "candidate_count": len(current),
                          "released_count": len(current) if approved else 0})
    return {"evaluation_days": len(decisions),
            "active_days": sum(d["released_count"] > 0 for d in decisions),
            "abstained_days": sum(d["released_count"] == 0 for d in decisions),
            "approved_releases": describe(accepted),
            "decisions": decisions}


def build_report(records, rejected):
    if not records:
        return {"status": "NO_VERIFIED_CANONICAL_POINTS",
                "source": "official NHL audited pregame CANONICAL POINT only",
                "rejections": rejected, "rules": {}, "notes": [
                    "No performance or improvement can be inferred without admissible input."]}
    rules = {}
    for rule in RULES:
        matched = [r for r in records if in_rule(r, rule)]
        rules[rule] = {
            "descriptive_full_history_not_out_of_sample": describe(matched),
            "walk_forward_conservative_gate": walk_forward(records, rule),
        }
    approved = [key for key, value in rules.items()
                if value["walk_forward_conservative_gate"]["approved_releases"]["settled"] >= 40
                and value["walk_forward_conservative_gate"]["approved_releases"]["wilson95"] is not None
                and value["walk_forward_conservative_gate"]["approved_releases"]["wilson95"]["low"] > TARGET_PRECISION]
    # Economic bins are DESCRIPTIVE ONLY. Historical observed results were already
    # inspected; they are not a prospective rule search or an automatic value gate.
    economic = {
        "all_frozen_pregame": describe(records),
        "model_ev_nonnegative": describe([
            r for r in records if r["p"] * r["odds"] >= 1.0]),
        "model_ev_at_least_5pct": describe([
            r for r in records if r["p"] * r["odds"] >= 1.05]),
        "model_ev_negative": describe([
            r for r in records if r["p"] * r["odds"] < 1.0]),
    }
    return {
        "status": "RESEARCH_ONLY_NO_PRODUCTION_MODEL_CHANGE",
        "through_date": max(r["date"] for r in records),
        "eligible_pregame_point_picks": len(records),
        "outcomes_settled": sum(r["outcome"] is not None for r in records),
        "dates_observed": len(set(r["date"] for r in records)),
        "rejections": rejected,
        "precision_target": TARGET_PRECISION,
        "minimum_prior_settled": MIN_PRIOR_SETTLED,
        "minimum_prior_dates": MIN_PRIOR_DATES,
        "validated_above_80pct_rule_names": approved,
        "rules": rules,
        "model_edge_economic_diagnostic_retrospective_only": economic,
        "betting_promotion_authorized": False,
        "limitations": [
            "The original all-picks dataset is a selected cohort, not all NHL skaters.",
            "Walk-forward gate learns ONLY from outcomes of earlier match dates.",
            "Threshold hypotheses are fixed; results do not prove a causal improvement.",
            "Conservative Wilson intervals do not account for intra-game dependence or multiple hypotheses.",
            "No archived trustworthy pregame PP1/xG/shot feature: these are NOT used.",
            "No postgame shots, TOI, xG, assists or goals are used to choose pregame picks.",
            "Do not retrain POINT/GOAL or claim an 80% hit rate from retrospective subsets.",
            "Absent or unresolved players are not losses and never train the gate.",
            "Prices are pregame archival odds, NOT actual personal wagers.",
            "A positive model edge is NOT proven profitable; do not activate an EV gate from these same outcomes.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path("docs/postgame/results"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/postgame/longitudinal/point_precision_walkforward.json"))
    args = parser.parse_args()
    records, rejected = read_canonical(args.results_dir)
    result = build_report(records, rejected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print("POINT_PRECISION_RESEARCH=" + json.dumps(
        {k: v for k, v in result.items() if k != "rules"}, separators=(",", ":")))


if __name__ == "__main__":
    main()
