#!/usr/bin/env python3
"""Frozen pregame POINT/GOAL probability audit against subsequently archived NHL results.

Read-only, research only. Never retrains/recalibrates any model; never authorizes
betting or silently turns non-participation into a loss. The first valid pregame
snapshot per (date, game, player, market) is compared with subsequent final
NHL outcome batches. Conflicting official snapshots remain unresolved.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import random
import sqlite3

UTC = timezone.utc
MARKETS = ("POINT", "GOAL")
FAMILY_COMPARISONS = ("frozen_probability", "player_smoothed_probability")


def timestamp(value):
    try:
        v = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return v.astimezone(UTC) if v.tzinfo else None
    except (ValueError, TypeError):
        return None


def number(value):
    try:
        v = float(value)
        return v if math.isfinite(v) and 0 <= v <= 1 else None
    except (ValueError, TypeError):
        return None


def logloss(y, p):
    p = max(1e-6, min(1 - 1e-6, p))
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def collect(archive):
    """Return unique pregame predictions, trustworthy outcome keys and exclusions."""
    con = sqlite3.connect("file:" + str(Path(archive).resolve()) + "?mode=ro", uri=True)
    exclusions = Counter()
    predictions = {}
    seen_outcomes = defaultdict(set)
    outcome_captures = defaultdict(list)
    try:
        for batch, payload, archived in con.execute(
            "SELECT batch_id,payload,archived_at_utc FROM batches WHERE kind='pregame' AND batch_id LIKE '%:predictions'"
        ):
            archived_at = timestamp(archived)
            try:
                items = json.loads(payload)
            except (TypeError, ValueError):
                exclusions["corrupt_pregame_payload"] += 1
                continue
            if not isinstance(items, list):
                exclusions["bad_pregame_shape"] += 1
                continue
            for row in items:
                try:
                    market = str(row.get("target", "")).upper()
                    key = (market, int(row["game_id"]), int(row["player_id"]))
                    dt = str(row["game_date"])
                    datetime.fromisoformat(dt)
                except (KeyError, ValueError, TypeError, AttributeError):
                    exclusions["missing_pregame_identity"] += 1
                    continue
                if market not in MARKETS:
                    exclusions["unsupported_market"] += 1
                    continue
                cap = timestamp(row.get("collected_at_utc"))
                event = timestamp(row.get("event_start_utc"))
                p = number(row.get("probability"))
                if not cap or not event or not archived_at or not (cap < event and archived_at < event):
                    exclusions["not_provably_pregame"] += 1
                    continue
                if event - cap > timedelta(days=7):
                    exclusions["stale_pregame_capture"] += 1
                    continue
                if p is None:
                    exclusions["bad_pregame_probability"] += 1
                    continue
                frozen = number(row.get("frozen_probability"))
                smooth = number(row.get("player_smoothed_probability"))
                point = dict(market=market, game_id=key[1], player_id=key[2],
                             date=dt, probability=p,
                             frozen_probability=frozen, player_smoothed_probability=smooth,
                             event_start=event, captured=cap, archived=archived_at,
                             is_production_approved=bool(row.get("production_approved", False)),
                             model_status=row.get("model_status"),
                             batch=batch)
                old = predictions.get(key)
                if old is None or (cap, archived_at, batch) < (
                    old["captured"], old["archived"], old["batch"]
                ):
                    predictions[key] = point

        for batch, payload, archived in con.execute(
            "SELECT batch_id,payload,archived_at_utc FROM batches WHERE kind='outcome'"
        ):
            observed_at = timestamp(archived)
            try:
                items = json.loads(payload)
            except (TypeError, ValueError):
                exclusions["corrupt_outcome_payload"] += 1
                continue
            if not isinstance(items, list):
                exclusions["bad_outcome_shape"] += 1
                continue
            for row in items:
                if row.get("game_status") != "FINAL" or row.get("evidence_status") != "final_stats_observed":
                    exclusions["outcome_not_final"] += 1
                    continue
                try:
                    key = (int(row["game_id"]), int(row["player_id"]))
                    goals = int(row["actual_goals"])
                    points = int(row["actual_points"])
                except (KeyError, ValueError, TypeError):
                    exclusions["bad_outcome_identity"] += 1
                    continue
                if goals < 0 or points < goals:
                    exclusions["bad_final_statistics"] += 1
                    continue
                if not observed_at:
                    exclusions["missing_outcome_timestamp"] += 1
                    continue
                seen_outcomes[key].add((goals, points))
                outcome_captures[key].append(observed_at)
    finally:
        con.close()

    conflicting = {key for key, values in seen_outcomes.items() if len(values) != 1}
    outcomes = {
        key: next(iter(values)) for key, values in seen_outcomes.items()
        if key not in conflicting
    }
    evaluated = {}
    for key, row in predictions.items():
        pair = (key[1], key[2])
        result = outcomes.get(pair)
        # An outcome recorded before the game's start is never proof of settlement.
        if result is not None and not any(ts > row["event_start"]
                                           for ts in outcome_captures[pair]):
            exclusions["outcome_before_puckdrop"] += 1
            result = None
        row = dict(row)
        row["label"] = int((result[0] if key[0] == "GOAL" else result[1]) >= 1) if result else None
        evaluated[key] = row
    return evaluated, {
        **dict(exclusions),
        "conflicting_official_identity_count": len(conflicting),
        "unique_pregame_picks": len(evaluated),
        "unique_final_outcomes": len(outcomes),
    }


def scores(rows, field):
    resolved = [row for row in rows if row["label"] is not None and row.get(field) is not None]
    if not resolved:
        return dict(n=0, brier=None, log_loss=None, hit_rate=None)
    n = len(resolved)
    return dict(n=n, game_dates=len({r["date"] for r in resolved}),
                hit_rate=round(sum(r["label"] for r in resolved) / n, 6),
                mean_probability=round(sum(r[field] for r in resolved) / n, 6),
                brier=round(sum((r["label"] - r[field]) ** 2 for r in resolved) / n, 6),
                log_loss=round(sum(logloss(r["label"], r[field]) for r in resolved) / n, 6))


def percentile(sorted_values, quantile):
    if not sorted_values:
        return None
    x = (len(sorted_values) - 1) * quantile
    a = int(x)
    b = min(a + 1, len(sorted_values) - 1)
    return sorted_values[a] * (1 - (x - a)) + sorted_values[b] * (x - a)


def paired_block_ci(rows, comparator, seed=20261128, replicates=2500, block=7):
    paired = [r for r in rows if r["label"] is not None and r.get(comparator) is not None]
    if not paired:
        return None
    bydate = defaultdict(lambda: [0.0, 0.0, 0])
    for r in paired:
        y, p, q = r["label"], r["probability"], r[comparator]
        item = bydate[r["date"]]
        item[0] += logloss(y, p) - logloss(y, q)
        item[1] += (y - p)**2 - (y - q)**2
        item[2] += 1
    dates = sorted(bydate)
    n_days = len(dates)
    delta_loss = sum(x[0] for x in bydate.values()) / len(paired)
    delta_brier = sum(x[1] for x in bydate.values()) / len(paired)
    rng = random.Random(seed)
    sums = []
    for _ in range(replicates):
        sampled = []
        for __ in range(math.ceil(n_days / block)):
            start = rng.randrange(n_days)
            sampled.extend(dates[(start + j) % n_days] for j in range(block))
        sampled = sampled[:n_days]
        total = sum(bydate[day][0] for day in sampled)
        denominator = sum(bydate[day][2] for day in sampled)
        sums.append(total / denominator)
    sums.sort()
    # Two predeclared comparisons (frozen and smoothed); Bonferroni coverage.
    alpha_each = 0.05 / len(FAMILY_COMPARISONS)
    limits = [percentile(sums, alpha_each / 2),
              percentile(sums, 1 - alpha_each / 2)]
    return dict(paired_n=len(paired), dates=n_days,
                delta_log_loss=round(delta_loss, 6),
                delta_brier=round(delta_brier, 6),
                ci_family_adjusted95=[round(x, 6) for x in limits],
                bootstrap_unit="circular 7 observed NHL dates",
                superiority_certified_on_this_observational_subset=(limits[1] < 0))


def audit(archive, protocol=None):
    rows, issues = collect(archive)
    bymarket = {market: [r for r in rows.values() if r["market"] == market] for market in MARKETS}
    output = dict(status="RESEARCH_ONLY_NO_MODEL_PROMOTION",
                  source_type="immutable pregame prediction batches + later NHL refreshed FINAL outcome batches",
                  source_archive=str(Path(archive)),
                  evidence_warning="NHL-derived archive, not independent review of every boxscore",
                  unverified_absences_never_counted_as_losses=True,
                  issues=issues,
                  markets={})
    for market, values in bymarket.items():
        dates = sorted({r["date"] for r in values})
        with_result = [r for r in values if r["label"] is not None]
        finished_game_ids = {r["game_id"] for r in with_result}
        completed = [r for r in values if r["game_id"] in finished_game_ids]
        entry = dict(unique_pregame=len(values), distinct_dates=len(dates),
                     newest_capture_day=dates[-1] if dates else None,
                     settled=len(with_result), unresolved=len(values)-len(with_result),
                     games_with_final_outcomes=len(finished_game_ids),
                     observed_participation_fraction=round(
                         len(with_result) / len(completed), 6
                     ) if completed else None,
                     outcomes=Counter(r["label"] for r in with_result),
                     probabilities={"model": scores(values, "probability")})
        if market == "GOAL":
            entry["probabilities"].update(
                {field: scores(values, field) for field in FAMILY_COMPARISONS})
            entry["paired_comparisons"] = {
                field: paired_block_ci(values, field) for field in FAMILY_COMPARISONS}
        # Plain JSON numbers, explicit no market ROI (verified matched individual odds unavailable).
        entry["outcomes"] = {str(k): v for k, v in entry["outcomes"].items()}
        output["markets"][market] = entry
    p = protocol or {}
    goal = output["markets"]["GOAL"]
    positives = int(goal["outcomes"].get("1", 0))
    observed = goal["observed_participation_fraction"]
    blockers = []
    if not p.get("promotion_allowed", False):
        blockers.append("promotion_locked_in_protocol")
    if goal["settled"] < p.get("min_rows", 20000):
        blockers.append("insufficient_settled_participations")
    if len({r["date"] for r in bymarket["GOAL"] if r["label"] is not None}) < p.get("min_dates", 90):
        blockers.append("insufficient_independent_game_dates")
    if positives < p.get("min_positive_outcomes", 2000):
        blockers.append("insufficient_positive_outcomes")
    if observed is None or observed < p.get("min_settlement_coverage", 0.98):
        blockers.append("unverified_participation_or_result_coverage")
    if issues.get("conflicting_official_identity_count", 0):
        blockers.append("conflicting_final_outcomes_require_independent_review")
    output["goal_promotion_blockers"] = blockers
    output["goal_promotion_authorized"] = False  # Never auto-promote a research model.
    output["notes"] = [
        "Market GOAL and POINT remain separate; not added to canonical bet outcomes.",
        "No success claimed on nonparticipants or missing finished games.",
        "Paired numerical improvement is not sufficient for promotion; prospective protocol is binding.",
        "The same player/game often occurs in both markets; cohorts are not independent.",
        "The final eligibility decision requires an explicit future locked review.",
        "PP1, probable goalies and xG are excluded unless independently evidenced before puckdrop.",
        "No payout/ROI is computed without verified market odds frozen pregame.",
    ]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--protocol", default=Path("configs/but_v2_prospective.json"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text()) if args.protocol.exists() else {}
    result = audit(args.archive, protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("PROSPECTIVE_MODEL_AUDIT=" + json.dumps(
        {k: v for k, v in result.items() if k != "markets"}, ensure_ascii=False))
    if result["goal_promotion_blockers"]:
        print("::warning::GOAL model remains research-only: " +
              ", ".join(result["goal_promotion_blockers"]))


if __name__ == "__main__":
    main()
