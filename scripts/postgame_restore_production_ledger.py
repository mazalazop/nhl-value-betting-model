#!/usr/bin/env python3
"""Restore a dated CANONICAL ledger from a real, successful POINT production job.

A context-only Actions run can re-upload an older artifact and still report
"success". Never use those runs as the source for postgame reconciliation.
The original artifact is not modified; this only materializes a working copy.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

from postgame_import_ledger import extract, parse_utc

ARTIFACT_PREFIX = "henachel-canonical-history-"
PRODUCTION_JOB = "Pipeline modèle POINTS"


def gh(*args):
    return subprocess.check_output(["gh", *map(str, args)], text=True)


def choose_production_run(runs, fetch_jobs, fetch_artifacts):
    """Select the newest *executed* POINT job with an unexpired ledger artifact."""
    for run in sorted(runs, key=lambda x: x.get("createdAt", ""), reverse=True):
        if run.get("conclusion") != "success":
            continue
        run_id = int(run["databaseId"])
        jobs = fetch_jobs(run_id)
        if not any(j.get("name") == PRODUCTION_JOB and j.get("conclusion") == "success"
                   for j in jobs):
            continue
        artifacts = fetch_artifacts(run_id)
        valid = [a for a in artifacts if a.get("name", "").startswith(ARTIFACT_PREFIX)
                 and not a.get("expired", False)]
        if valid:
            return run_id, max(valid, key=lambda a: a.get("created_at", ""))
    raise RuntimeError(
        "No successful POINT production job with an unexpired canonical ledger "
        "in the inspected runs. Context-only success cannot substitute for production."
    )


def inspect_ledger(source, snapshots, as_of, max_lag_days=3):
    with Path(source).open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    grouped, exclusions = extract(records)
    ids = {row["bet_id"] for rows in grouped.values() for row in rows}
    if not ids:
        raise RuntimeError("Selected production ledger contains no admissible pregame picks")
    # Previously archived identities must remain present in a newer source.
    # An artifact missing historical picks is a regression, not a fresh ledger.
    old_ids = set()
    for path in sorted(Path(snapshots).glob("*-canonical.csv")):
        with path.open(encoding="utf-8-sig", newline="") as handle:
            old_ids.update(row["bet_id"] for row in csv.DictReader(handle)
                           if row.get("bet_id"))
    lost = old_ids - ids
    if lost:
        raise RuntimeError(
            f"Production history omits {len(lost)} already archived bet IDs; "
            f"example {sorted(lost)[:3]}. Refusing a regressed ledger."
        )
    latest = max(grouped)
    earliest = min(grouped)
    latest_date = dt.date.fromisoformat(latest)
    lag = (as_of - latest_date).days
    since = max(dt.date(2026, 10, 3), as_of - dt.timedelta(days=13))
    gaps = []
    date = since
    while date <= as_of:
        if str(date) not in grouped:
            gaps.append(str(date))
        date += dt.timedelta(days=1)
    captured = [parse_utc(row.get("captured_at")) for row in records]
    captured = [value for value in captured if value is not None]
    health = {
        "status": "FRESH" if lag <= max_lag_days else "STALE",
        "original_rows": len(records),
        "eligible_pregame_picks": len(ids),
        "first_eligible_date": earliest,
        "last_eligible_date": latest,
        "latest_capture_utc": max(captured).isoformat() if captured else None,
        "lag_days_vs_as_of": lag,
        "as_of_date": str(as_of),
        "excluded_rows_by_reason": exclusions,
        "archived_pregame_ids_checked": len(old_ids),
        "recent_dates_without_eligible_pregame_snapshot": gaps,
        "gap_interpretation": (
            "Unverified missing source, NOT proven picks or losses; "
            "some dates may have no NHL games or no eligible bets."
        ),
    }
    if lag > max_lag_days:
        raise RuntimeError(
            f"Stale production ledger: last eligible pick {latest}; "
            f"as-of {as_of}; lag {lag} days. Refusing a green no-op audit."
        )
    return health


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snapshots", type=Path, required=True)
    parser.add_argument("--status-path", type=Path, required=True)
    parser.add_argument("--as-of", type=dt.date.fromisoformat,
                        default=dt.datetime.now(dt.timezone.utc).date()
                        - dt.timedelta(days=1))
    parser.add_argument("--max-lag-days", type=int, default=3)
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()
    if not os.environ.get("GH_TOKEN") or not os.environ.get("GITHUB_REPOSITORY"):
        raise RuntimeError("GH_TOKEN and GITHUB_REPOSITORY are required")
    repo = os.environ["GITHUB_REPOSITORY"]
    runs = json.loads(gh("run", "list", "-w", "henachel-points-global",
                         "-b", "main", "-L", str(args.limit),
                         "--json", "databaseId,conclusion,createdAt"))
    def get_jobs(run_id):
        return json.loads(gh("run", "view", run_id, "--json", "jobs"))["jobs"]
    def get_artifacts(run_id):
        return json.loads(gh("api", f"repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100"))["artifacts"]
    run_id, artifact = choose_production_run(runs, get_jobs, get_artifacts)
    destination = args.output_dir / f"run-{run_id}"
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gh", "run", "download", str(run_id), "--pattern",
                    ARTIFACT_PREFIX + "*", "--dir", str(destination)],
                   check=True)
    matches = list(destination.rglob("master_daily_bets_history.csv"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one canonical ledger in {run_id}, found {len(matches)}")
    source = matches[0]
    health = inspect_ledger(source, args.snapshots, args.as_of, args.max_lag_days)
    health.update(source_run_id=run_id, artifact_id=artifact["id"],
                  artifact_created_at_utc=artifact.get("created_at"),
                  ledger_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    copy = args.output_dir / "master_daily_bets_history.csv"
    if copy.exists():
        raise FileExistsError(f"Refusing to overwrite an existing working ledger: {copy}")
    shutil.copyfile(source, copy)
    args.status_path.parent.mkdir(parents=True, exist_ok=True)
    args.status_path.write_text(json.dumps(health, indent=2, ensure_ascii=False) + "\n",
                                encoding="utf-8")
    print("CANONICAL_SOURCE_HEALTH=" + json.dumps(health, separators=(",", ":")))
    if health["recent_dates_without_eligible_pregame_snapshot"]:
        print("::warning::CANONICAL dates without admissible pregame snapshots: " +
              ", ".join(health["recent_dates_without_eligible_pregame_snapshot"]))
    print(f"CANONICAL_ARTIFACT_RUN={run_id}")


if __name__ == "__main__":
    main()
