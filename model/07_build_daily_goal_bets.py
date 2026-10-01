#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build independent BUTEURS picks from matched model/odds rows."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-date", required=True)
    ap.add_argument("--input-csv", default=str(OUT / "06b_matched_goal_edges.csv"))
    ap.add_argument("--max-picks", type=int, default=5)
    ap.add_argument("--one-pick-per-player", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-odds", type=float, default=1.01)
    args = ap.parse_args()

    path = Path(args.input_csv)
    if not path.exists():
        raise FileNotFoundError(path)

    try:
        df = pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        df = pd.DataFrame()

    if df.empty:
        out = pd.DataFrame()
    else:
        df = df[df["run_date"].astype(str) == args.run_date].copy()
        df["model_probability"] = pd.to_numeric(df["model_probability"], errors="coerce")
        df["edge_probability"] = pd.to_numeric(df["edge_probability"], errors="coerce")
        df["odds_decimal"] = pd.to_numeric(df["odds_decimal"], errors="coerce")

        # Odds > 1 is the only mechanical odds constraint. Edge stays informative.
        df = df[df["odds_decimal"] >= args.min_odds].copy()
        df = df.sort_values(
            ["model_probability", "edge_probability", "odds_decimal"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        if args.one_pick_per_player:
            df = df.drop_duplicates("player_name", keep="first").reset_index(drop=True)

        out = df.head(args.max_picks).copy()
        out["recommended_flag"] = True
        out["recommendation_rank"] = np.arange(1, len(out) + 1)
        out["is_value_bet"] = (out["edge_probability"] >= 0.02).astype(int)
        out["is_value_bet_label"] = np.where(out["is_value_bet"] == 1, "yes", "no")

    out_path = OUT / "07_daily_goal_bets.csv"
    summary_path = OUT / "07_daily_goal_bets_summary.json"
    out.to_csv(out_path, index=False)

    summary = {
        "status": "ok",
        "run_date": args.run_date,
        "market": "player_goals",
        "max_picks": args.max_picks,
        "input_candidates": int(len(df)) if not df.empty else 0,
        "daily_goal_picks": int(len(out)),
        "positive_ev_rows": int((out["ev_per_unit"] > 0).sum()) if not out.empty and "ev_per_unit" in out.columns else 0,
        "output": str(out_path),
    }
    summary_path.write_text(__import__("json").dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
