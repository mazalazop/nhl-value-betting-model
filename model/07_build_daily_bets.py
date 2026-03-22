#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_INPUT_CSV = DEFAULT_OUTPUTS_DIR / "06_matched_point_edges.csv"

REQUIRED_COLUMNS = [
    "bet_id",
    "run_date",
    "bet_status",
    "result",
    "actual_stat_value",
    "settled_at",
    "recommended_flag",
    "recommendation_rank",
    "date_match",
    "player_name",
    "team",
    "opponent",
    "bookmaker",
    "market",
    "stat",
    "threshold",
    "odds_decimal",
    "implied_probability",
    "model_probability",
    "edge_probability",
]

OPTIONAL_NUMERIC_COLUMNS = [
    "ev_per_unit",
    "kelly_fraction",
    "rank_proba_sur_date",
    "rank_proba_sur_match",
    "value_gap",
    "hard_exclude_hot_streak_pre",
]

OPTIONAL_TEXT_COLUMNS = [
    "is_positive_ev",
    "hard_exclude_hot_streak_reason",
]


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")



def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construire les picks quotidiens à partir des candidats matchés du script 06, "
            "avec priorité à la probabilité modèle."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=str(DEFAULT_INPUT_CSV),
        help="Chemin du CSV produit par 06.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUTS_DIR),
        help="Dossier outputs du repo.",
    )
    parser.add_argument(
        "--run-date",
        type=str,
        default=None,
        help="Date du run au format YYYY-MM-DD. Si absente, prise depuis le fichier 06.",
    )
    parser.add_argument(
        "--max-picks",
        type=int,
        default=10,
        help="Nombre maximum de picks recommandés.",
    )
    parser.add_argument(
        "--min-odds",
        type=float,
        default=1.40,
        help="Cote minimale par défaut pour conserver un pick.",
    )
    parser.add_argument(
        "--override-min-model-proba",
        type=float,
        default=0.90,
        help="Seuil de probabilité modèle qui autorise un pick même sous la cote minimale.",
    )
    parser.add_argument(
        "--value-threshold",
        type=float,
        default=0.02,
        help="Seuil de gap proba modèle - proba implicite pour marquer is_value_bet=1.",
    )
    parser.add_argument(
        "--one-pick-per-player",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Conserver au plus un pick par joueur (utiliser --no-one-pick-per-player pour désactiver).",
    )
    parser.add_argument(
        "--disable-hot-streak-exclude",
        action="store_true",
        help="Désactiver l'exclusion des joueurs en série chaude anormale.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Load / validate
# -----------------------------------------------------------------------------

def load_candidates(path: Path) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path, low_memory=False)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier 06 : {missing}")

    df = df.copy()

    numeric_cols = [
        "threshold",
        "odds_decimal",
        "implied_probability",
        "model_probability",
        "edge_probability",
        *OPTIONAL_NUMERIC_COLUMNS,
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    text_cols = [
        "run_date",
        "date_match",
        "bet_status",
        "result",
        "player_name",
        "team",
        "opponent",
        "market",
        "stat",
        "bookmaker",
        *OPTIONAL_TEXT_COLUMNS,
    ]
    for col in text_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].astype(str)

    df["bet_status"] = df["bet_status"].replace({"nan": "pending", "": "pending"})
    df["result"] = df["result"].replace({"nan": ""})

    # value gap / flag for information only
    df["value_gap"] = df["model_probability"] - df["implied_probability"]

    # hot streak flag if present, else neutral
    if "hard_exclude_hot_streak_pre" not in df.columns:
        df["hard_exclude_hot_streak_pre"] = 0
    df["hard_exclude_hot_streak_pre"] = pd.to_numeric(
        df["hard_exclude_hot_streak_pre"], errors="coerce"
    ).fillna(0)
    df["hard_exclude_hot_streak_pre"] = df["hard_exclude_hot_streak_pre"].astype(int)

    return df



def choose_run_date(df: pd.DataFrame, explicit_run_date: str | None) -> str:
    if explicit_run_date:
        return explicit_run_date

    candidates = [x for x in df["run_date"].dropna().astype(str).unique().tolist() if x and x != "nan"]
    if not candidates:
        raise ValueError("Impossible de déterminer run_date. Passe --run-date explicitement.")

    # By default, use the most recent run date present in the 06 file.
    return sorted(candidates)[-1]


# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------

def build_daily_bets(
    candidates_df: pd.DataFrame,
    run_date: str,
    max_picks: int,
    min_odds: float,
    override_min_model_proba: float,
    value_threshold: float,
    one_pick_per_player: bool,
    disable_hot_streak_exclude: bool,
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    df = candidates_df.copy()
    df = df[df["run_date"] == run_date].copy()

    stats: Dict[str, Any] = {
        "rows_after_run_date_filter": int(len(df)),
        "rows_removed_low_odds": 0,
        "rows_removed_hot_streak": 0,
        "rows_after_player_dedup": 0,
    }

    if df.empty:
        return df, stats

    # Value-bet flag kept as secondary information only.
    df["is_value_bet"] = (df["value_gap"] >= value_threshold).astype(int)

    # Eligibility: odds >= min_odds OR model_probability >= override threshold
    odds_ok = df["odds_decimal"] >= min_odds
    proba_override_ok = df["model_probability"] >= override_min_model_proba
    keep_mask = odds_ok | proba_override_ok
    stats["rows_removed_low_odds"] = int((~keep_mask).sum())
    df = df[keep_mask].copy()

    if df.empty:
        return df, stats

    # Hard exclude hot streak if present
    if not disable_hot_streak_exclude:
        hot_mask = df["hard_exclude_hot_streak_pre"].fillna(0).astype(int) == 1
        stats["rows_removed_hot_streak"] = int(hot_mask.sum())
        df = df[~hot_mask].copy()

    if df.empty:
        return df, stats

    # Main ranking aligned with sheet display logic:
    # model_probability first, then edge_probability, then odds_decimal.
    df = df.sort_values(
        ["model_probability", "edge_probability", "odds_decimal"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    if one_pick_per_player:
        df = df.drop_duplicates(subset=["player_name"], keep="first").reset_index(drop=True)

    stats["rows_after_player_dedup"] = int(len(df))

    if max_picks > 0:
        df = df.head(max_picks).copy()

    if df.empty:
        return df, stats

    df["recommended_flag"] = True
    df["recommendation_rank"] = np.arange(1, len(df) + 1)
    df["display_rank"] = df["recommendation_rank"]
    df["bet_status"] = df["bet_status"].replace({"": "pending", "nan": "pending"})

    # Friendly text label for sheet publication
    df["is_value_bet_label"] = np.where(df["is_value_bet"] == 1, "yes", "no")

    # Stable column order without dropping extra columns if present
    front_cols = [
        "bet_id",
        "run_date",
        "recommended_flag",
        "recommendation_rank",
        "display_rank",
        "bet_status",
        "result",
        "actual_stat_value",
        "settled_at",
        "date_match",
        "player_name",
        "team",
        "opponent",
        "bookmaker",
        "market",
        "stat",
        "threshold",
        "odds_decimal",
        "implied_probability",
        "model_probability",
        "edge_probability",
        "value_gap",
        "is_value_bet",
        "is_value_bet_label",
        "hard_exclude_hot_streak_pre",
        "ev_per_unit",
        "kelly_fraction",
    ]
    ordered_cols = [c for c in front_cols if c in df.columns] + [c for c in df.columns if c not in front_cols]
    df = df[ordered_cols].copy()

    return df, stats


# -----------------------------------------------------------------------------
# History
# -----------------------------------------------------------------------------

def append_history(master_path: Path, daily_df: pd.DataFrame) -> Dict[str, int]:
    master_path.parent.mkdir(parents=True, exist_ok=True)

    if master_path.exists():
        master_df = pd.read_csv(master_path, low_memory=False)
        rows_before = len(master_df)
    else:
        master_df = pd.DataFrame(columns=daily_df.columns)
        rows_before = 0

    existing_bet_ids = (
        set(master_df["bet_id"].astype(str).tolist())
        if (not master_df.empty and "bet_id" in master_df.columns)
        else set()
    )

    to_add_df = daily_df[~daily_df["bet_id"].astype(str).isin(existing_bet_ids)].copy()
    combined_df = pd.concat([master_df, to_add_df], ignore_index=True)
    combined_df.to_csv(master_path, index=False)

    return {
        "rows_before": int(rows_before),
        "rows_added": int(len(to_add_df)),
        "rows_after": int(len(combined_df)),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    daily_dir = output_dir / "history" / "daily_bets"
    history_dir = output_dir / "history"
    daily_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    candidates_df = load_candidates(input_csv)
    run_date = choose_run_date(candidates_df, args.run_date)

    daily_df, build_stats = build_daily_bets(
        candidates_df=candidates_df,
        run_date=run_date,
        max_picks=args.max_picks,
        min_odds=args.min_odds,
        override_min_model_proba=args.override_min_model_proba,
        value_threshold=args.value_threshold,
        one_pick_per_player=args.one_pick_per_player,
        disable_hot_streak_exclude=args.disable_hot_streak_exclude,
    )

    daily_csv_path = output_dir / "07_daily_bets.csv"
    daily_json_path = output_dir / "07_daily_bets.json"
    summary_json_path = output_dir / "07_daily_bets_summary.json"
    dated_daily_csv_path = daily_dir / f"{run_date}_07_daily_bets.csv"
    master_history_path = history_dir / "master_daily_bets_history.csv"

    daily_df.to_csv(daily_csv_path, index=False)
    daily_df.to_csv(dated_daily_csv_path, index=False)

    write_json(
        daily_json_path,
        {"run_date": run_date, "rows_count": int(len(daily_df)), "rows": daily_df.to_dict(orient="records")},
    )

    history_append = append_history(master_history_path, daily_df)

    preview_cols = [
        "recommendation_rank",
        "player_name",
        "team",
        "opponent",
        "odds_decimal",
        "implied_probability",
        "model_probability",
        "edge_probability",
        "value_gap",
        "is_value_bet_label",
    ]
    top_preview: List[Dict[str, Any]] = []
    if not daily_df.empty:
        existing_preview_cols = [c for c in preview_cols if c in daily_df.columns]
        top_preview = daily_df[existing_preview_cols].head(20).to_dict(orient="records")

    summary_payload = {
        "status": "ok",
        "run_date": run_date,
        "inputs": {
            "input_csv": str(input_csv),
        },
        "rules": {
            "max_picks": int(args.max_picks),
            "min_odds": float(args.min_odds),
            "override_min_model_proba": float(args.override_min_model_proba),
            "value_threshold": float(args.value_threshold),
            "one_pick_per_player": bool(args.one_pick_per_player),
            "disable_hot_streak_exclude": bool(args.disable_hot_streak_exclude),
        },
        "counts": {
            "candidate_rows_input": int(len(candidates_df[candidates_df["run_date"] == run_date])),
            "daily_bets_rows": int(len(daily_df)),
        },
        "build_stats": build_stats,
        "history_append": history_append,
        "top_preview": top_preview,
        "outputs": {
            "daily_csv": str(daily_csv_path),
            "daily_json": str(daily_json_path),
            "summary_json": str(summary_json_path),
            "dated_daily_csv": str(dated_daily_csv_path),
            "master_history_csv": str(master_history_path),
        },
    }
    write_json(summary_json_path, summary_payload)

    print("07_build_daily_bets.py")
    print(f"Input csv : {input_csv}")
    print(f"Run date : {run_date}")
    print(f"Daily bets rows : {len(daily_df)}")
    print(f"History rows added : {history_append['rows_added']}")
    print(f"History rows after : {history_append['rows_after']}")
    print("")
    print("=== RULES ===")
    print(f"- max_picks={args.max_picks}")
    print(f"- min_odds={args.min_odds}")
    print(f"- override_min_model_proba={args.override_min_model_proba}")
    print(f"- value_threshold={args.value_threshold}")
    print(f"- one_pick_per_player={args.one_pick_per_player}")
    print(f"- disable_hot_streak_exclude={args.disable_hot_streak_exclude}")
    print("")
    print("=== OUTPUTS ===")
    print(f"- {daily_csv_path}")
    print(f"- {daily_json_path}")
    print(f"- {summary_json_path}")
    print(f"- {dated_daily_csv_path}")
    print(f"- {master_history_path}")


if __name__ == "__main__":
    main()
