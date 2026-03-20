#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_HISTORY_CSV = DEFAULT_OUTPUTS_DIR / "history" / "master_daily_bets_history.csv"
DEFAULT_SUMMARY_JSON = DEFAULT_OUTPUTS_DIR / "09_settlement_summary.json"
DEFAULT_SETTLED_CSV = DEFAULT_OUTPUTS_DIR / "09_settled_rows.csv"
DEFAULT_UNRESOLVED_CSV = DEFAULT_OUTPUTS_DIR / "09_unresolved_pending_rows.csv"

STATS_CANDIDATE_PATHS = [
    PROJECT_ROOT / "data" / "raw" / "stats.csv",
    Path("/content/drive/MyDrive/Henachel_NHL/stats.csv"),
    Path("/content/drive/MyDrive/Henachel_NHL_repo_backup/data/raw/stats.csv"),
]

REQUIRED_HISTORY_COLUMNS = [
    "bet_id",
    "run_date",
    "bet_status",
    "date_match",
    "id_match",
    "id_joueur",
    "stat",
    "threshold",
    "outcome_key",
]

REQUIRED_STATS_COLUMNS = [
    "id_joueur",
    "id_match",
    "date_match",
    "points",
]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def find_stats_path(explicit_path: Optional[str]) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        require_file(path)
        return path

    for path in STATS_CANDIDATE_PATHS:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Impossible de trouver stats.csv. Passe --stats-csv explicitement."
    )


def load_history(path: Path) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path, low_memory=False)

    missing = [c for c in REQUIRED_HISTORY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans l'historique : {missing}")

    df = df.copy()
    df["date_match"] = pd.to_datetime(df["date_match"], errors="coerce")
    df["run_date"] = pd.to_datetime(df["run_date"], errors="coerce")
    df["id_match"] = pd.to_numeric(df["id_match"], errors="coerce")
    df["id_joueur"] = pd.to_numeric(df["id_joueur"], errors="coerce")
    df["threshold"] = pd.to_numeric(df["threshold"], errors="coerce")
    df["bet_status"] = df["bet_status"].astype(str).str.strip().str.lower()

    return df


def load_stats(path: Path) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path, low_memory=False)

    missing = [c for c in REQUIRED_STATS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans stats.csv : {missing}")

    df = df.copy()
    df["date_match"] = pd.to_datetime(df["date_match"], errors="coerce")
    df["id_match"] = pd.to_numeric(df["id_match"], errors="coerce")
    df["id_joueur"] = pd.to_numeric(df["id_joueur"], errors="coerce")
    df["points"] = pd.to_numeric(df["points"], errors="coerce")

    df = df.sort_values(["date_match", "id_match", "id_joueur"]).drop_duplicates(
        subset=["id_match", "id_joueur"], keep="last"
    )

    return df


def settle_result(actual_value: float, threshold: float, outcome_key: str) -> str:
    if pd.isna(actual_value) or pd.isna(threshold):
        return ""

    key = str(outcome_key or "").strip().lower()
    t = float(threshold)
    v = float(actual_value)

    if key.endswith("_plus") or key.endswith("+"):
        return "win" if v >= t else "loss"

    if key.endswith("_minus"):
        return "win" if v < t else "loss"

    # fallback simple : comportement over/plus
    return "win" if v >= t else "loss"


def dataframe_to_sheet_values(df: pd.DataFrame) -> list[list[Any]]:
    safe_df = df.copy()
    for col in safe_df.columns:
        safe_df[col] = safe_df[col].map(
            lambda x: "" if pd.isna(x) else (x.strftime("%Y-%m-%d") if isinstance(x, (pd.Timestamp, datetime, date)) else x)
        )
    return [list(safe_df.columns)] + safe_df.values.tolist()


def rewrite_history_sheet(history_df: pd.DataFrame, sheet_id: str, credentials_json: str, worksheet_name: str) -> None:
    import gspread
    from google.oauth2.service_account import Credentials

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_file(credentials_json, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet_name)

    values = dataframe_to_sheet_values(history_df)
    ws.clear()
    ws.update(values=values, range_name="A1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Settle les picks pending à partir de stats.csv")
    parser.add_argument("--history-csv", type=str, default=str(DEFAULT_HISTORY_CSV))
    parser.add_argument("--stats-csv", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUTS_DIR))
    parser.add_argument("--run-date", type=str, default=str(date.today()))
    parser.add_argument("--sheet-id", type=str, default=None)
    parser.add_argument("--credentials-json", type=str, default=None)
    parser.add_argument("--history-worksheet", type=str, default="history_raw")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    history_path = Path(args.history_csv)
    stats_path = find_stats_path(args.stats_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_date = pd.to_datetime(args.run_date).normalize()
    settled_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("09_settle_previous_bets.py")
    print(f"History CSV            : {history_path}")
    print(f"Stats CSV              : {stats_path}")
    print(f"Run date               : {run_date.strftime('%Y-%m-%d')}")

    history_df = load_history(history_path)
    stats_df = load_stats(stats_path)

    pending_mask = (
        (history_df["bet_status"] == "pending")
        & history_df["date_match"].notna()
        & (history_df["date_match"] < run_date)
    )
    pending_df = history_df.loc[pending_mask].copy()

    print(f"Pending rows to inspect: {len(pending_df)}")

    if pending_df.empty:
        summary_payload = {
            "status": "ok",
            "run_date": run_date.strftime("%Y-%m-%d"),
            "history_csv": str(history_path),
            "stats_csv": str(stats_path),
            "pending_rows_before": 0,
            "settled_rows_count": 0,
            "unresolved_rows_count": 0,
            "sheet_updated": False,
            "note": "Aucune ligne pending antérieure à run_date.",
        }
        write_json(output_dir / DEFAULT_SUMMARY_JSON.name, summary_payload)
        pd.DataFrame().to_csv(output_dir / DEFAULT_SETTLED_CSV.name, index=False)
        pd.DataFrame().to_csv(output_dir / DEFAULT_UNRESOLVED_CSV.name, index=False)
        print("Rien à settle.")
        return

    pending_df = pending_df.reset_index().rename(columns={"index": "history_index"})
    merged = pending_df.merge(
        stats_df[["id_match", "id_joueur", "date_match", "points"]].rename(
            columns={"date_match": "stats_date_match", "points": "actual_points"}
        ),
        on=["id_match", "id_joueur"],
        how="left",
    )

    merged["actual_stat_value"] = merged["actual_points"]
    merged["settled_result"] = merged.apply(
        lambda r: settle_result(r["actual_stat_value"], r["threshold"], r["outcome_key"]),
        axis=1,
    )

    settled_df = merged[merged["actual_stat_value"].notna()].copy()
    unresolved_df = merged[merged["actual_stat_value"].isna()].copy()

    for _, row in settled_df.iterrows():
        idx = int(row["history_index"])
        history_df.at[idx, "actual_stat_value"] = float(row["actual_stat_value"])
        history_df.at[idx, "result"] = row["settled_result"]
        history_df.at[idx, "bet_status"] = "settled"
        history_df.at[idx, "settled_at"] = settled_at

    history_df.to_csv(history_path, index=False)

    settled_export = settled_df[
        [
            "bet_id",
            "run_date",
            "date_match",
            "id_match",
            "id_joueur",
            "player_name",
            "team",
            "opponent",
            "threshold",
            "outcome_key",
            "actual_stat_value",
            "settled_result",
        ]
    ].rename(columns={"settled_result": "result"}).copy()

    unresolved_export = unresolved_df[
        [
            "bet_id",
            "run_date",
            "date_match",
            "id_match",
            "id_joueur",
            "player_name",
            "team",
            "opponent",
            "threshold",
            "outcome_key",
        ]
    ].copy()

    settled_export.to_csv(output_dir / DEFAULT_SETTLED_CSV.name, index=False)
    unresolved_export.to_csv(output_dir / DEFAULT_UNRESOLVED_CSV.name, index=False)

    sheet_updated = False
    if args.sheet_id and args.credentials_json:
        rewrite_history_sheet(
            history_df=history_df,
            sheet_id=args.sheet_id,
            credentials_json=args.credentials_json,
            worksheet_name=args.history_worksheet,
        )
        sheet_updated = True

    summary_payload = {
        "status": "ok",
        "run_date": run_date.strftime("%Y-%m-%d"),
        "history_csv": str(history_path),
        "stats_csv": str(stats_path),
        "pending_rows_before": int(len(pending_df)),
        "settled_rows_count": int(len(settled_export)),
        "unresolved_rows_count": int(len(unresolved_export)),
        "sheet_updated": sheet_updated,
        "outputs": {
            "settled_csv": str(output_dir / DEFAULT_SETTLED_CSV.name),
            "unresolved_csv": str(output_dir / DEFAULT_UNRESOLVED_CSV.name),
            "summary_json": str(output_dir / DEFAULT_SUMMARY_JSON.name),
        },
    }
    write_json(output_dir / DEFAULT_SUMMARY_JSON.name, summary_payload)

    print(f"Settled rows            : {len(settled_export)}")
    print(f"Unresolved rows         : {len(unresolved_export)}")
    print(f"History updated         : {history_path}")
    print(f"History sheet updated   : {sheet_updated}")
    print(f"Summary JSON            : {output_dir / DEFAULT_SUMMARY_JSON.name}")


if __name__ == "__main__":
    main()
