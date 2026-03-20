#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import gspread
from google.oauth2.service_account import Credentials


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_INPUT_CSV = DEFAULT_OUTPUTS_DIR / "07_daily_bets.csv"
DEFAULT_CREDS_JSON = PROJECT_ROOT / "secrets" / "henachel-service-account.json"
DEFAULT_DAILY_WS = "daily_picks"
DEFAULT_HISTORY_WS = "history_raw"

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

REQUIRED_INPUT_COLUMNS = [
    "bet_id",
    "run_date",
    "date_match",
    "player_name",
    "team",
    "opponent",
    "odds_decimal",
    "model_probability",
    "edge_probability",
    "ev_per_unit",
    "result",
    "bet_status",
]

DISPLAY_COLUMNS = [
    "run_date",
    "date_match",
    "player_name",
    "team",
    "opponent",
    "odds_decimal",
    "model_probability",
    "edge_probability",
    "ev_per_unit",
    "result",
    "bet_status",
]



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publier 07_daily_bets.csv dans Google Sheets.")
    parser.add_argument("--input-csv", type=str, default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--sheet-id", type=str, required=True)
    parser.add_argument("--credentials-json", type=str, default=str(DEFAULT_CREDS_JSON))
    parser.add_argument("--daily-worksheet", type=str, default=DEFAULT_DAILY_WS)
    parser.add_argument("--history-worksheet", type=str, default=DEFAULT_HISTORY_WS)
    return parser.parse_args()



def load_daily_bets(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input CSV introuvable : {path}")

    df = pd.read_csv(path, low_memory=False)

    legacy_rename = {
        "player_name_model": "player_name",
        "team_code_model": "team",
        "opponent_code_model": "opponent",
        "model_probability_calibrated": "model_probability",
        "kelly_fraction_full": "kelly_fraction",
    }
    for old, new in legacy_rename.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans 07_daily_bets.csv : {missing}")

    df = df.copy()
    for col in ["odds_decimal", "model_probability", "edge_probability", "ev_per_unit"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["run_date", "date_match", "result", "bet_status"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    return df



def authorize_gspread(credentials_json: Path) -> gspread.Client:
    if not credentials_json.exists():
        raise FileNotFoundError(f"Credentials JSON introuvable : {credentials_json}")
    creds = Credentials.from_service_account_file(str(credentials_json), scopes=SCOPES)
    return gspread.authorize(creds)



def get_or_create_worksheet(sh: gspread.Spreadsheet, title: str, rows: int = 2000, cols: int = 50):
    try:
        return sh.worksheet(title)
    except gspread.WorksheetNotFound:
        return sh.add_worksheet(title=title, rows=rows, cols=cols)



def df_to_sheet_values(df: pd.DataFrame) -> List[List[str]]:
    clean = df.copy().fillna("")
    for col in clean.columns:
        clean[col] = clean[col].astype(str)
    return [list(clean.columns)] + clean.values.tolist()



def write_replace(ws, df: pd.DataFrame) -> None:
    ws.clear()
    values = df_to_sheet_values(df)
    ws.update("A1", values)



def read_ws_as_df(ws) -> pd.DataFrame:
    records = ws.get_all_records()
    if not records:
        return pd.DataFrame()
    return pd.DataFrame(records)



def build_daily_display_df(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in DISPLAY_COLUMNS if c in df.columns]
    return df[available].copy()



def merge_history(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if existing_df.empty:
        merged = new_df.copy()
    else:
        for col in new_df.columns:
            if col not in existing_df.columns:
                existing_df[col] = ""
        for col in existing_df.columns:
            if col not in new_df.columns:
                new_df[col] = ""
        existing_df = existing_df[new_df.columns]
        merged = pd.concat([existing_df, new_df], ignore_index=True)

    if "bet_id" not in merged.columns:
        raise ValueError("La colonne bet_id est requise pour dédupliquer l'historique.")

    merged = merged.drop_duplicates(subset=["bet_id"], keep="first").reset_index(drop=True)
    return merged



def main() -> None:
    args = parse_args()

    input_csv = Path(args.input_csv)
    credentials_json = Path(args.credentials_json)

    daily_bets_df = load_daily_bets(input_csv)
    daily_display_df = build_daily_display_df(daily_bets_df)

    gc = authorize_gspread(credentials_json)
    sh = gc.open_by_key(args.sheet_id)

    daily_ws = get_or_create_worksheet(sh, args.daily_worksheet)
    history_ws = get_or_create_worksheet(sh, args.history_worksheet)

    existing_history_df = read_ws_as_df(history_ws)
    merged_history_df = merge_history(existing_history_df, daily_bets_df)

    write_replace(daily_ws, daily_display_df)
    write_replace(history_ws, merged_history_df)

    rows_before = int(len(existing_history_df))
    rows_after = int(len(merged_history_df))
    rows_added = int(rows_after - rows_before)

    print("08_publish_to_google_sheet.py")
    print(f"Input CSV              : {input_csv}")
    print(f"Sheet title            : {sh.title}")
    print(f"Worksheet daily        : {args.daily_worksheet}")
    print(f"Worksheet history      : {args.history_worksheet}")
    print(f"Daily rows written     : {len(daily_display_df)}")
    print(f"History rows before    : {rows_before}")
    print(f"History rows after     : {rows_after}")
    print(f"History rows added     : {rows_added}")


if __name__ == "__main__":
    main()
