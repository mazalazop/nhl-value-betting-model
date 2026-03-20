from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import gspread
import pandas as pd
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

    numeric_cols = [
        "odds_decimal",
        "implied_probability",
        "model_probability_raw",
        "model_probability",
        "edge_probability",
        "edge_probability_pct_points",
        "ev_per_unit",
        "kelly_fraction",
    ]
    for col in numeric_cols:
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


def read_ws_as_df(ws) -> pd.DataFrame:
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    header = values[0]
    data = values[1:]
    if not header:
        return pd.DataFrame()
    if not data:
        return pd.DataFrame(columns=header)
    return pd.DataFrame(data, columns=header)


def _format_pct(value) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value) * 100:.2f}%"


def _format_float(value, ndigits: int = 3) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{ndigits}f}"


def build_daily_display_df(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.copy()

    # Vue utilisateur: seulement les picks recommandés si la colonne existe
    if "recommended_flag" in daily.columns:
        daily = daily[daily["recommended_flag"].astype(str).str.lower().isin(["true", "1", "yes"]) | (daily["recommended_flag"] == True)]  # noqa: E712

    # Nom complet bookmaker pour l'affichage
    display_player_col = "player_name_bookmaker" if "player_name_bookmaker" in daily.columns else "player_name"

    # Tri demandé: proba modèle décroissante puis edge puis cote
    daily = daily.sort_values(
        by=["model_probability", "edge_probability", "odds_decimal"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)

    daily["display_rank"] = range(1, len(daily) + 1)

    edge_col = "edge_probability_pct_points" if "edge_probability_pct_points" in daily.columns else None

    out = pd.DataFrame({
        "display_rank": daily["display_rank"],
        "run_date": daily["run_date"],
        "date_match": daily["date_match"],
        "player_name": daily[display_player_col],
        "team": daily["team"],
        "opponent": daily["opponent"],
        "market": daily["market"] if "market" in daily.columns else "",
        "odds_decimal": daily["odds_decimal"].map(lambda x: _format_float(x, 2)),
        "implied_probability_pct": daily["implied_probability"].map(_format_pct) if "implied_probability" in daily.columns else "",
        "model_probability_pct": daily["model_probability"].map(_format_pct),
        "edge_probability_pct": (
            daily[edge_col].map(lambda x: f"{float(x):.2f}" if not pd.isna(x) else "")
            if edge_col else daily["edge_probability"].map(lambda x: f"{float(x) * 100:.2f}" if not pd.isna(x) else "")
        ),
        "ev_per_unit": daily["ev_per_unit"].map(lambda x: _format_float(x, 3)),
    })

    return out


def merge_history(existing_history_df: pd.DataFrame, daily_bets_df: pd.DataFrame) -> pd.DataFrame:
    daily = daily_bets_df.copy()

    # Harmoniser colonnes si l'onglet est vide ou partiel
    if existing_history_df is None or existing_history_df.empty:
        return daily

    existing = existing_history_df.copy()
    all_cols = list(dict.fromkeys(list(existing.columns) + list(daily.columns)))

    for col in all_cols:
        if col not in existing.columns:
            existing[col] = ""
        if col not in daily.columns:
            daily[col] = ""

    existing = existing[all_cols]
    daily = daily[all_cols]

    if "bet_id" not in existing.columns or "bet_id" not in daily.columns:
        return pd.concat([existing, daily], ignore_index=True)

    existing_ids = set(existing["bet_id"].astype(str))
    to_add = daily[~daily["bet_id"].astype(str).isin(existing_ids)].copy()

    merged = pd.concat([existing, to_add], ignore_index=True)
    return merged


def df_to_sheet_values(df: pd.DataFrame) -> List[List[str]]:
    clean = df.copy().fillna("")
    for col in clean.columns:
        clean[col] = clean[col].astype(str)
    return [list(clean.columns)] + clean.values.tolist()


def write_replace(ws, df: pd.DataFrame) -> None:
    ws.clear()
    values = df_to_sheet_values(df)
    ws.update("A1", values, value_input_option="USER_ENTERED")


def apply_basic_sheet_style(sh: gspread.Spreadsheet, ws, n_rows: int, n_cols: int) -> None:
    sheet_id = ws.id
    requests = [
        {
            "updateSheetProperties": {
                "properties": {
                    "sheetId": sheet_id,
                    "gridProperties": {"frozenRowCount": 1},
                },
                "fields": "gridProperties.frozenRowCount",
            }
        },
        {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": 0,
                    "endRowIndex": 1,
                    "startColumnIndex": 0,
                    "endColumnIndex": max(n_cols, 1),
                },
                "cell": {
                    "userEnteredFormat": {
                        "backgroundColor": {"red": 0.11, "green": 0.19, "blue": 0.45},
                        "textFormat": {
                            "foregroundColor": {"red": 1, "green": 1, "blue": 1},
                            "bold": True,
                        }
                    }
                },
                "fields": "userEnteredFormat(backgroundColor,textFormat.foregroundColor,textFormat.bold)",
            }
        },
        {
            "setBasicFilter": {
                "filter": {
                    "range": {
                        "sheetId": sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": max(n_rows, 1),
                        "startColumnIndex": 0,
                        "endColumnIndex": max(n_cols, 1),
                    }
                }
            }
        },
    ]

    for col_idx in range(max(n_cols, 1)):
        requests.append({
            "autoResizeDimensions": {
                "dimensions": {
                    "sheetId": sheet_id,
                    "dimension": "COLUMNS",
                    "startIndex": col_idx,
                    "endIndex": col_idx + 1,
                }
            }
        })

    sh.batch_update({"requests": requests})



def clear_conditional_format_rules(sh: gspread.Spreadsheet, ws) -> None:
    try:
        metadata = sh.fetch_sheet_metadata()
    except Exception:
        return

    sheets = metadata.get("sheets", [])
    target = None
    for sheet in sheets:
        props = sheet.get("properties", {})
        if props.get("sheetId") == ws.id:
            target = sheet
            break

    if not target:
        return

    rules = target.get("conditionalFormats", [])
    if not rules:
        return

    requests = []
    for idx in reversed(range(len(rules))):
        requests.append({
            "deleteConditionalFormatRule": {
                "sheetId": ws.id,
                "index": idx,
            }
        })
    sh.batch_update({"requests": requests})

def apply_history_conditional_formatting(sh: gspread.Spreadsheet, ws, history_df: pd.DataFrame) -> None:
    if history_df.empty:
        return

    sheet_id = ws.id
    headers = list(history_df.columns)
    requests = []

    def _col_index(name: str):
        return headers.index(name) if name in headers else None

    result_idx = _col_index("result")
    if result_idx is not None:
        # win -> vert
        requests.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "endRowIndex": max(len(history_df) + 1, 2),
                        "startColumnIndex": result_idx,
                        "endColumnIndex": result_idx + 1,
                    }],
                    "booleanRule": {
                        "condition": {
                            "type": "TEXT_EQ",
                            "values": [{"userEnteredValue": "win"}],
                        },
                        "format": {
                            "backgroundColor": {"red": 0.72, "green": 0.88, "blue": 0.80},
                            "textFormat": {"bold": True},
                        },
                    },
                },
                "index": 0,
            }
        })
        # loss -> rouge
        requests.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "endRowIndex": max(len(history_df) + 1, 2),
                        "startColumnIndex": result_idx,
                        "endColumnIndex": result_idx + 1,
                    }],
                    "booleanRule": {
                        "condition": {
                            "type": "TEXT_EQ",
                            "values": [{"userEnteredValue": "loss"}],
                        },
                        "format": {
                            "backgroundColor": {"red": 0.96, "green": 0.74, "blue": 0.74},
                            "textFormat": {"bold": True},
                        },
                    },
                },
                "index": 0,
            }
        })

    bet_status_idx = _col_index("bet_status")
    if bet_status_idx is not None:
        requests.append({
            "addConditionalFormatRule": {
                "rule": {
                    "ranges": [{
                        "sheetId": sheet_id,
                        "startRowIndex": 1,
                        "endRowIndex": max(len(history_df) + 1, 2),
                        "startColumnIndex": bet_status_idx,
                        "endColumnIndex": bet_status_idx + 1,
                    }],
                    "booleanRule": {
                        "condition": {
                            "type": "TEXT_EQ",
                            "values": [{"userEnteredValue": "pending"}],
                        },
                        "format": {
                            "backgroundColor": {"red": 0.93, "green": 0.93, "blue": 0.93},
                            "textFormat": {"bold": True},
                        },
                    },
                },
                "index": 0,
            }
        })

    if requests:
        sh.batch_update({"requests": requests})


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

    apply_basic_sheet_style(sh, daily_ws, len(daily_display_df) + 1, len(daily_display_df.columns))
    apply_basic_sheet_style(sh, history_ws, len(merged_history_df) + 1, len(merged_history_df.columns))
    clear_conditional_format_rules(sh, history_ws)
    apply_history_conditional_formatting(sh, history_ws, merged_history_df)

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
