#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import unicodedata
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from difflib import SequenceMatcher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEFAULT_MODEL_CSV = DEFAULT_OUTPUTS_DIR / "predictions_upcoming_point_enrichi_calibre_v2.csv"

ODDS_CANDIDATE_PATHS = [
    DEFAULT_OUTPUTS_DIR / "normalized_points_odds.json",
    PROJECT_ROOT / "normalized_points_odds.json",
    PROJECT_ROOT / "data" / "external" / "normalized_points_odds.json",
    PROJECT_ROOT / "data" / "bookmaker" / "normalized_points_odds.json",
]

REQUIRED_MODEL_COLUMNS = [
    "date_match",
    "id_match",
    "id_joueur",
    "nom",
    "position",
    "team_player_match",
    "adversaire_match",
    "is_home_player",
    "proba_point_1p_calibree",
    "proba_point_1p_raw",
    "rank_proba_sur_date",
    "rank_proba_sur_match",
]

REQUIRED_ODDS_TOP_LEVEL_KEYS = [
    "bookmaker",
    "market",
    "stat",
    "threshold",
    "outcome_label",
    "rows",
]

REQUIRED_ODDS_ROW_COLUMNS = [
    "bookmaker",
    "market",
    "stat",
    "threshold",
    "outcome_label",
    "outcome_key",
    "event_url",
    "event_id",
    "event_slug",
    "home_team",
    "away_team",
    "team",
    "player_name",
    "odds_decimal",
    "implied_probability",
]

TEAM_CODE_TO_NAMES = {
    "ANA": ["anaheim ducks"],
    "BOS": ["boston bruins"],
    "BUF": ["buffalo sabres"],
    "CGY": ["calgary flames"],
    "CAR": ["carolina hurricanes"],
    "CHI": ["chicago blackhawks"],
    "COL": ["colorado avalanche"],
    "CBJ": ["columbus blue jackets"],
    "DAL": ["dallas stars"],
    "DET": ["detroit red wings"],
    "EDM": ["edmonton oilers"],
    "FLA": ["florida panthers"],
    "LAK": ["los angeles kings", "la kings"],
    "MIN": ["minnesota wild"],
    "MTL": ["montreal canadiens"],
    "NSH": ["nashville predators"],
    "NJD": ["new jersey devils"],
    "NYI": ["new york islanders"],
    "NYR": ["new york rangers"],
    "OTT": ["ottawa senators"],
    "PHI": ["philadelphia flyers"],
    "PIT": ["pittsburgh penguins"],
    "SJS": ["san jose sharks"],
    "SEA": ["seattle kraken"],
    "STL": ["st. louis blues", "st louis blues"],
    "TBL": ["tampa bay lightning"],
    "TOR": ["toronto maple leafs"],
    "UTA": ["utah mammoth", "utah hockey club"],
    "VAN": ["vancouver canucks"],
    "VGK": ["vegas golden knights"],
    "WSH": ["washington capitals"],
    "WPG": ["winnipeg jets"],
}

PLAYER_NAME_OVERRIDES = {
    "sebastian aho fin": "sebastian aho",
    "sebastian aho (fin)": "sebastian aho",
}

FUZZY_MIN_SCORE = 0.965


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def strip_accents(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_text(text: Any) -> str:
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return ""
    value = str(text).strip().lower()
    value = strip_accents(value)
    value = value.replace("’", "'").replace("`", "'")
    value = re.sub(r"\([^)]*\)", " ", value)
    value = value.replace(".", " ")
    value = re.sub(r"[^a-z0-9'\- ]+", " ", value)
    value = value.replace("-", " ")
    value = PLAYER_NAME_OVERRIDES.get(value, value)
    value = re.sub(r"\b(jr|sr)\b", " ", value)
    return normalize_spaces(value)


def normalize_player_join_name(text: Any) -> str:
    """
    Build a stable join key that aligns abbreviated model names such as
    "J. Hughes" with full bookmaker names such as "Jack Hughes".

    Rule used:
    - strip accents/punctuation via normalize_text
    - keep first initial
    - keep surname (last token)

    Examples:
    - "J. Hughes" -> "j hughes"
    - "Jack Hughes" -> "j hughes"
    - "Jean-Gabriel Pageau" -> "j pageau"
    - "J.T. Miller" -> "j miller"
    """
    value = normalize_text(text)
    if not value:
        return ""
    parts = [p for p in value.split() if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0][0]} {parts[-1]}"


def is_numeric_like_name(text: Any) -> bool:
    value = normalize_text(text)
    return bool(value) and value.isdigit()


def canonical_team_primary(team_code: Any) -> Optional[str]:
    key = str(team_code or "").strip().upper()
    aliases = TEAM_CODE_TO_NAMES.get(key)
    if not aliases:
        return None
    return aliases[0]


def canonical_team_aliases(team_code: Any) -> List[str]:
    key = str(team_code or "").strip().upper()
    aliases = TEAM_CODE_TO_NAMES.get(key, [])
    return [normalize_text(x) for x in aliases if normalize_text(x)]


def matchup_key_from_team_names(team_a: Any, team_b: Any) -> Tuple[str, str]:
    ordered = sorted([normalize_text(team_a), normalize_text(team_b)])
    return ordered[0], ordered[1]


def matchup_key_from_codes(team_code: Any, opp_code: Any) -> Tuple[str, str]:
    return matchup_key_from_team_names(
        canonical_team_primary(team_code),
        canonical_team_primary(opp_code),
    )


def find_odds_json_path(explicit_path: Optional[str]) -> Path:
    if explicit_path:
        path = Path(explicit_path)
        require_file(path)
        return path
    for path in ODDS_CANDIDATE_PATHS:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Impossible de trouver normalized_points_odds.json. "
        "Chemins testés : " + ", ".join(str(p) for p in ODDS_CANDIDATE_PATHS)
    )


def load_model_predictions(path: Path) -> pd.DataFrame:
    require_file(path)
    df = pd.read_csv(path, low_memory=False)
    missing = [c for c in REQUIRED_MODEL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans les prédictions modèle : {missing}")

    df = df.copy()
    df["date_match"] = pd.to_datetime(df["date_match"], errors="coerce")
    df["team_player_match"] = df["team_player_match"].astype(str).str.upper().str.strip()
    df["adversaire_match"] = df["adversaire_match"].astype(str).str.upper().str.strip()

    for col in ["id_match", "id_joueur", "rank_proba_sur_date", "rank_proba_sur_match"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["proba_point_1p_calibree", "proba_point_1p_raw", "is_home_player"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["player_name_normalized_raw"] = df["nom"].apply(normalize_text)
    df["player_name_normalized"] = df["nom"].apply(normalize_player_join_name)
    df["model_name_is_numeric"] = df["nom"].apply(is_numeric_like_name)
    df["team_aliases"] = df["team_player_match"].apply(canonical_team_aliases)
    df["team_primary_name"] = df["team_player_match"].apply(canonical_team_primary)
    df["opponent_primary_name"] = df["adversaire_match"].apply(canonical_team_primary)
    df["matchup_key"] = df.apply(
        lambda r: matchup_key_from_codes(r["team_player_match"], r["adversaire_match"]),
        axis=1,
    )
    return df


def load_odds_json(path: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
    require_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    missing_top = [k for k in REQUIRED_ODDS_TOP_LEVEL_KEYS if k not in payload]
    if missing_top:
        raise ValueError(f"Clés manquantes dans normalized_points_odds.json : {missing_top}")

    rows = payload["rows"]
    if not isinstance(rows, list):
        raise ValueError("normalized_points_odds.json['rows'] doit être une liste.")

    df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=REQUIRED_ODDS_ROW_COLUMNS)
    missing_cols = [c for c in REQUIRED_ODDS_ROW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans les rows bookmaker : {missing_cols}")

    df = df.copy()
    for col in ["threshold", "odds_decimal", "implied_probability"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["home_team", "away_team", "team", "player_name"]:
        df[col] = df[col].astype(str)

    df["player_name_normalized_raw"] = df["player_name"].apply(normalize_text)
    df["player_name_normalized_join"] = df["player_name"].apply(normalize_player_join_name)
    df["team_name_normalized_join"] = df["team"].apply(normalize_text)
    df["matchup_key"] = df.apply(
        lambda r: matchup_key_from_team_names(r["home_team"], r["away_team"]),
        axis=1,
    )
    return payload, df


def match_rows(model_df, odds_df, run_date, now=None):
    from henachel.matching import match_point_rows
    return match_point_rows(model_df, odds_df, run_date, TEAM_CODE_TO_NAMES, now=now)


def append_master_history(master_path, daily_df):
    from henachel.history import append_ledger
    return append_ledger(master_path, daily_df)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joindre les prédictions modèle POINT avec les cotes Unibet normalisées.")
    parser.add_argument("--model-csv", type=str, default=str(DEFAULT_MODEL_CSV))
    parser.add_argument("--odds-json", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUTS_DIR))
    parser.add_argument("--run-date", type=str, default=date.today().isoformat())
    parser.add_argument("--disable-fuzzy", action="store_true")
    return parser.parse_args()


def main() -> None:
    global FUZZY_MIN_SCORE

    args = parse_args()
    output_dir = Path(args.output_dir)
    history_dir = output_dir / "history"
    history_daily_dir = history_dir / "daily_candidates"
    ensure_dir(output_dir)
    ensure_dir(history_dir)
    ensure_dir(history_daily_dir)

    model_csv_path = Path(args.model_csv)
    odds_path = find_odds_json_path(args.odds_json)

    matched_csv_path = output_dir / "06_matched_point_edges.csv"
    matched_json_path = output_dir / "06_matched_point_edges.json"
    unmatched_model_path = output_dir / "06_unmatched_model_rows.csv"
    unmatched_bookmaker_path = output_dir / "06_unmatched_bookmaker_rows.csv"
    summary_path = output_dir / "06_matching_summary.json"
    dated_candidates_path = history_daily_dir / f"{args.run_date}_06_matched_point_edges.csv"
    master_history_path = history_dir / "master_candidates_history.csv"

    require_file(model_csv_path)

    print("06_match_model_to_unibet_odds.py")
    print(f"Input model predictions : {model_csv_path}")
    print(f"Input odds json         : {odds_path}")
    print(f"Run date                : {args.run_date}")

    model_df = load_model_predictions(model_csv_path)
    odds_payload, odds_df = load_odds_json(odds_path)

    if args.disable_fuzzy:
        FUZZY_MIN_SCORE = 1.1

    matched_df, unmatched_model_df, unmatched_bookmaker_df, match_summary = match_rows(model_df, odds_df, args.run_date)

    if not matched_df.empty:
        matched_df = matched_df.sort_values(
            ["ev_per_unit", "edge_probability", "model_probability"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    matched_df.to_csv(matched_csv_path, index=False)
    matched_df.to_csv(dated_candidates_path, index=False)
    unmatched_model_df.to_csv(unmatched_model_path, index=False)
    unmatched_bookmaker_df.to_csv(unmatched_bookmaker_path, index=False)

    write_json(
        matched_json_path,
        {"rows_count": int(len(matched_df)), "rows": matched_df.to_dict(orient="records")},
    )

    append_stats = append_master_history(master_history_path, matched_df)

    positive_ev_df = matched_df[matched_df["ev_per_unit"] > 0].copy() if not matched_df.empty else matched_df.copy()
    top_positive_ev_cols = [
        "bet_id",
        "player_name",
        "team",
        "opponent",
        "odds_decimal",
        "model_probability",
        "implied_probability",
        "edge_probability",
        "ev_per_unit",
        "kelly_fraction",
        "match_method",
    ]
    top_positive_ev = positive_ev_df.head(20)[top_positive_ev_cols].to_dict(orient="records") if not positive_ev_df.empty else []

    summary_payload = {
        "status": "ok" if len(matched_df) else ("no_odds" if odds_df.empty else "no_verified_matches"),
        "run_date": args.run_date,
        "market": "player_points",
        "threshold": 1,
        "target_bookmaker": odds_payload.get("bookmaker"),
        "target_date_from_model": (
            str(model_df["date_match"].dropna().dt.strftime("%Y-%m-%d").min())
            if model_df["date_match"].notna().any() else None
        ),
        "inputs": {
            "model_predictions_path": str(model_csv_path),
            "odds_json_path": str(odds_path),
        },
        "match_summary": match_summary,
        "positive_ev_rows_count": int(len(positive_ev_df)),
        "history_append": append_stats,
        "top_positive_ev_preview": top_positive_ev,
        "outputs": {
            "matched_csv": str(matched_csv_path),
            "matched_json": str(matched_json_path),
            "unmatched_model_csv": str(unmatched_model_path),
            "unmatched_bookmaker_csv": str(unmatched_bookmaker_path),
            "summary_json": str(summary_path),
            "dated_candidates_csv": str(dated_candidates_path),
            "master_history_csv": str(master_history_path),
        },
    }
    write_json(summary_path, summary_payload)

    print("")
    print("=== MATCHING SUMMARY ===")
    print(f"model rows            : {match_summary['model_rows_count']}")
    print(f"bookmaker rows        : {match_summary['bookmaker_rows_count']}")
    print(f"matched rows          : {match_summary['matched_rows_count']}")
    print(f"unmatched model rows  : {match_summary['unmatched_model_rows_count']}")
    print(f"unmatched bookmaker   : {match_summary['unmatched_bookmaker_rows_count']}")
    print(f"exact matches         : {match_summary['exact_match_count']}")
    print(f"fuzzy matches         : {match_summary['fuzzy_match_count']}")
    print("")
    print("=== HISTORY ===")
    print(f"dated snapshot        : {dated_candidates_path}")
    print(f"master history        : {master_history_path}")
    print("")
    print("=== OUTPUTS ===")
    print(f"- {matched_csv_path}")
    print(f"- {matched_json_path}")
    print(f"- {unmatched_model_path}")
    print(f"- {unmatched_bookmaker_path}")
    print(f"- {summary_path}")


if __name__ == "__main__":
    main()
