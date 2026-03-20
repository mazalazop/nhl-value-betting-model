#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model/06_match_model_to_unibet_odds.py

Objectif
--------
Joindre les prédictions modèle POINT 1+ du repo principal avec les cotes
bookmaker normalisées issues du scraper Unibet.

Entrées
-------
- outputs/predictions_upcoming_point_enrichi_calibre_v2.csv
- normalized_points_odds.json (emplacement configurable)

Sorties
-------
- outputs/06_matched_point_edges.csv
- outputs/06_matched_point_edges.json
- outputs/06_unmatched_model_rows.csv
- outputs/06_unmatched_bookmaker_rows.csv
- outputs/06_matching_summary.json

Principes
---------
- matching conservateur, d'abord exact sur joueur + équipe + matchup
- normalisation robuste des noms (accents, parenthèses, ponctuation)
- aucun fuzzy matching agressif par défaut
- fallback fuzzy autorisé seulement s'il est unique et très fort
- toutes les lignes non matchées sont exportées pour audit
"""

from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from difflib import SequenceMatcher


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MODEL_PREDICTIONS_PATH = OUTPUTS_DIR / "predictions_upcoming_point_enrichi_calibre_v2.csv"

ODDS_CANDIDATE_PATHS = [
    OUTPUTS_DIR / "normalized_points_odds.json",
    PROJECT_ROOT / "normalized_points_odds.json",
    PROJECT_ROOT / "data" / "external" / "normalized_points_odds.json",
    PROJECT_ROOT / "data" / "bookmaker" / "normalized_points_odds.json",
]

MATCHED_CSV_PATH = OUTPUTS_DIR / "06_matched_point_edges.csv"
MATCHED_JSON_PATH = OUTPUTS_DIR / "06_matched_point_edges.json"
UNMATCHED_MODEL_PATH = OUTPUTS_DIR / "06_unmatched_model_rows.csv"
UNMATCHED_BOOKMAKER_PATH = OUTPUTS_DIR / "06_unmatched_bookmaker_rows.csv"
SUMMARY_PATH = OUTPUTS_DIR / "06_matching_summary.json"

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
    # bookmaker-specific qualifiers
    "sebastian aho fin": "sebastian aho",
    "sebastian aho (fin)": "sebastian aho",
}

FUZZY_MIN_SCORE = 0.965


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")


def ensure_output_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


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
    value = normalize_spaces(value)
    return value


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
    a = normalize_text(team_a)
    b = normalize_text(team_b)
    ordered = sorted([a, b])
    return ordered[0], ordered[1]


def matchup_key_from_codes(team_code: Any, opp_code: Any) -> Tuple[str, str]:
    team = canonical_team_primary(team_code)
    opp = canonical_team_primary(opp_code)
    return matchup_key_from_team_names(team, opp)


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

    df["player_name_normalized"] = df["nom"].apply(normalize_text)
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

    df = pd.DataFrame(rows)
    missing_cols = [c for c in REQUIRED_ODDS_ROW_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans les rows bookmaker : {missing_cols}")

    df = df.copy()
    for col in ["threshold", "odds_decimal", "implied_probability"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["home_team", "away_team", "team", "player_name"]:
        df[col] = df[col].astype(str)

    df["player_name_normalized_join"] = df["player_name"].apply(normalize_text)
    df["team_name_normalized_join"] = df["team"].apply(normalize_text)
    df["home_team_normalized_join"] = df["home_team"].apply(normalize_text)
    df["away_team_normalized_join"] = df["away_team"].apply(normalize_text)
    df["matchup_key"] = df.apply(
        lambda r: matchup_key_from_team_names(r["home_team"], r["away_team"]),
        axis=1,
    )

    return payload, df


def exact_candidate_subset(model_row: pd.Series, odds_df: pd.DataFrame) -> pd.DataFrame:
    subset = odds_df[
        (odds_df["matchup_key"] == model_row["matchup_key"])
        & (odds_df["player_name_normalized_join"] == model_row["player_name_normalized"])
    ].copy()

    if subset.empty:
        return subset

    aliases = set(model_row["team_aliases"] or [])
    if aliases:
        subset = subset[subset["team_name_normalized_join"].isin(aliases)].copy()

    return subset


def fuzzy_candidate_subset(model_row: pd.Series, odds_df: pd.DataFrame) -> pd.DataFrame:
    if model_row["model_name_is_numeric"]:
        return odds_df.iloc[0:0].copy()

    subset = odds_df[
        (odds_df["matchup_key"] == model_row["matchup_key"])
        & (odds_df["team_name_normalized_join"].isin(set(model_row["team_aliases"] or [])))
    ].copy()

    if subset.empty:
        return subset

    model_name = model_row["player_name_normalized"]
    subset["fuzzy_score"] = subset["player_name_normalized_join"].apply(
        lambda x: SequenceMatcher(None, model_name, x).ratio()
    )
    subset = subset.sort_values(["fuzzy_score", "odds_decimal"], ascending=[False, True]).reset_index()
    subset = subset[subset["fuzzy_score"] >= FUZZY_MIN_SCORE].copy()

    if subset.empty:
        return subset

    best = float(subset.iloc[0]["fuzzy_score"])
    subset = subset[subset["fuzzy_score"] == best].copy()
    return subset


def kelly_fraction_decimal_odds(p: float, odds_decimal: float) -> float:
    if p is None or pd.isna(p) or odds_decimal is None or pd.isna(odds_decimal):
        return np.nan
    if p <= 0 or odds_decimal <= 1:
        return 0.0
    b = float(odds_decimal - 1.0)
    q = float(1.0 - p)
    k = (b * p - q) / b
    return float(max(k, 0.0))


def build_matched_row(model_row: pd.Series, odds_row: pd.Series, match_method: str, fuzzy_score: Optional[float]) -> Dict[str, Any]:
    model_proba = float(model_row["proba_point_1p_calibree"])
    raw_proba = float(model_row["proba_point_1p_raw"])
    implied = float(odds_row["implied_probability"])
    odds_decimal = float(odds_row["odds_decimal"])

    fair_odds = float(np.nan if model_proba <= 0 else 1.0 / model_proba)
    edge_probability = float(model_proba - implied)
    ev_per_unit = float(model_proba * (odds_decimal - 1.0) - (1.0 - model_proba))
    kelly = kelly_fraction_decimal_odds(model_proba, odds_decimal)

    return {
        "date_match": model_row["date_match"].strftime("%Y-%m-%d") if pd.notna(model_row["date_match"]) else None,
        "id_match": int(model_row["id_match"]) if pd.notna(model_row["id_match"]) else None,
        "id_joueur": int(model_row["id_joueur"]) if pd.notna(model_row["id_joueur"]) else None,
        "player_name_model": model_row["nom"],
        "player_name_model_normalized": model_row["player_name_normalized"],
        "player_name_bookmaker": odds_row["player_name"],
        "player_name_bookmaker_normalized": odds_row["player_name_normalized_join"],
        "position": model_row["position"],
        "team_code_model": model_row["team_player_match"],
        "opponent_code_model": model_row["adversaire_match"],
        "team_name_model": model_row["team_primary_name"],
        "opponent_name_model": model_row["opponent_primary_name"],
        "is_home_player_model": float(model_row["is_home_player"]) if pd.notna(model_row["is_home_player"]) else None,
        "rank_proba_sur_date": int(model_row["rank_proba_sur_date"]) if pd.notna(model_row["rank_proba_sur_date"]) else None,
        "rank_proba_sur_match": int(model_row["rank_proba_sur_match"]) if pd.notna(model_row["rank_proba_sur_match"]) else None,
        "model_variant": model_row.get("model_variant"),
        "calibration_method": model_row.get("calibration_method"),

        "bookmaker": odds_row["bookmaker"],
        "market": odds_row["market"],
        "stat": odds_row["stat"],
        "threshold": int(odds_row["threshold"]) if pd.notna(odds_row["threshold"]) else None,
        "outcome_label": odds_row["outcome_label"],
        "outcome_key": odds_row["outcome_key"],
        "event_url": odds_row["event_url"],
        "event_id": odds_row["event_id"],
        "event_slug": odds_row["event_slug"],
        "home_team": odds_row["home_team"],
        "away_team": odds_row["away_team"],
        "team_name_bookmaker": odds_row["team"],
        "odds_decimal": odds_decimal,
        "implied_probability": implied,

        "model_probability_raw": raw_proba,
        "model_probability_calibrated": model_proba,
        "fair_odds_model": fair_odds,
        "edge_probability": edge_probability,
        "edge_probability_pct_points": edge_probability * 100.0,
        "ev_per_unit": ev_per_unit,
        "kelly_fraction_full": kelly,
        "is_positive_ev": bool(ev_per_unit > 0),
        "match_method": match_method,
        "fuzzy_score": fuzzy_score,
    }


def match_rows(model_df: pd.DataFrame, odds_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    matched_rows: List[Dict[str, Any]] = []
    unmatched_model_rows: List[Dict[str, Any]] = []
    matched_odds_indices = set()

    exact_match_count = 0
    fuzzy_match_count = 0
    duplicate_candidate_count = 0

    for _, model_row in model_df.sort_values(
        ["date_match", "rank_proba_sur_match", "rank_proba_sur_date", "nom"],
        ascending=[True, True, True, True],
    ).iterrows():
        subset_exact = exact_candidate_subset(model_row, odds_df)
        subset_exact = subset_exact[~subset_exact.index.isin(matched_odds_indices)].copy()

        if len(subset_exact) == 1:
            odds_idx = subset_exact.index[0]
            matched_odds_indices.add(odds_idx)
            matched_rows.append(build_matched_row(model_row, subset_exact.loc[odds_idx], "exact", None))
            exact_match_count += 1
            continue

        if len(subset_exact) > 1:
            duplicate_candidate_count += 1
            unmatched_model_rows.append({
                "reason": "multiple_exact_candidates",
                "candidate_count": int(len(subset_exact)),
                "date_match": model_row["date_match"].strftime("%Y-%m-%d") if pd.notna(model_row["date_match"]) else None,
                "id_match": model_row["id_match"],
                "id_joueur": model_row["id_joueur"],
                "nom": model_row["nom"],
                "player_name_normalized": model_row["player_name_normalized"],
                "team_player_match": model_row["team_player_match"],
                "adversaire_match": model_row["adversaire_match"],
                "proba_point_1p_calibree": model_row["proba_point_1p_calibree"],
            })
            continue

        subset_fuzzy = fuzzy_candidate_subset(model_row, odds_df)
        subset_fuzzy = subset_fuzzy[~subset_fuzzy["index"].isin(matched_odds_indices)].copy() if "index" in subset_fuzzy.columns else subset_fuzzy

        if len(subset_fuzzy) == 1:
            odds_idx = int(subset_fuzzy.iloc[0]["index"])
            matched_odds_indices.add(odds_idx)
            matched_rows.append(build_matched_row(
                model_row,
                odds_df.loc[odds_idx],
                "fuzzy",
                float(subset_fuzzy.iloc[0]["fuzzy_score"]),
            ))
            fuzzy_match_count += 1
            continue

        unmatched_model_rows.append({
            "reason": "no_match",
            "candidate_count": int(len(subset_fuzzy)) if len(subset_fuzzy) > 0 else 0,
            "date_match": model_row["date_match"].strftime("%Y-%m-%d") if pd.notna(model_row["date_match"]) else None,
            "id_match": model_row["id_match"],
            "id_joueur": model_row["id_joueur"],
            "nom": model_row["nom"],
            "player_name_normalized": model_row["player_name_normalized"],
            "model_name_is_numeric": bool(model_row["model_name_is_numeric"]),
            "team_player_match": model_row["team_player_match"],
            "adversaire_match": model_row["adversaire_match"],
            "proba_point_1p_calibree": model_row["proba_point_1p_calibree"],
            "rank_proba_sur_match": model_row["rank_proba_sur_match"],
            "rank_proba_sur_date": model_row["rank_proba_sur_date"],
        })

    unmatched_odds_df = odds_df[~odds_df.index.isin(matched_odds_indices)].copy()

    matched_df = pd.DataFrame(matched_rows)
    unmatched_model_df = pd.DataFrame(unmatched_model_rows)
    unmatched_bookmaker_df = unmatched_odds_df[[
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
    ]].rename(columns={"team": "team_name_bookmaker", "player_name": "player_name_bookmaker"}).copy()

    summary = {
        "model_rows_count": int(len(model_df)),
        "bookmaker_rows_count": int(len(odds_df)),
        "matched_rows_count": int(len(matched_df)),
        "unmatched_model_rows_count": int(len(unmatched_model_df)),
        "unmatched_bookmaker_rows_count": int(len(unmatched_bookmaker_df)),
        "exact_match_count": int(exact_match_count),
        "fuzzy_match_count": int(fuzzy_match_count),
        "duplicate_candidate_count": int(duplicate_candidate_count),
        "match_rate_vs_bookmaker_rows": float(len(matched_df) / len(odds_df)) if len(odds_df) else 0.0,
        "match_rate_vs_model_rows": float(len(matched_df) / len(model_df)) if len(model_df) else 0.0,
    }

    return matched_df, unmatched_model_df, unmatched_bookmaker_df, summary


def save_outputs(
    matched_df: pd.DataFrame,
    unmatched_model_df: pd.DataFrame,
    unmatched_bookmaker_df: pd.DataFrame,
    summary_payload: Dict[str, Any],
) -> None:
    matched_df.to_csv(MATCHED_CSV_PATH, index=False)
    unmatched_model_df.to_csv(UNMATCHED_MODEL_PATH, index=False)
    unmatched_bookmaker_df.to_csv(UNMATCHED_BOOKMAKER_PATH, index=False)

    matched_json_rows = matched_df.to_dict(orient="records")
    write_json(
        MATCHED_JSON_PATH,
        {
            "rows_count": int(len(matched_json_rows)),
            "rows": matched_json_rows,
        },
    )
    write_json(SUMMARY_PATH, summary_payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Joindre les prédictions modèle POINT avec les cotes Unibet normalisées.")
    parser.add_argument(
        "--odds-json",
        type=str,
        default=None,
        help="Chemin explicite vers normalized_points_odds.json. Si absent, plusieurs chemins standards sont testés.",
    )
    parser.add_argument(
        "--disable-fuzzy",
        action="store_true",
        help="Désactiver totalement le fallback fuzzy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_output_dir()
    require_file(MODEL_PREDICTIONS_PATH)
    odds_path = find_odds_json_path(args.odds_json)

    print("06_match_model_to_unibet_odds.py")
    print(f"Input model predictions : {MODEL_PREDICTIONS_PATH}")
    print(f"Input odds json         : {odds_path}")

    model_df = load_model_predictions(MODEL_PREDICTIONS_PATH)
    odds_payload, odds_df = load_odds_json(odds_path)

    if args.disable_fuzzy:
        global FUZZY_MIN_SCORE
        FUZZY_MIN_SCORE = 1.1  # impossible threshold => fuzzy off

    matched_df, unmatched_model_df, unmatched_bookmaker_df, match_summary = match_rows(model_df, odds_df)

    matched_df = matched_df.sort_values(
        ["ev_per_unit", "edge_probability", "model_probability_calibrated"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    positive_ev_df = matched_df[matched_df["ev_per_unit"] > 0].copy()
    top_positive_ev = positive_ev_df.head(20)[[
        "player_name_model",
        "team_code_model",
        "event_slug",
        "odds_decimal",
        "model_probability_calibrated",
        "implied_probability",
        "edge_probability",
        "ev_per_unit",
        "kelly_fraction_full",
        "match_method",
    ]].to_dict(orient="records")

    summary_payload = {
        "status": "ok",
        "market": "player_points",
        "threshold": 1,
        "target_bookmaker": odds_payload.get("bookmaker"),
        "target_date_from_model": (
            str(model_df["date_match"].dropna().dt.strftime("%Y-%m-%d").min())
            if model_df["date_match"].notna().any() else None
        ),
        "inputs": {
            "model_predictions_path": str(MODEL_PREDICTIONS_PATH),
            "odds_json_path": str(odds_path),
        },
        "match_summary": match_summary,
        "positive_ev_rows_count": int(len(positive_ev_df)),
        "top_positive_ev_preview": top_positive_ev,
        "outputs": {
            "matched_csv": str(MATCHED_CSV_PATH),
            "matched_json": str(MATCHED_JSON_PATH),
            "unmatched_model_csv": str(UNMATCHED_MODEL_PATH),
            "unmatched_bookmaker_csv": str(UNMATCHED_BOOKMAKER_PATH),
            "summary_json": str(SUMMARY_PATH),
        },
        "notes": [
            "Matching conservateur : exact joueur + équipe + matchup d'abord",
            "Fallback fuzzy seulement si unique et très fort",
            "Les noms purement numériques côté modèle ne sont pas matchés automatiquement",
            "Les lignes non matchées sont exportées pour audit",
        ],
    }

    save_outputs(
        matched_df=matched_df,
        unmatched_model_df=unmatched_model_df,
        unmatched_bookmaker_df=unmatched_bookmaker_df,
        summary_payload=summary_payload,
    )

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
    print("=== OUTPUTS ===")
    print(f"- {MATCHED_CSV_PATH}")
    print(f"- {MATCHED_JSON_PATH}")
    print(f"- {UNMATCHED_MODEL_PATH}")
    print(f"- {UNMATCHED_BOOKMAKER_PATH}")
    print(f"- {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
