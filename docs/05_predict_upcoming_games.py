#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model/05_predict_upcoming_games.py

Objectif
--------
Produire des prédictions POINT (joueur marque au moins 1 point) pour les matchs à venir,
à partir de sources amont du repo :

- data/raw/matchs.csv
- data/raw/joueurs.csv
- data/final/base_features_context_v2.csv
- data/raw/team_standings_daily.csv (optionnel mais recommandé)

Principes
---------
- aucune donnée du match futur n'est utilisée comme cible déguisée ;
- seules des features connues avant match sont construites ;
- l'univers des joueurs à prédire est construit d'abord depuis l'historique réel du pipeline ;
- le modèle POINT est réentraîné localement dans ce script ;
- calibration sigmoid temporellement propre sur une fenêtre récente antérieure à la date cible ;
- prise en compte du contexte standings / fin de saison quand team_standings_daily.csv existe.

Sorties
-------
- outputs/predictions_upcoming_point_enrichi_v2.csv
- outputs/predictions_upcoming_point_enrichi_calibre_v2.csv
- outputs/05_predict_upcoming_games_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_POINT_PATH = MODELS_DIR / "point_model_enrichi.joblib"
CALIBRATOR_POINT_PATH = MODELS_DIR / "point_calibrator_sigmoid.joblib"
MODEL_META_PATH = MODELS_DIR / "point_model_meta.json"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FINAL_DIR = DATA_DIR / "final"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

MATCHS_PATH = RAW_DIR / "matchs.csv"
JOUEURS_PATH = RAW_DIR / "joueurs.csv"
FEATURES_HISTORY_PATH = FINAL_DIR / "base_features_context_v2.csv"
TEAM_STANDINGS_PATH = RAW_DIR / "team_standings_daily.csv"

PRED_UPCOMING_RAW_PATH = OUTPUTS_DIR / "predictions_upcoming_point_enrichi_v2.csv"
PRED_UPCOMING_CAL_PATH = OUTPUTS_DIR / "predictions_upcoming_point_enrichi_calibre_v2.csv"
SUMMARY_PATH = OUTPUTS_DIR / "05_predict_upcoming_games_summary.json"

RANDOM_STATE = 42
DEFAULT_RECENT_LOOKBACK_DAYS = 45

DEFAULT_TEAM_GF = 2.8
DEFAULT_TEAM_GA = 2.8
DEFAULT_TEAM_WINRATE = 0.5
DEFAULT_LAST10_HIT_RATE = 0.45
DEFAULT_LAST20_HIT_RATE = 0.45
DEFAULT_SEASON_HIT_RATE = 0.45
DEFAULT_POINTS_PER_GAME = 0.55
DEFAULT_WILDCARD_DISTANCE = 0.0
DEFAULT_CONFERENCE_RANK = 8.5
DEFAULT_DIVISION_RANK = 4.5
DEFAULT_GAMES_REMAINING = 82.0

LONG_ABSENCE_DAYS = 10
RETURN_TOI_MIN = 15.0
RETURN_RATIO_TOI_MIN = 0.75
HISTORICAL_CURRENT_WEIGHT_DEFAULT = 0.60
HISTORICAL_CURRENT_WEIGHT_RETURN = 0.80

TEAM_CODE_ALIASES = {
    "ARI": "UTA",
    "PHX": "UTA",
    "PHO": "UTA",
}

TARGET_CANDIDATES = [
    "target_point_1p",
    "a_marque_un_point",
    "target_points_1p",
    "target_point",
    "label_point_1p",
    "y_point",
    "point_1p",
]

DATE_CANDIDATES = [
    "date_match",
    "date",
    "match_date",
    "game_date",
    "date_game",
]

META_OUTPUT_COLUMNS = [
    "date_match",
    "id_match",
    "saison",
    "id_joueur",
    "nom",
    "position",
    "team_player_match",
    "adversaire_match",
    "is_home_player",
]

# Cohérent avec 02_train_point_model.py
FEATURE_WHITELIST = [
    "is_home_player",
    "saison",
    "nb_matchs_avant_match",
    "jours_repos_raw",
    "is_premier_match_joueur",
    "jours_repos",
    "tirs_moy_5",
    "toi_moy_5",
    "pp_moy_5",
    "points_moy_5",
    "buts_moy_5",
    "passes_moy_5",
    "tirs_moy_10",
    "toi_moy_10",
    "points_moy_10",
    "buts_moy_10",
    "passes_moy_10",
    "nb_matchs_joues_10",
    "hist_ok_5",
    "hist_ok_10",
    "tirs_par_60_5",
    "points_par_60_5",
    "buts_par_60_5",
    "nb_matchs_vs_adv_avant",
    "points_vs_adv_5",
    "buts_vs_adv_5",
    "tirs_vs_adv_5",
    "points_vs_adv_shrunk",
    "buts_vs_adv_shrunk",
    "tirs_vs_adv_shrunk",
    "jours_absence_pre_match",
    "games_missed_proxy",
    "absence_longue_flag",
    "retour_episode",
    "return_from_absence_flag",
    "matchs_depuis_retour_avant_match",
    "toi_pre_absence_ref",
    "pp_pre_absence_ref",
    "toi_moy_retour_3_avant_match",
    "pp_moy_retour_3_avant_match",
    "ratio_toi_retour_vs_pre_absence",
    "ratio_pp_retour_vs_pre_absence",
    "return_stabilized_flag",
    "historical_current_weight",
    "historical_prev_weight",
    "point_hit_rate_last_5",
    "point_hit_rate_last_10",
    "point_hit_rate_last_20",
    "point_hit_rate_season_pre",
    "point_hit_rate_prev_season",
    "point_hit_rate_weighted_pre",
    "points_per_game_season_pre",
    "points_per_game_prev_season",
    "points_per_game_weighted_pre",
    "recent_vs_expected_gap",
    "current_point_streak_pre",
    "current_no_point_streak_pre",
    "max_point_streak_last_2_seasons_pre",
    "max_no_point_streak_last_2_seasons_pre",
    "count_5plus_point_streaks_last_2_seasons_pre",
    "is_home_team",
    "jours_repos_team",
    "team_back_to_back",
    "team_back_to_back_away",
    "consecutive_away_games",
    "team_winrate_5",
    "team_gf_moy_5",
    "team_ga_moy_5",
    "team_games_played_pre_approx",
    "games_played_team_pre",
    "games_remaining_team_pre",
    "team_points_pre",
    "conference_rank_pre",
    "division_rank_pre",
    "conference_cutoff_points_pre",
    "wildcard_distance_pre",
    "point_pctg_pre",
    "goal_differential_pre",
    "l10_points_pre",
    "late_season_flag",
    "playoff_pressure_simple",
]

REQUIRED_HISTORY_COLUMNS = [
    "id_joueur",
    "id_match",
    "date_match",
    "team_player_match",
    "adversaire_match",
    "is_home_player",
    "saison",
    "points",
    "buts",
    "passes",
    "tirs",
    "temps_de_glace",
    "temps_pp",
]

EXTRA_OUTPUT_COLUMNS = [
    "days_since_last_game",
    "point_hit_rate_last_10",
    "point_hit_rate_last_20",
    "point_hit_rate_weighted_pre",
    "recent_vs_expected_gap",
    "current_point_streak_pre",
    "current_no_point_streak_pre",
    "hard_exclude_hot_streak_pre",
    "games_remaining_team_pre",
    "team_points_pre",
    "wildcard_distance_pre",
    "playoff_pressure_simple",
]

DEFAULTS = {
    "jours_repos": 7.0,
    "jours_repos_team": 3.0,
    "team_back_to_back": 0.0,
    "team_back_to_back_away": 0.0,
    "consecutive_away_games": 0.0,
    "team_winrate_5": DEFAULT_TEAM_WINRATE,
    "team_gf_moy_5": DEFAULT_TEAM_GF,
    "team_ga_moy_5": DEFAULT_TEAM_GA,
    "team_games_played_pre_approx": 0.0,
    "points_vs_adv_shrunk": 0.0,
    "buts_vs_adv_shrunk": 0.0,
    "tirs_vs_adv_shrunk": 0.0,
    "jours_absence_pre_match": 3.0,
    "games_missed_proxy": 0.0,
    "matchs_depuis_retour_avant_match": 0.0,
    "return_from_absence_flag": 0.0,
    "return_stabilized_flag": 0.0,
    "historical_current_weight": HISTORICAL_CURRENT_WEIGHT_DEFAULT,
    "historical_prev_weight": 1.0 - HISTORICAL_CURRENT_WEIGHT_DEFAULT,
    "point_hit_rate_last_5": DEFAULT_LAST10_HIT_RATE,
    "point_hit_rate_last_10": DEFAULT_LAST10_HIT_RATE,
    "point_hit_rate_last_20": DEFAULT_LAST20_HIT_RATE,
    "point_hit_rate_season_pre": DEFAULT_SEASON_HIT_RATE,
    "point_hit_rate_prev_season": DEFAULT_SEASON_HIT_RATE,
    "point_hit_rate_weighted_pre": DEFAULT_SEASON_HIT_RATE,
    "points_per_game_season_pre": DEFAULT_POINTS_PER_GAME,
    "points_per_game_prev_season": DEFAULT_POINTS_PER_GAME,
    "points_per_game_weighted_pre": DEFAULT_POINTS_PER_GAME,
    "recent_vs_expected_gap": 0.0,
    "current_point_streak_pre": 0.0,
    "current_no_point_streak_pre": 0.0,
    "max_point_streak_last_2_seasons_pre": 0.0,
    "max_no_point_streak_last_2_seasons_pre": 0.0,
    "count_5plus_point_streaks_last_2_seasons_pre": 0.0,
    "hard_exclude_hot_streak_pre": 0.0,
    "games_played_team_pre": 0.0,
    "games_remaining_team_pre": DEFAULT_GAMES_REMAINING,
    "team_points_pre": np.nan,
    "conference_rank_pre": DEFAULT_CONFERENCE_RANK,
    "division_rank_pre": DEFAULT_DIVISION_RANK,
    "conference_cutoff_points_pre": np.nan,
    "wildcard_distance_pre": DEFAULT_WILDCARD_DISTANCE,
    "point_pctg_pre": np.nan,
    "goal_differential_pre": np.nan,
    "l10_points_pre": np.nan,
    "late_season_flag": 0.0,
    "playoff_pressure_simple": 0.0,
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")


def ensure_output_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def verifier_colonnes(df: pd.DataFrame, colonnes: List[str]) -> None:
    manquantes = [c for c in colonnes if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes : {manquantes}")


def normalize_boolean_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype(int)
    return out


def normalize_team_code(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    value = str(value).strip().upper()
    value = TEAM_CODE_ALIASES.get(value, value)
    return value or None


def normalize_position(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    value = str(value).strip().upper()
    mapping = {"LW": "L", "RW": "R", "LD": "D", "RD": "D"}
    return mapping.get(value, value)


def normalize_name(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def parse_mmss_to_minutes(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan
    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 2:
                mm, ss = parts
                return float(mm) + float(ss) / 60.0
            if len(parts) == 3:
                hh, mm, ss = parts
                return float(hh) * 60.0 + float(mm) + float(ss) / 60.0
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def safe_mean_last_n(series: pd.Series, n: int) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return 0.0
    return float(s.tail(n).mean())


def safe_ratio(num: float, den: float) -> float:
    if den is None or pd.isna(den) or den == 0:
        return np.nan
    if num is None or pd.isna(num):
        return np.nan
    return float(num / den)


def find_target_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            out = df.copy()
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out = out[out[col].notna()].copy()
            out[col] = out[col].astype(int)
            return out, col

    if "points" in df.columns:
        out = df.copy()
        out["points"] = pd.to_numeric(out["points"], errors="coerce")
        out = out[out["points"].notna()].copy()
        out["target_point_1p"] = (out["points"] >= 1).astype(int)
        return out, "target_point_1p"

    raise ValueError("Impossible de trouver une cible POINT exploitable dans la base historique.")


def find_date_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    for col in DATE_CANDIDATES:
        if col in df.columns:
            out = df.copy()
            out[col] = pd.to_datetime(out[col], errors="coerce")
            out = out[out[col].notna()].copy()
            out = out.sort_values([col, "id_match", "id_joueur"], na_position="last").reset_index(drop=True)
            return out, col
    raise ValueError("Impossible de trouver une colonne date exploitable.")


def compute_sample_weights(y: pd.Series) -> np.ndarray:
    y_array = y.astype(int).to_numpy()
    positives = int(y_array.sum())
    negatives = int(len(y_array) - positives)
    if positives == 0 or negatives == 0:
        return np.ones(len(y_array), dtype=float)
    pos_weight = negatives / positives
    return np.where(y_array == 1, pos_weight, 1.0).astype(float)


def to_numeric_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in cols:
        if col not in df.columns:
            out[col] = np.nan
            continue
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            series = series.astype(int)
        else:
            series = pd.to_numeric(series, errors="coerce")
        out[col] = series
    return out.replace([np.inf, -np.inf], np.nan).astype(float)


def keep_train_valid_features_only(
    X_train: pd.DataFrame,
    X_score: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    not_all_nan = ~X_train.isna().all(axis=0)
    variable = X_train.nunique(dropna=True) > 1
    kept_cols = X_train.columns[not_all_nan & variable].tolist()
    dropped_cols = [c for c in X_train.columns if c not in kept_cols]

    if len(kept_cols) < 5:
        raise ValueError(
            f"Trop peu de features exploitables après filtrage train-only : {len(kept_cols)}"
        )

    return X_train[kept_cols].copy(), X_score[kept_cols].copy(), kept_cols, dropped_cols


def normalize_season_code(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    if s.endswith(".0"):
        s = s[:-2]
    if len(s) < 8:
        return None
    return s[:8]


def previous_season_code(value: Any) -> Optional[str]:
    s = normalize_season_code(value)
    if s is None:
        return None
    try:
        start = int(s[:4]) - 1
        end = int(s[4:8]) - 1
        return f"{start}{end}"
    except ValueError:
        return None


def parse_season_start_year(value: Any) -> Optional[int]:
    s = normalize_season_code(value)
    if s is None:
        return None
    try:
        return int(s[:4])
    except ValueError:
        return None


def choose_target_date(matchs: pd.DataFrame, target_date_str: Optional[str]) -> pd.Timestamp:
    matchs = matchs.copy()
    matchs["date_match"] = pd.to_datetime(matchs["date_match"], errors="coerce")

    fut = matchs[matchs["status"].astype(str).str.upper() == "FUT"].copy()
    fut = fut[fut["date_match"].notna()].copy()

    if fut.empty:
        raise ValueError("Aucun match FUT trouvé dans matchs.csv.")

    if target_date_str is None:
        today = pd.Timestamp.today().normalize()
        fut_upcoming = fut[fut["date_match"].dt.normalize() >= today].copy()
        if fut_upcoming.empty:
            raise ValueError(
                "Aucun match FUT à partir d'aujourd'hui dans matchs.csv. "
                "Des lignes FUT anciennes existent peut-être encore dans la source."
            )
        return pd.Timestamp(fut_upcoming["date_match"].min().normalize())

    target_date = pd.to_datetime(target_date_str, errors="coerce")
    if pd.isna(target_date):
        raise ValueError(f"Date cible invalide : {target_date_str}")

    target_date = pd.Timestamp(target_date.normalize())

    if not ((fut["date_match"].dt.normalize() == target_date).any()):
        dates_disponibles = sorted(fut["date_match"].dt.strftime("%Y-%m-%d").unique().tolist())[:20]
        raise ValueError(
            f"Aucun match FUT trouvé pour la date cible {target_date.date()}. "
            f"Exemples de dates FUT disponibles : {dates_disponibles}"
        )

    return target_date


def load_matchs() -> pd.DataFrame:
    df = pd.read_csv(MATCHS_PATH, low_memory=False)
    verifier_colonnes(
        df,
        [
            "id_match",
            "date_match",
            "saison",
            "id_equipe_domicile",
            "id_equipe_exterieur",
            "buts_domicile",
            "buts_exterieur",
            "status",
        ],
    )
    df = df.copy()
    df["date_match"] = pd.to_datetime(df["date_match"], errors="coerce")
    for col in ["id_equipe_domicile", "id_equipe_exterieur", "status"]:
        df[col] = df[col].apply(normalize_team_code)
    return df.sort_values(["date_match", "id_match"]).reset_index(drop=True)


def load_joueurs() -> pd.DataFrame:
    df = pd.read_csv(JOUEURS_PATH, low_memory=False)
    verifier_colonnes(df, ["id_joueur", "nom", "position", "id_equipe"])
    df = df.copy()
    df["id_joueur"] = pd.to_numeric(df["id_joueur"], errors="coerce")
    df = df[df["id_joueur"].notna()].copy()
    df["id_joueur"] = df["id_joueur"].astype(int)
    df["nom"] = df["nom"].apply(normalize_name)
    df["position"] = df["position"].apply(normalize_position)
    df["id_equipe"] = df["id_equipe"].apply(normalize_team_code)
    df = df.drop_duplicates(subset=["id_joueur"], keep="first").reset_index(drop=True)
    return df


def load_history() -> Tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(FEATURES_HISTORY_PATH, low_memory=False)
    df = normalize_boolean_like_columns(df)
    df, target_col = find_target_column(df)
    df, date_col = find_date_column(df)
    verifier_colonnes(df, REQUIRED_HISTORY_COLUMNS)

    for col in [
        "id_joueur",
        "id_match",
        "is_home_player",
        "saison",
        "points",
        "buts",
        "passes",
        "tirs",
        "temps_de_glace",
        "temps_pp",
        "retour_episode",
        "toi_pre_absence_ref",
        "pp_pre_absence_ref",
    ]:
        if col in df.columns:
            if col in {"temps_de_glace", "temps_pp"}:
                df[col] = df[col].apply(parse_mmss_to_minutes)
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    if "season_source" not in df.columns:
        df["season_source"] = df["saison"]
    df["season_source"] = df["season_source"].apply(normalize_season_code)

    for col in ["team_player_match", "adversaire_match"]:
        df[col] = df[col].apply(normalize_team_code)

    if "nom" in df.columns:
        df["nom"] = df["nom"].apply(normalize_name)

    if "position" in df.columns:
        df["position"] = df["position"].apply(normalize_position)

    return df.reset_index(drop=True), target_col, date_col


def _compute_conference_cutoff_points(group: pd.DataFrame) -> float:
    conf_seq = pd.to_numeric(group.get("conference_sequence"), errors="coerce")
    points = pd.to_numeric(group.get("points"), errors="coerce")

    mask = conf_seq.notna() & points.notna() & (conf_seq == 8)
    if mask.any():
        return float(points.loc[mask].iloc[0])

    points_sorted = points.dropna().sort_values(ascending=False).tolist()
    if len(points_sorted) >= 8:
        return float(points_sorted[7])
    if len(points_sorted) > 0:
        return float(points_sorted[-1])
    return np.nan


def load_standings_optional() -> Tuple[Optional[pd.DataFrame], Dict[str, Any], Dict[str, pd.DataFrame]]:
    if not TEAM_STANDINGS_PATH.exists():
        return None, {
            "standings_loaded": False,
            "standings_file": str(TEAM_STANDINGS_PATH),
            "reason": "file_missing",
        }, {}

    standings = pd.read_csv(TEAM_STANDINGS_PATH, low_memory=False)
    required = [
        "date_snapshot",
        "team_abbrev",
        "conference_abbrev",
        "division_abbrev",
        "games_played",
        "games_remaining",
        "points",
        "conference_sequence",
        "division_sequence",
    ]
    verifier_colonnes(standings, required)

    standings = standings.copy()
    standings["date_snapshot"] = pd.to_datetime(standings["date_snapshot"], errors="coerce")
    standings = standings[standings["date_snapshot"].notna()].copy()
    standings["team_abbrev"] = standings["team_abbrev"].apply(normalize_team_code)
    standings["conference_abbrev"] = standings["conference_abbrev"].astype(str).str.upper().str.strip()
    standings["division_abbrev"] = standings["division_abbrev"].astype(str).str.upper().str.strip()

    numeric_cols = [
        "games_played",
        "games_remaining",
        "points",
        "conference_sequence",
        "division_sequence",
        "wildcard_sequence",
        "point_pctg",
        "goal_differential",
        "l10_points",
    ]
    for col in numeric_cols:
        if col in standings.columns:
            standings[col] = pd.to_numeric(standings[col], errors="coerce")

    standings = standings.sort_values(
        ["date_snapshot", "conference_abbrev", "conference_sequence", "team_abbrev"]
    ).reset_index(drop=True)

    conf_cutoff = (
        standings.dropna(subset=["conference_abbrev"])
        .groupby(["date_snapshot", "conference_abbrev"], dropna=False)
        .apply(_compute_conference_cutoff_points)
        .reset_index(name="conference_cutoff_points")
    )
    standings = standings.merge(
        conf_cutoff,
        on=["date_snapshot", "conference_abbrev"],
        how="left",
        validate="many_to_one",
    )
    standings["wildcard_distance"] = pd.to_numeric(standings["points"], errors="coerce") - pd.to_numeric(
        standings["conference_cutoff_points"], errors="coerce"
    )
    standings["standings_lookup_date"] = standings["date_snapshot"]
    standings = standings.sort_values(["team_abbrev", "standings_lookup_date"]).reset_index(drop=True)

    standings_by_team = {
        str(team): grp.sort_values("standings_lookup_date").reset_index(drop=True)
        for team, grp in standings.groupby("team_abbrev", sort=False)
    }

    summary = {
        "standings_loaded": True,
        "standings_file": str(TEAM_STANDINGS_PATH),
        "standings_rows": int(len(standings)),
        "standings_dates": int(standings["date_snapshot"].nunique()),
        "standings_teams": int(standings["team_abbrev"].nunique()),
        "standings_min_date": standings["date_snapshot"].min().strftime("%Y-%m-%d") if len(standings) else None,
        "standings_max_date": standings["date_snapshot"].max().strftime("%Y-%m-%d") if len(standings) else None,
    }
    return standings, summary, standings_by_team


def select_future_matches(matchs: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    fut = matchs[
        (matchs["status"].astype(str).str.upper() == "FUT")
        & (matchs["date_match"].dt.normalize() == target_date)
    ].copy()

    if fut.empty:
        raise ValueError(f"Aucun match FUT trouvé pour la date cible {target_date.date()}.")

    return fut.sort_values(["date_match", "id_match"]).reset_index(drop=True)


def build_schedule_team_rows(matchs: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["id_match", "date_match", "id_equipe_domicile", "id_equipe_exterieur"]
    verifier_colonnes(matchs, base_cols)
    home = matchs[base_cols].rename(
        columns={"id_equipe_domicile": "team_code", "id_equipe_exterieur": "opp_code"}
    )
    home = home[["id_match", "date_match", "team_code", "opp_code"]].copy()
    home["is_home_team"] = 1

    away = matchs[base_cols].rename(
        columns={"id_equipe_exterieur": "team_code", "id_equipe_domicile": "opp_code"}
    )
    away = away[["id_match", "date_match", "team_code", "opp_code"]].copy()
    away["is_home_team"] = 0

    out = pd.concat([home, away], ignore_index=True)
    out["team_code"] = out["team_code"].apply(normalize_team_code)
    out["opp_code"] = out["opp_code"].apply(normalize_team_code)
    out["date_match"] = pd.to_datetime(out["date_match"], errors="coerce")
    return out.sort_values(["team_code", "date_match", "id_match"]).reset_index(drop=True)


def build_completed_team_rows(matchs: pd.DataFrame) -> pd.DataFrame:
    completed = matchs[
        matchs["date_match"].notna()
        & matchs["buts_domicile"].notna()
        & matchs["buts_exterieur"].notna()
    ].copy()

    home = completed.rename(
        columns={
            "id_equipe_domicile": "team_code",
            "id_equipe_exterieur": "opp_code",
            "buts_domicile": "gf",
            "buts_exterieur": "ga",
        }
    )[["id_match", "date_match", "team_code", "opp_code", "gf", "ga"]].copy()
    home["is_home_team"] = 1

    away = completed.rename(
        columns={
            "id_equipe_exterieur": "team_code",
            "id_equipe_domicile": "opp_code",
            "buts_exterieur": "gf",
            "buts_domicile": "ga",
        }
    )[["id_match", "date_match", "team_code", "opp_code", "gf", "ga"]].copy()
    away["is_home_team"] = 0

    out = pd.concat([home, away], ignore_index=True)
    out["date_match"] = pd.to_datetime(out["date_match"], errors="coerce")
    out["team_code"] = out["team_code"].apply(normalize_team_code)
    out["opp_code"] = out["opp_code"].apply(normalize_team_code)
    out["gf"] = pd.to_numeric(out["gf"], errors="coerce")
    out["ga"] = pd.to_numeric(out["ga"], errors="coerce")
    return out.sort_values(["team_code", "date_match", "id_match"]).reset_index(drop=True)


def lookup_standings_pre_game(
    team_code: Optional[str],
    game_date: pd.Timestamp,
    standings_by_team: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    defaults = {
        "games_played_team_pre": np.nan,
        "games_remaining_team_pre": np.nan,
        "team_points_pre": np.nan,
        "conference_rank_pre": np.nan,
        "division_rank_pre": np.nan,
        "conference_cutoff_points_pre": np.nan,
        "wildcard_distance_pre": np.nan,
        "point_pctg_pre": np.nan,
        "goal_differential_pre": np.nan,
        "l10_points_pre": np.nan,
    }
    if team_code is None:
        return defaults

    team_df = standings_by_team.get(team_code)
    if team_df is None or team_df.empty:
        return defaults

    lookup_date = pd.Timestamp(game_date).normalize() - pd.Timedelta(days=1)
    team_df = team_df[team_df["standings_lookup_date"] <= lookup_date]
    if team_df.empty:
        return defaults

    row = team_df.iloc[-1]
    return {
        "games_played_team_pre": pd.to_numeric(row.get("games_played"), errors="coerce"),
        "games_remaining_team_pre": pd.to_numeric(row.get("games_remaining"), errors="coerce"),
        "team_points_pre": pd.to_numeric(row.get("points"), errors="coerce"),
        "conference_rank_pre": pd.to_numeric(row.get("conference_sequence"), errors="coerce"),
        "division_rank_pre": pd.to_numeric(row.get("division_sequence"), errors="coerce"),
        "conference_cutoff_points_pre": pd.to_numeric(row.get("conference_cutoff_points"), errors="coerce"),
        "wildcard_distance_pre": pd.to_numeric(row.get("wildcard_distance"), errors="coerce"),
        "point_pctg_pre": pd.to_numeric(row.get("point_pctg"), errors="coerce"),
        "goal_differential_pre": pd.to_numeric(row.get("goal_differential"), errors="coerce"),
        "l10_points_pre": pd.to_numeric(row.get("l10_points"), errors="coerce"),
    }


def compute_team_context_for_future_row(
    team_code: Optional[str],
    opp_code: Optional[str],
    is_home_team: int,
    game_date: pd.Timestamp,
    game_id: Any,
    schedule_team_rows: pd.DataFrame,
    completed_team_rows: pd.DataFrame,
    standings_by_team: Dict[str, pd.DataFrame],
) -> Dict[str, float]:
    schedule_hist = schedule_team_rows[
        (schedule_team_rows["team_code"] == team_code)
        & (
            (schedule_team_rows["date_match"] < game_date)
            | (
                (schedule_team_rows["date_match"] == game_date)
                & (schedule_team_rows["id_match"].astype(str) < str(game_id))
            )
        )
    ].sort_values(["date_match", "id_match"])

    if len(schedule_hist) == 0:
        jours_repos_team = DEFAULTS["jours_repos_team"]
        consecutive_prior_away = 0
    else:
        last_team_date = pd.Timestamp(schedule_hist["date_match"].max())
        jours_repos_team = float((game_date.normalize() - last_team_date.normalize()).days)
        jours_repos_team = float(min(max(jours_repos_team, 0.0), 14.0))

        consecutive_prior_away = 0
        for _, row in schedule_hist.sort_values(["date_match", "id_match"], ascending=[False, False]).iterrows():
            prev_is_home = int(pd.to_numeric(row["is_home_team"], errors="coerce"))
            if prev_is_home == 0:
                consecutive_prior_away += 1
            else:
                break

    team_back_to_back = 1.0 if jours_repos_team <= 1 else 0.0
    team_back_to_back_away = 1.0 if team_back_to_back == 1.0 and int(is_home_team) == 0 else 0.0
    consecutive_away_games = float(consecutive_prior_away + 1 if int(is_home_team) == 0 else 0)

    completed_hist = completed_team_rows[
        (completed_team_rows["team_code"] == team_code)
        & (completed_team_rows["date_match"] < game_date)
    ].sort_values(["date_match", "id_match"])

    team_games_played_pre_approx = float(len(completed_hist))

    if len(completed_hist) == 0:
        team_winrate_5 = DEFAULTS["team_winrate_5"]
        team_gf_moy_5 = DEFAULTS["team_gf_moy_5"]
        team_ga_moy_5 = DEFAULTS["team_ga_moy_5"]
    else:
        tail5 = completed_hist.tail(5).copy()
        tail5["team_win"] = (
            pd.to_numeric(tail5["gf"], errors="coerce") > pd.to_numeric(tail5["ga"], errors="coerce")
        ).astype(int)
        team_winrate_5 = float(tail5["team_win"].mean())
        team_gf_moy_5 = float(pd.to_numeric(tail5["gf"], errors="coerce").mean())
        team_ga_moy_5 = float(pd.to_numeric(tail5["ga"], errors="coerce").mean())

    standings_ctx = lookup_standings_pre_game(team_code=team_code, game_date=game_date, standings_by_team=standings_by_team)

    games_played_team_pre = standings_ctx["games_played_team_pre"]
    if pd.isna(games_played_team_pre):
        games_played_team_pre = team_games_played_pre_approx

    games_remaining_team_pre = standings_ctx["games_remaining_team_pre"]
    if pd.isna(games_remaining_team_pre):
        games_remaining_team_pre = max(0.0, 82.0 - team_games_played_pre_approx)

    wildcard_distance_pre = standings_ctx["wildcard_distance_pre"]
    if pd.isna(wildcard_distance_pre):
        wildcard_distance_pre = DEFAULT_WILDCARD_DISTANCE

    late_season_flag = float(1 if games_remaining_team_pre <= 20 else 0)
    playoff_pressure_simple = 0.0
    if late_season_flag == 1.0:
        if -4 <= float(wildcard_distance_pre) <= 4:
            playoff_pressure_simple = 1.0
        elif float(wildcard_distance_pre) >= 8 or float(wildcard_distance_pre) <= -8:
            playoff_pressure_simple = -1.0

    out = {
        "is_home_team": float(is_home_team),
        "jours_repos_team": jours_repos_team,
        "team_back_to_back": team_back_to_back,
        "team_back_to_back_away": team_back_to_back_away,
        "consecutive_away_games": consecutive_away_games,
        "team_winrate_5": team_winrate_5,
        "team_gf_moy_5": team_gf_moy_5,
        "team_ga_moy_5": team_ga_moy_5,
        "team_games_played_pre_approx": team_games_played_pre_approx,
        "games_played_team_pre": games_played_team_pre,
        "games_remaining_team_pre": games_remaining_team_pre,
        "team_points_pre": standings_ctx["team_points_pre"],
        "conference_rank_pre": standings_ctx["conference_rank_pre"],
        "division_rank_pre": standings_ctx["division_rank_pre"],
        "conference_cutoff_points_pre": standings_ctx["conference_cutoff_points_pre"],
        "wildcard_distance_pre": wildcard_distance_pre,
        "point_pctg_pre": standings_ctx["point_pctg_pre"],
        "goal_differential_pre": standings_ctx["goal_differential_pre"],
        "l10_points_pre": standings_ctx["l10_points_pre"],
        "late_season_flag": late_season_flag,
        "playoff_pressure_simple": playoff_pressure_simple,
    }

    for col, default_val in DEFAULTS.items():
        if col in out and (out[col] is None or pd.isna(out[col])):
            out[col] = default_val
    return out


def compute_last_two_seasons_streak_stats(hist_player: pd.DataFrame, target_season_code: Optional[str]) -> Tuple[int, int, int]:
    if hist_player.empty:
        return 0, 0, 0

    temp = hist_player.copy()
    temp["season_code"] = temp["season_source"].apply(normalize_season_code)

    target_start = parse_season_start_year(target_season_code)
    if target_start is not None:
        temp["season_start_year"] = temp["season_code"].apply(parse_season_start_year)
        temp = temp[temp["season_start_year"].notna() & (temp["season_start_year"] >= target_start - 1)].copy()

    hits = (pd.to_numeric(temp["points"], errors="coerce").fillna(0) >= 1).astype(int).tolist()
    if not hits:
        return 0, 0, 0

    max_hit = 0
    max_no = 0
    count_5plus = 0
    hit_run = 0
    no_run = 0
    for value in hits:
        if value == 1:
            hit_run += 1
            no_run = 0
        else:
            no_run += 1
            hit_run = 0
        if hit_run > max_hit:
            max_hit = hit_run
        if no_run > max_no:
            max_no = no_run
        if hit_run == 5:
            count_5plus += 1
    return int(max_hit), int(max_no), int(count_5plus)


def compute_current_streaks(hist_player: pd.DataFrame, target_season_code: Optional[str]) -> Tuple[int, int]:
    if hist_player.empty or target_season_code is None:
        return 0, 0

    temp = hist_player.copy()
    temp["season_code"] = temp["season_source"].apply(normalize_season_code)
    temp = temp[temp["season_code"] == target_season_code].copy()
    if temp.empty:
        return 0, 0

    hits = (pd.to_numeric(temp["points"], errors="coerce").fillna(0) >= 1).astype(int).tolist()

    current_point_streak = 0
    for v in reversed(hits):
        if v == 1:
            current_point_streak += 1
        else:
            break

    current_no_point_streak = 0
    for v in reversed(hits):
        if v == 0:
            current_no_point_streak += 1
        else:
            break

    return int(current_point_streak), int(current_no_point_streak)


def compute_player_features_for_future_row(
    hist_player: pd.DataFrame,
    player_id: int,
    game_date: pd.Timestamp,
    saison: Any,
    team_code: Optional[str],
    opp_code: Optional[str],
    is_home_player: int,
) -> Dict[str, Any]:
    hist_player = hist_player.sort_values(["date_match", "id_match"]).reset_index(drop=True).copy()
    hist_player["season_source"] = hist_player["season_source"].apply(normalize_season_code)
    target_season_code = normalize_season_code(saison)
    n_prev = int(len(hist_player))

    last_date = pd.Timestamp(hist_player["date_match"].iloc[-1]) if n_prev > 0 else None
    last_season_code = hist_player["season_source"].iloc[-1] if n_prev > 0 else None
    same_season_prev = bool(last_season_code == target_season_code) if (last_season_code and target_season_code) else False

    if last_date is None or not same_season_prev:
        jours_repos_raw = np.nan
        jours_repos = DEFAULTS["jours_repos"]
        jours_absence_pre_match = DEFAULTS["jours_absence_pre_match"]
    else:
        delta_days = float((game_date.normalize() - last_date.normalize()).days)
        jours_repos_raw = delta_days
        jours_repos = float(min(max(delta_days, 0.0), 14.0))
        jours_absence_pre_match = float(min(max(delta_days, 0.0), 90.0))

    games_missed_proxy = float(max(jours_absence_pre_match - 4.0, 0.0))
    absence_longue_flag = float(1.0 if same_season_prev and jours_absence_pre_match >= LONG_ABSENCE_DAYS else 0.0)
    is_premier_match_joueur = float(1 if n_prev == 0 else 0)

    shots = pd.to_numeric(hist_player["tirs"], errors="coerce")
    toi = pd.to_numeric(hist_player["temps_de_glace"], errors="coerce")
    pp = pd.to_numeric(hist_player["temps_pp"], errors="coerce")
    points = pd.to_numeric(hist_player["points"], errors="coerce")
    goals = pd.to_numeric(hist_player["buts"], errors="coerce")
    assists = pd.to_numeric(hist_player["passes"], errors="coerce")

    tirs_moy_5 = safe_mean_last_n(shots, 5) if n_prev > 0 else 0.0
    toi_moy_5 = safe_mean_last_n(toi, 5) if n_prev > 0 else 0.0
    pp_moy_5 = safe_mean_last_n(pp, 5) if n_prev > 0 else 0.0
    points_moy_5 = safe_mean_last_n(points, 5) if n_prev > 0 else 0.0
    buts_moy_5 = safe_mean_last_n(goals, 5) if n_prev > 0 else 0.0
    passes_moy_5 = safe_mean_last_n(assists, 5) if n_prev > 0 else 0.0

    tirs_moy_10 = safe_mean_last_n(shots, 10) if n_prev > 0 else 0.0
    toi_moy_10 = safe_mean_last_n(toi, 10) if n_prev > 0 else 0.0
    points_moy_10 = safe_mean_last_n(points, 10) if n_prev > 0 else 0.0
    buts_moy_10 = safe_mean_last_n(goals, 10) if n_prev > 0 else 0.0
    passes_moy_10 = safe_mean_last_n(assists, 10) if n_prev > 0 else 0.0

    nb_matchs_joues_10 = float(min(n_prev, 10))
    hist_ok_5 = float(1 if n_prev >= 5 else 0)
    hist_ok_10 = float(1 if n_prev >= 10 else 0)

    tirs_par_60_5 = safe_ratio(tirs_moy_5 * 60.0, toi_moy_5)
    points_par_60_5 = safe_ratio(points_moy_5 * 60.0, toi_moy_5)
    buts_par_60_5 = safe_ratio(buts_moy_5 * 60.0, toi_moy_5)

    point_hits = (points.fillna(0) >= 1).astype(int)
    point_hit_rate_last_5 = safe_mean_last_n(point_hits, 5) if n_prev > 0 else DEFAULT_LAST10_HIT_RATE
    point_hit_rate_last_10 = safe_mean_last_n(point_hits, 10) if n_prev > 0 else DEFAULT_LAST10_HIT_RATE
    point_hit_rate_last_20 = safe_mean_last_n(point_hits, 20) if n_prev > 0 else DEFAULT_LAST20_HIT_RATE

    current_season_hist = hist_player[hist_player["season_source"] == target_season_code].copy()
    season_games_before_match = len(current_season_hist)
    if season_games_before_match > 0:
        point_hit_rate_season_pre = float((pd.to_numeric(current_season_hist["points"], errors="coerce").fillna(0) >= 1).mean())
        points_per_game_season_pre = float(pd.to_numeric(current_season_hist["points"], errors="coerce").mean())
    else:
        point_hit_rate_season_pre = np.nan
        points_per_game_season_pre = np.nan

    prev_season_code = previous_season_code(target_season_code)
    prev_season_hist = hist_player[hist_player["season_source"] == prev_season_code].copy()
    if len(prev_season_hist) > 0:
        point_hit_rate_prev_season = float((pd.to_numeric(prev_season_hist["points"], errors="coerce").fillna(0) >= 1).mean())
        points_per_game_prev_season = float(pd.to_numeric(prev_season_hist["points"], errors="coerce").mean())
    else:
        point_hit_rate_prev_season = np.nan
        points_per_game_prev_season = np.nan

    hist_vs_opp = hist_player[hist_player["adversaire_match"] == opp_code].copy()
    nb_matchs_vs_adv_avant = float(len(hist_vs_opp))
    points_vs_adv_5 = safe_mean_last_n(pd.to_numeric(hist_vs_opp["points"], errors="coerce"), 5) if len(hist_vs_opp) > 0 else np.nan
    buts_vs_adv_5 = safe_mean_last_n(pd.to_numeric(hist_vs_opp["buts"], errors="coerce"), 5) if len(hist_vs_opp) > 0 else np.nan
    tirs_vs_adv_5 = safe_mean_last_n(pd.to_numeric(hist_vs_opp["tirs"], errors="coerce"), 5) if len(hist_vs_opp) > 0 else np.nan

    k_shrink = 3.0
    points_emp = float(points_moy_10 if pd.isna(points_vs_adv_5) else points_vs_adv_5)
    buts_emp = float(buts_moy_10 if pd.isna(buts_vs_adv_5) else buts_vs_adv_5)
    tirs_emp = float(tirs_moy_10 if pd.isna(tirs_vs_adv_5) else tirs_vs_adv_5)

    points_vs_adv_shrunk = float(
        (points_emp * nb_matchs_vs_adv_avant + points_moy_10 * k_shrink) / (nb_matchs_vs_adv_avant + k_shrink)
    )
    buts_vs_adv_shrunk = float(
        (buts_emp * nb_matchs_vs_adv_avant + buts_moy_10 * k_shrink) / (nb_matchs_vs_adv_avant + k_shrink)
    )
    tirs_vs_adv_shrunk = float(
        (tirs_emp * nb_matchs_vs_adv_avant + tirs_moy_10 * k_shrink) / (nb_matchs_vs_adv_avant + k_shrink)
    )

    last_retour_episode = 0.0
    if n_prev > 0 and "retour_episode" in hist_player.columns:
        last_val = pd.to_numeric(hist_player["retour_episode"].iloc[-1], errors="coerce")
        last_retour_episode = float(0.0 if pd.isna(last_val) else last_val)

    retour_episode = float(last_retour_episode + 1.0 if absence_longue_flag == 1.0 else last_retour_episode)

    if n_prev == 0 or absence_longue_flag == 1.0:
        matchs_depuis_retour_avant_match = 0.0
        episode_hist = hist_player.iloc[0:0].copy()
    else:
        if "retour_episode" in hist_player.columns:
            episode_hist = hist_player[
                pd.to_numeric(hist_player["retour_episode"], errors="coerce").fillna(0.0) == retour_episode
            ].copy()
        else:
            episode_hist = hist_player.copy()
        matchs_depuis_retour_avant_match = float(len(episode_hist))

    if absence_longue_flag == 1.0:
        toi_pre_absence_ref = float(toi_moy_10)
        pp_pre_absence_ref = float(pp_moy_5)
    else:
        toi_series_source = episode_hist["toi_pre_absence_ref"] if "toi_pre_absence_ref" in episode_hist.columns else pd.Series(dtype=float)
        pp_series_source = episode_hist["pp_pre_absence_ref"] if "pp_pre_absence_ref" in episode_hist.columns else pd.Series(dtype=float)
        toi_series = pd.to_numeric(toi_series_source, errors="coerce").dropna()
        pp_series = pd.to_numeric(pp_series_source, errors="coerce").dropna()
        toi_pre_absence_ref = float(toi_series.iloc[-1]) if len(toi_series) > 0 else np.nan
        pp_pre_absence_ref = float(pp_series.iloc[-1]) if len(pp_series) > 0 else np.nan

    toi_source = episode_hist["temps_de_glace"] if "temps_de_glace" in episode_hist.columns else pd.Series(dtype=float)
    pp_source = episode_hist["temps_pp"] if "temps_pp" in episode_hist.columns else pd.Series(dtype=float)
    toi_moy_retour_3_avant_match = safe_mean_last_n(pd.to_numeric(toi_source, errors="coerce"), 3) if len(episode_hist) > 0 else np.nan
    pp_moy_retour_3_avant_match = safe_mean_last_n(pd.to_numeric(pp_source, errors="coerce"), 3) if len(episode_hist) > 0 else np.nan

    ratio_toi_retour_vs_pre_absence = safe_ratio(toi_moy_retour_3_avant_match, toi_pre_absence_ref)
    ratio_pp_retour_vs_pre_absence = safe_ratio(pp_moy_retour_3_avant_match, pp_pre_absence_ref)

    return_from_absence_flag = float(1 if retour_episode > 0 else 0)
    return_stabilized_flag = float(
        (retour_episode > 0)
        and (matchs_depuis_retour_avant_match >= 3)
        and ((0.0 if pd.isna(toi_moy_retour_3_avant_match) else toi_moy_retour_3_avant_match) >= RETURN_TOI_MIN)
        and ((0.0 if pd.isna(ratio_toi_retour_vs_pre_absence) else ratio_toi_retour_vs_pre_absence) >= RETURN_RATIO_TOI_MIN)
    )

    historical_current_weight = float(
        HISTORICAL_CURRENT_WEIGHT_RETURN if return_stabilized_flag == 1.0 else HISTORICAL_CURRENT_WEIGHT_DEFAULT
    )
    historical_prev_weight = float(1.0 - historical_current_weight)

    if pd.notna(point_hit_rate_season_pre) and pd.notna(point_hit_rate_prev_season):
        point_hit_rate_weighted_pre = historical_current_weight * point_hit_rate_season_pre + historical_prev_weight * point_hit_rate_prev_season
    else:
        point_hit_rate_weighted_pre = point_hit_rate_season_pre if pd.notna(point_hit_rate_season_pre) else point_hit_rate_prev_season
    if pd.isna(point_hit_rate_weighted_pre):
        point_hit_rate_weighted_pre = DEFAULT_SEASON_HIT_RATE

    if pd.notna(points_per_game_season_pre) and pd.notna(points_per_game_prev_season):
        points_per_game_weighted_pre = historical_current_weight * points_per_game_season_pre + historical_prev_weight * points_per_game_prev_season
    else:
        points_per_game_weighted_pre = points_per_game_season_pre if pd.notna(points_per_game_season_pre) else points_per_game_prev_season
    if pd.isna(points_per_game_weighted_pre):
        points_per_game_weighted_pre = DEFAULT_POINTS_PER_GAME

    current_point_streak_pre, current_no_point_streak_pre = compute_current_streaks(
        hist_player=hist_player,
        target_season_code=target_season_code,
    )
    (
        max_point_streak_last_2_seasons_pre,
        max_no_point_streak_last_2_seasons_pre,
        count_5plus_point_streaks_last_2_seasons_pre,
    ) = compute_last_two_seasons_streak_stats(hist_player=hist_player, target_season_code=target_season_code)

    recent_combo = 0.6 * (point_hit_rate_last_10 if not pd.isna(point_hit_rate_last_10) else point_hit_rate_last_20) + 0.4 * (
        point_hit_rate_last_20 if not pd.isna(point_hit_rate_last_20) else point_hit_rate_last_10
    )
    if pd.isna(recent_combo):
        recent_combo = DEFAULT_LAST10_HIT_RATE

    recent_vs_expected_gap = float(point_hit_rate_weighted_pre - recent_combo)

    hard_exclude_hot_streak_pre = float(
        (current_point_streak_pre >= 5)
        and (count_5plus_point_streaks_last_2_seasons_pre < 2)
        and (current_point_streak_pre >= max_point_streak_last_2_seasons_pre)
    )

    row = {
        "id_joueur": int(player_id),
        "date_match": game_date,
        "saison": pd.to_numeric(saison, errors="coerce"),
        "team_player_match": team_code,
        "adversaire_match": opp_code,
        "is_home_player": float(is_home_player),
        "nb_matchs_avant_match": float(n_prev),
        "jours_repos_raw": jours_repos_raw,
        "is_premier_match_joueur": is_premier_match_joueur,
        "jours_repos": jours_repos,
        "tirs_moy_5": tirs_moy_5,
        "toi_moy_5": toi_moy_5,
        "pp_moy_5": pp_moy_5,
        "points_moy_5": points_moy_5,
        "buts_moy_5": buts_moy_5,
        "passes_moy_5": passes_moy_5,
        "tirs_moy_10": tirs_moy_10,
        "toi_moy_10": toi_moy_10,
        "points_moy_10": points_moy_10,
        "buts_moy_10": buts_moy_10,
        "passes_moy_10": passes_moy_10,
        "nb_matchs_joues_10": nb_matchs_joues_10,
        "hist_ok_5": hist_ok_5,
        "hist_ok_10": hist_ok_10,
        "tirs_par_60_5": tirs_par_60_5,
        "points_par_60_5": points_par_60_5,
        "buts_par_60_5": buts_par_60_5,
        "nb_matchs_vs_adv_avant": nb_matchs_vs_adv_avant,
        "points_vs_adv_5": points_moy_10 if pd.isna(points_vs_adv_5) else float(points_vs_adv_5),
        "buts_vs_adv_5": buts_moy_10 if pd.isna(buts_vs_adv_5) else float(buts_vs_adv_5),
        "tirs_vs_adv_5": tirs_moy_10 if pd.isna(tirs_vs_adv_5) else float(tirs_vs_adv_5),
        "points_vs_adv_shrunk": points_vs_adv_shrunk,
        "buts_vs_adv_shrunk": buts_vs_adv_shrunk,
        "tirs_vs_adv_shrunk": tirs_vs_adv_shrunk,
        "jours_absence_pre_match": jours_absence_pre_match,
        "games_missed_proxy": games_missed_proxy,
        "absence_longue_flag": absence_longue_flag,
        "retour_episode": retour_episode,
        "return_from_absence_flag": return_from_absence_flag,
        "matchs_depuis_retour_avant_match": matchs_depuis_retour_avant_match,
        "toi_pre_absence_ref": toi_pre_absence_ref,
        "pp_pre_absence_ref": pp_pre_absence_ref,
        "toi_moy_retour_3_avant_match": toi_moy_retour_3_avant_match,
        "pp_moy_retour_3_avant_match": pp_moy_retour_3_avant_match,
        "ratio_toi_retour_vs_pre_absence": ratio_toi_retour_vs_pre_absence,
        "ratio_pp_retour_vs_pre_absence": ratio_pp_retour_vs_pre_absence,
        "return_stabilized_flag": return_stabilized_flag,
        "historical_current_weight": historical_current_weight,
        "historical_prev_weight": historical_prev_weight,
        "point_hit_rate_last_5": point_hit_rate_last_5,
        "point_hit_rate_last_10": point_hit_rate_last_10,
        "point_hit_rate_last_20": point_hit_rate_last_20,
        "point_hit_rate_season_pre": point_hit_rate_season_pre,
        "point_hit_rate_prev_season": point_hit_rate_prev_season,
        "point_hit_rate_weighted_pre": point_hit_rate_weighted_pre,
        "points_per_game_season_pre": points_per_game_season_pre,
        "points_per_game_prev_season": points_per_game_prev_season,
        "points_per_game_weighted_pre": points_per_game_weighted_pre,
        "recent_vs_expected_gap": recent_vs_expected_gap,
        "current_point_streak_pre": float(current_point_streak_pre),
        "current_no_point_streak_pre": float(current_no_point_streak_pre),
        "max_point_streak_last_2_seasons_pre": float(max_point_streak_last_2_seasons_pre),
        "max_no_point_streak_last_2_seasons_pre": float(max_no_point_streak_last_2_seasons_pre),
        "count_5plus_point_streaks_last_2_seasons_pre": float(count_5plus_point_streaks_last_2_seasons_pre),
        "hard_exclude_hot_streak_pre": hard_exclude_hot_streak_pre,
    }

    for col, default_val in DEFAULTS.items():
        if col in row and (row[col] is None or pd.isna(row[col])):
            row[col] = default_val
    return row


def build_recent_player_pool(
    history: pd.DataFrame,
    joueurs_lookup: pd.DataFrame,
    target_date: pd.Timestamp,
    slate_teams: List[str],
    include_goalies: bool = False,
    recent_lookback_days: int = DEFAULT_RECENT_LOOKBACK_DAYS,
) -> pd.DataFrame:
    hist = history[history["date_match"] < target_date].copy()
    if hist.empty:
        raise ValueError("Aucun historique disponible avant la date cible pour construire l'univers futur.")

    hist = hist.sort_values(["date_match", "id_match", "id_joueur"]).reset_index(drop=True)
    latest = hist.groupby("id_joueur", as_index=False).tail(1).copy()
    latest["team_player_match"] = latest["team_player_match"].apply(normalize_team_code)
    latest = latest[latest["team_player_match"].isin(slate_teams)].copy()

    latest["days_since_last_game"] = (
        target_date.normalize() - pd.to_datetime(latest["date_match"], errors="coerce").dt.normalize()
    ).dt.days

    latest = latest[
        latest["days_since_last_game"].notna()
        & (latest["days_since_last_game"] >= 0)
        & (latest["days_since_last_game"] <= recent_lookback_days)
    ].copy()

    joueurs_lookup = joueurs_lookup.copy().rename(
        columns={"nom": "nom_lookup", "position": "position_lookup", "id_equipe": "id_equipe_lookup"}
    )
    latest = latest.merge(
        joueurs_lookup[["id_joueur", "nom_lookup", "position_lookup", "id_equipe_lookup"]],
        on="id_joueur",
        how="left",
        validate="one_to_one",
    )

    latest["nom"] = latest.get("nom").apply(normalize_name) if "nom" in latest.columns else None
    latest["position"] = latest.get("position").apply(normalize_position) if "position" in latest.columns else None
    latest["nom"] = latest["nom"].fillna(latest["nom_lookup"])
    latest["position"] = latest["position"].fillna(latest["position_lookup"])

    if not include_goalies:
        latest = latest[latest["position"] != "G"].copy()

    latest["nom"] = latest["nom"].fillna(latest["id_joueur"].astype(str))
    latest["position"] = latest["position"].fillna("UNK")

    latest = latest.sort_values(
        ["team_player_match", "days_since_last_game", "date_match", "id_match", "id_joueur"],
        ascending=[True, True, False, False, True],
    ).reset_index(drop=True)

    if latest.empty:
        raise ValueError(
            "Aucun joueur candidat après filtrage par équipe / récence. "
            "Élargis éventuellement --recent-lookback-days."
        )
    return latest


def build_upcoming_universe(
    future_matches: pd.DataFrame,
    history: pd.DataFrame,
    matchs_all: pd.DataFrame,
    player_pool: pd.DataFrame,
    standings_by_team: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    schedule_team_rows = build_schedule_team_rows(matchs_all)
    completed_team_rows = build_completed_team_rows(matchs_all)

    history_by_player: Dict[int, pd.DataFrame] = {
        int(pid): grp.sort_values(["date_match", "id_match"]).reset_index(drop=True)
        for pid, grp in history.groupby("id_joueur")
    }

    rows: List[Dict[str, Any]] = []

    for _, match in future_matches.iterrows():
        game_id = match["id_match"]
        game_date = pd.Timestamp(match["date_match"])
        saison = match["saison"]
        home_team = normalize_team_code(match["id_equipe_domicile"])
        away_team = normalize_team_code(match["id_equipe_exterieur"])

        for team_code, opp_code, is_home in [(home_team, away_team, 1), (away_team, home_team, 0)]:
            roster_team = player_pool[player_pool["team_player_match"] == team_code].copy()

            for _, player in roster_team.iterrows():
                player_id = int(player["id_joueur"])
                hist_player = history_by_player.get(player_id, pd.DataFrame(columns=history.columns))

                base_row = compute_player_features_for_future_row(
                    hist_player=hist_player,
                    player_id=player_id,
                    game_date=game_date,
                    saison=saison,
                    team_code=team_code,
                    opp_code=opp_code,
                    is_home_player=is_home,
                )

                team_ctx = compute_team_context_for_future_row(
                    team_code=team_code,
                    opp_code=opp_code,
                    is_home_team=is_home,
                    game_date=game_date,
                    game_id=game_id,
                    schedule_team_rows=schedule_team_rows,
                    completed_team_rows=completed_team_rows,
                    standings_by_team=standings_by_team,
                )

                out_row = {
                    "date_match": game_date,
                    "id_match": game_id,
                    "saison": pd.to_numeric(saison, errors="coerce"),
                    "id_joueur": player_id,
                    "nom": player["nom"],
                    "position": player["position"],
                    "team_player_match": team_code,
                    "adversaire_match": opp_code,
                    "is_home_player": float(is_home),
                    "days_since_last_game": pd.to_numeric(player.get("days_since_last_game"), errors="coerce"),
                }
                out_row.update(base_row)
                out_row.update(team_ctx)
                rows.append(out_row)

    if not rows:
        raise ValueError("Aucune ligne upcoming construite. Vérifie le player pool et les matchs FUT.")

    df = pd.DataFrame(rows)
    for col in FEATURE_WHITELIST:
        if col not in df.columns:
            df[col] = np.nan
    return df.sort_values(["date_match", "id_match", "team_player_match", "nom"]).reset_index(drop=True)


def build_temporal_fit_and_calib_splits(
    history: pd.DataFrame,
    date_col: str,
    target_date: pd.Timestamp,
) -> Dict[str, Any]:
    history = history[history[date_col] < target_date].copy()
    history = history.sort_values([date_col, "id_match", "id_joueur"]).reset_index(drop=True)

    unique_dates = sorted(history[date_col].dt.strftime("%Y-%m-%d").unique().tolist())
    if len(unique_dates) < 20:
        raise ValueError(
            "Pas assez de dates historiques avant la date cible pour un fit/calibration propre "
            f"(trouvé : {len(unique_dates)})."
        )

    fit_end = max(1, int(len(unique_dates) * 0.85))
    if fit_end >= len(unique_dates):
        fit_end = len(unique_dates) - 1

    fit_dates = set(unique_dates[:fit_end])
    calib_dates = set(unique_dates[fit_end:])

    fit_df = history[history[date_col].dt.strftime("%Y-%m-%d").isin(fit_dates)].copy()
    calib_df = history[history[date_col].dt.strftime("%Y-%m-%d").isin(calib_dates)].copy()

    if fit_df.empty or calib_df.empty:
        raise ValueError("Split fit/calibration invalide avant la date cible.")

    return {
        "fit": fit_df,
        "calib": calib_df,
        "meta": {
            "fit_start": min(fit_dates),
            "fit_end": max(fit_dates),
            "calib_start": min(calib_dates),
            "calib_end": max(calib_dates),
        },
    }


def fit_point_model_and_calibrator(
    history: pd.DataFrame,
    target_col: str,
    date_col: str,
    target_date: pd.Timestamp,
) -> Tuple[HistGradientBoostingClassifier, LogisticRegression, Dict[str, Any]]:
    splits = build_temporal_fit_and_calib_splits(history=history, date_col=date_col, target_date=target_date)

    fit_df = splits["fit"]
    calib_df = splits["calib"]
    split_meta = splits["meta"]

    X_fit = to_numeric_frame(fit_df, FEATURE_WHITELIST)
    X_calib = to_numeric_frame(calib_df, FEATURE_WHITELIST)
    X_fit, X_calib, kept_cols, dropped_cols = keep_train_valid_features_only(X_fit, X_calib)

    y_fit = fit_df[target_col].astype(int)
    y_calib = calib_df[target_col].astype(int)

    if y_fit.nunique() < 2:
        raise ValueError("Le jeu fit ne contient qu'une seule classe.")

    model = HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=300,
        max_depth=6,
        min_samples_leaf=50,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )
    model.fit(X_fit, y_fit, sample_weight=compute_sample_weights(y_fit))

    calib_raw_proba = model.predict_proba(X_calib)[:, 1]
    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        C=1e6,
        random_state=RANDOM_STATE,
    )
    calibrator.fit(calib_raw_proba.reshape(-1, 1), y_calib)

    summary = {
        "fit_rows": int(len(fit_df)),
        "calib_rows": int(len(calib_df)),
        "fit_positive_rate": float(y_fit.mean()),
        "calib_positive_rate": float(y_calib.mean()),
        "fit_calibration_split": split_meta,
        "feature_cols_kept": kept_cols,
        "feature_cols_dropped_train_only": dropped_cols,
        "calibration_method": "sigmoid",
    }

    # Persist model + calibrator for reuse
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_POINT_PATH)
    joblib.dump(calibrator, CALIBRATOR_POINT_PATH)
    write_json(MODEL_META_PATH, {
        "target_date": str(target_date.date()),
        "feature_cols_kept": kept_cols,
        "fit_rows": int(len(fit_df)),
        "calib_rows": int(len(calib_df)),
        "calibration_method": "sigmoid",
        "fit_end": split_meta.get("fit_end"),
        "calib_end": split_meta.get("calib_end"),
    })
    print(f"[model] sauvegardé dans {MODELS_DIR}")

    return model, calibrator, summary


def build_prediction_outputs(
    upcoming_df: pd.DataFrame,
    raw_proba: np.ndarray,
    cal_proba: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = upcoming_df.copy()
    base["proba_point_1p_raw"] = raw_proba
    base["proba_point_1p_calibree"] = cal_proba
    base["model_variant"] = "enrichi"
    base["calibration_method"] = "sigmoid"

    base["rank_proba_sur_date"] = (
        base.groupby("date_match")["proba_point_1p_calibree"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    base["rank_proba_sur_match"] = (
        base.groupby("id_match")["proba_point_1p_calibree"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    cols_out = META_OUTPUT_COLUMNS + EXTRA_OUTPUT_COLUMNS + [
        "model_variant",
        "calibration_method",
        "proba_point_1p_raw",
        "proba_point_1p_calibree",
        "rank_proba_sur_date",
        "rank_proba_sur_match",
    ]
    cols_out = [c for c in cols_out if c in base.columns]

    pred_cal = base[cols_out].sort_values(
        ["date_match", "proba_point_1p_calibree", "id_match", "nom"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)

    pred_raw = base[cols_out].copy()
    if "proba_point_1p_calibree" in pred_raw.columns:
        pred_raw = pred_raw.drop(columns=["proba_point_1p_calibree"])
    pred_raw = pred_raw.sort_values(
        ["date_match", "proba_point_1p_raw", "id_match", "nom"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)

    return pred_raw, pred_cal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prédire les matchs à venir pour le marché POINT.")
    parser.add_argument(
        "--target-date",
        type=str,
        default=None,
        help="Date cible au format YYYY-MM-DD. Par défaut : première date FUT disponible à partir d'aujourd'hui.",
    )
    parser.add_argument(
        "--include-goalies",
        action="store_true",
        help="Inclure les gardiens dans l'univers de prédiction.",
    )
    parser.add_argument(
        "--recent-lookback-days",
        type=int,
        default=DEFAULT_RECENT_LOOKBACK_DAYS,
        help="Récence maximale (en jours) depuis la dernière apparition historique du joueur pour l'inclure dans le slate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    require_file(MATCHS_PATH)
    require_file(JOUEURS_PATH)
    require_file(FEATURES_HISTORY_PATH)
    ensure_output_dir()

    print("05_predict_upcoming_games.py")
    print(f"Input matchs   : {MATCHS_PATH}")
    print(f"Input joueurs  : {JOUEURS_PATH}")
    print(f"Input history  : {FEATURES_HISTORY_PATH}")
    print(f"Input standings: {TEAM_STANDINGS_PATH} (optionnel)")

    matchs = load_matchs()
    joueurs = load_joueurs()
    history, target_col, date_col = load_history()
    _, standings_summary, standings_by_team = load_standings_optional()

    target_date = choose_target_date(matchs=matchs, target_date_str=args.target_date)
    future_matches = select_future_matches(matchs=matchs, target_date=target_date)

    history_before_target = history[history[date_col] < target_date].copy()
    if history_before_target.empty:
        raise ValueError("Aucune ligne historique disponible avant la date cible.")

    teams = sorted(
        set(future_matches["id_equipe_domicile"].astype(str).str.upper().tolist())
        | set(future_matches["id_equipe_exterieur"].astype(str).str.upper().tolist())
    )

    player_pool = build_recent_player_pool(
        history=history_before_target,
        joueurs_lookup=joueurs,
        target_date=target_date,
        slate_teams=teams,
        include_goalies=args.include_goalies,
        recent_lookback_days=int(args.recent_lookback_days),
    )

    upcoming_universe = build_upcoming_universe(
        future_matches=future_matches,
        history=history_before_target,
        matchs_all=matchs,
        player_pool=player_pool,
        standings_by_team=standings_by_team,
    )

    model, calibrator, fit_summary = fit_point_model_and_calibrator(
        history=history_before_target,
        target_col=target_col,
        date_col=date_col,
        target_date=target_date,
    )

    X_future = to_numeric_frame(upcoming_universe, FEATURE_WHITELIST)
    feature_cols_kept = fit_summary["feature_cols_kept"]
    X_future = X_future[feature_cols_kept].copy()

    raw_proba = model.predict_proba(X_future)[:, 1]
    cal_proba = calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]

    pred_raw, pred_cal = build_prediction_outputs(
        upcoming_df=upcoming_universe,
        raw_proba=raw_proba,
        cal_proba=cal_proba,
    )

    pred_raw.to_csv(PRED_UPCOMING_RAW_PATH, index=False)
    pred_cal.to_csv(PRED_UPCOMING_CAL_PATH, index=False)

    players_per_team = (
        player_pool["team_player_match"].value_counts().sort_index().to_dict()
        if not player_pool.empty else {}
    )

    summary = {
        "status": "ok",
        "target_date": str(target_date.date()),
        "future_matches_rows": int(len(future_matches)),
        "future_players_rows": int(len(upcoming_universe)),
        "future_teams": teams,
        "include_goalies": bool(args.include_goalies),
        "recent_lookback_days": int(args.recent_lookback_days),
        "history_rows_before_target": int(len(history_before_target)),
        "player_pool_rows": int(len(player_pool)),
        "player_pool_per_team": players_per_team,
        "target_column_used": target_col,
        "date_column_used": date_col,
        "standings_summary": standings_summary,
        "fit_summary": fit_summary,
        "outputs": {
            "predictions_upcoming_point_enrichi_v2.csv": str(PRED_UPCOMING_RAW_PATH),
            "predictions_upcoming_point_enrichi_calibre_v2.csv": str(PRED_UPCOMING_CAL_PATH),
        },
        "notes": [
            "Universe future construit depuis la dernière apparition historique connue avant la date cible",
            "joueurs.csv utilisé comme lookup complémentaire, pas comme roster large principal",
            "Aucune colonne de résultat futur utilisée",
            "Réentraînement local nécessaire car le repo ne sauvegarde pas encore d'artefact modèle POINT",
            "Calibration sigmoid ajustée sur une fenêtre historique récente antérieure à la date cible",
            "Le contexte standings / playoffs est injecté si team_standings_daily.csv est disponible",
        ],
    }
    write_json(SUMMARY_PATH, summary)

    print("")
    print("=== DATE CIBLE ===")
    print(target_date.date())

    print("")
    print("=== FUTURE MATCHES ===")
    print(f"rows : {len(future_matches)}")
    print(f"teams: {teams}")

    print("")
    print("=== PLAYER POOL ===")
    print(f"rows : {len(player_pool)}")
    print(f"lookback_days : {args.recent_lookback_days}")

    print("")
    print("=== FUTURE UNIVERSE ===")
    print(f"players rows : {len(upcoming_universe)}")
    print(f"goalies kept : {args.include_goalies}")

    print("")
    print("=== STANDINGS ===")
    print(f"loaded : {standings_summary.get('standings_loaded')}")
    if standings_summary.get("standings_loaded"):
        print(f"dates  : {standings_summary.get('standings_dates')}")
        print(f"teams  : {standings_summary.get('standings_teams')}")

    print("")
    print("=== FIT / CALIBRATION ===")
    print(f"fit rows   : {fit_summary['fit_rows']}")
    print(f"calib rows : {fit_summary['calib_rows']}")
    print(f"features   : {len(fit_summary['feature_cols_kept'])}")
    print(f"calibration: {fit_summary['calibration_method']}")

    print("")
    print("=== SORTIES ===")
    print(f"- {PRED_UPCOMING_RAW_PATH}")
    print(f"- {PRED_UPCOMING_CAL_PATH}")
    print(f"- {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
