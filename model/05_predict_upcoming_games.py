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
- l'univers utilise un roster NHL récent, avec fallback historique explicite ;
- le modèle POINT est réentraîné localement dans ce script ;
- calibration commune à 03, choisie temporellement avant la date cible ;
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

import numpy as np
import pandas as pd
from henachel.features import build_future_features
from henachel.calibration import select_calibrator
from henachel.data import final_mask


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    "start_time_utc",
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






def find_target_column(df):
    from henachel.point import find_target
    return find_target(df, TARGET_CANDIDATES)


def find_date_column(df):
    from henachel.point import find_date
    return find_date(df, DATE_CANDIDATES)


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






def choose_target_date(matchs, target_date_str):
    # NHL schedule gameDate, not the French local calendar date of puck drop.
    value = target_date_str or pd.Timestamp.now(tz='America/New_York').date().isoformat()
    parsed = pd.to_datetime(value, errors='raise')
    if parsed.tzinfo is not None: raise ValueError('Target must be an NHL calendar date')
    return parsed.normalize()


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
    if not final_mask(df).all():
        raise ValueError("Non-final or unknown games in feature history")
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




def load_standings_optional():
    from henachel.features import charger_standings
    standings, summary = charger_standings(TEAM_STANDINGS_PATH)
    by_team = {str(team): group.copy() for team, group in standings.groupby('team_abbrev')} if standings is not None else {}
    return standings, summary, by_team


def select_future_matches(matchs: pd.DataFrame, target_date: pd.Timestamp) -> pd.DataFrame:
    fut = matchs[
        (matchs["status"].astype(str).str.upper() == "FUT")
        & (matchs["date_match"].dt.normalize() == target_date)
    ].copy()

    return fut.sort_values(["date_match", "id_match"]).reset_index(drop=True)
















def build_recent_player_pool(history, joueurs_lookup, target_date, slate_teams,
                             include_goalies=False, recent_lookback_days=45):
    from henachel.rosters import choose_players
    path = RAW_DIR / 'roster_current.csv'
    roster = pd.read_csv(path) if path.exists() else None
    hist = history.copy()
    lookup = joueurs_lookup.set_index('id_joueur')
    for column in ['nom', 'position']:
        if column not in hist: hist[column] = hist.id_joueur.map(lookup[column])
        else: hist[column] = hist[column].fillna(hist.id_joueur.map(lookup[column]))
    return choose_players(hist, roster, slate_teams, target_date,
                          lookback_days=recent_lookback_days, include_goalies=include_goalies)


def build_upcoming_universe(future_matches, history, matchs_all, player_pool, standings_by_team):
    standings = pd.concat(list(standings_by_team.values()), ignore_index=True) if standings_by_team else None
    return build_future_features(history, future_matches, player_pool, standings)


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
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
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

    from henachel.point import point_model
    model = point_model()
    model.fit(X_fit, y_fit, sample_weight=compute_sample_weights(y_fit))

    calib_raw_proba = model.predict_proba(X_calib)[:, 1]
    calibrator, calibration_info = select_calibrator(calib_raw_proba, y_calib, calib_df[date_col])

    summary = {
        "fit_rows": int(len(fit_df)),
        "calib_rows": int(len(calib_df)),
        "fit_positive_rate": float(y_fit.mean()),
        "calib_positive_rate": float(y_calib.mean()),
        "fit_calibration_split": split_meta,
        "feature_cols_kept": kept_cols,
        "feature_cols_dropped_train_only": dropped_cols,
        "calibration_method": calibrator.method,
        "calibration_diagnostics": calibration_info,
    }
    return model, calibrator, summary


def build_prediction_outputs(
    upcoming_df: pd.DataFrame,
    raw_proba: np.ndarray,
    cal_proba: np.ndarray,
    calibration_method: str = "unknown",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    base = upcoming_df.copy()
    base["proba_point_1p_raw"] = raw_proba
    base["proba_point_1p_calibree"] = cal_proba
    base["model_variant"] = "enrichi"
    base["calibration_method"] = calibration_method

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


def write_empty_predictions(status, target_date):
    frame = pd.DataFrame(columns=META_OUTPUT_COLUMNS + EXTRA_OUTPUT_COLUMNS)
    raw, calibrated = build_prediction_outputs(frame, np.array([]), np.array([]))
    raw.to_csv(PRED_UPCOMING_RAW_PATH, index=False)
    calibrated.to_csv(PRED_UPCOMING_CAL_PATH, index=False)
    write_json(SUMMARY_PATH, {"status": status, "target_date": str(target_date.date())})


def main() -> None:
    args = parse_args()

    require_file(MATCHS_PATH)
    ensure_output_dir()

    print("05_predict_upcoming_games.py")
    print(f"Input matchs   : {MATCHS_PATH}")
    print(f"Input joueurs  : {JOUEURS_PATH}")
    print(f"Input history  : {FEATURES_HISTORY_PATH}")
    print(f"Input standings: {TEAM_STANDINGS_PATH} (optionnel)")

    matchs = load_matchs()
    target_date = choose_target_date(matchs=matchs, target_date_str=args.target_date)
    future_matches = select_future_matches(matchs=matchs, target_date=target_date)
    if future_matches.empty:
        write_empty_predictions('no_games', target_date)
        return
    require_file(JOUEURS_PATH)
    require_file(FEATURES_HISTORY_PATH)
    joueurs = load_joueurs()
    history, target_col, date_col = load_history()
    _, standings_summary, standings_by_team = load_standings_optional()

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

    if player_pool.empty:
        write_empty_predictions("no_eligible_players", target_date)
        return

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
    cal_proba = calibrator.predict(raw_proba)

    pred_raw, pred_cal = build_prediction_outputs(
        upcoming_df=upcoming_universe,
        raw_proba=raw_proba,
        cal_proba=cal_proba,
        calibration_method=calibrator.method,
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
            "Calibration choisie sur une séparation temporelle interne puis ajustée avant la date cible",
            "Le contexte standings / playoffs est injecté si team_standings_daily.csv est disponible",
        ],
    }
    from henachel.manifest import manifest
    from henachel.point import POINT_PARAMS
    summary["manifest"] = manifest([FEATURES_HISTORY_PATH, MATCHS_PATH, TEAM_STANDINGS_PATH, RAW_DIR / "roster_current.csv"], POINT_PARAMS)
    import joblib
    joblib.dump({"model": model, "calibrator": calibrator, "features": feature_cols_kept, "manifest": summary["manifest"], "fit": fit_summary}, OUTPUTS_DIR / "05_point_bundle.joblib")
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
