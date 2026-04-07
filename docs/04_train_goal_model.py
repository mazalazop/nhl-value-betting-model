#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model/04_train_goal_model.py

Objectif
--------
Entraîner le modèle BUT (joueur marque au moins 1 but) à partir de
data/final/base_features_context_v2.csv.

Même architecture que 02_train_point_model.py :
- validation temporelle
- aucune feature du match courant
- whitelist explicite
- modèle HistGradientBoostingClassifier

Sorties
-------
- outputs/metrics_modele_but_v2.csv
- outputs/predictions_validation_but_v2.csv
- outputs/predictions_test_but_v2.csv
- outputs/04_train_goal_model_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FINAL_DIR = DATA_DIR / "final"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FEATURES_PATH = FINAL_DIR / "base_features_context_v2.csv"

RANDOM_STATE = 42

TARGET_CANDIDATES = [
    "a_marque_un_but",
    "target_goal_1p",
    "target_but",
]

DATE_CANDIDATES = ["date_match", "date", "match_date", "game_date"]

META_CANDIDATES = [
    "date_match", "id_match", "id_joueur", "nom", "player_name",
    "team_player_match", "adversaire_match", "position",
    "home_road_flag", "is_home_player", "saison", "season_source",
]

FEATURE_WHITELIST = [
    "is_home_player", "saison", "nb_matchs_avant_match", "jours_repos",
    "tirs_moy_5", "toi_moy_5", "pp_moy_5", "buts_moy_5",
    "tirs_moy_10", "toi_moy_10", "buts_moy_10",
    "nb_matchs_joues_10", "hist_ok_5", "hist_ok_10",
    "tirs_par_60_5", "buts_par_60_5",
    "goal_hit_rate_last_10", "goal_hit_rate_season_pre",
    "goal_hit_rate_prev_season",
    "nb_matchs_vs_adv_avant", "buts_vs_adv_shrunk", "tirs_vs_adv_shrunk",
    "jours_absence_pre_match", "absence_longue_flag",
    "return_stabilized_flag",
    "current_point_streak_pre", "current_no_point_streak_pre",
    # Team context
    "is_home_team", "jours_repos_team", "team_back_to_back",
    "team_winrate_5", "team_gf_moy_5", "team_ga_moy_5",
    "games_remaining_team_pre", "wildcard_distance_pre",
    "late_season_flag", "playoff_pressure_simple",
]

FORBIDDEN_FEATURE_COLUMNS = {
    "a_marque_un_point", "a_marque_un_but", "buts", "passes", "points",
    "tirs", "temps_de_glace", "temps_pp", "plus_moins", "penalty_minutes",
    "buts_domicile", "buts_exterieur", "buts_match_equipe", "buts_match_adversaire",
    "victoire_equipe", "defaite_equipe", "diff_buts_equipe", "status",
    "id_match", "id_joueur", "date_match", "team_player_match", "adversaire_match",
    "nom", "player_name", "position", "home_road_flag", "season_source",
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")


def ensure_output_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def find_target_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            out = df.copy()
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out = out[out[col].notna()].copy()
            out[col] = out[col].astype(int)
            return out, col
    if "buts" in df.columns:
        out = df.copy()
        out["buts"] = pd.to_numeric(out["buts"], errors="coerce")
        out = out[out["buts"].notna()].copy()
        out["a_marque_un_but"] = (out["buts"] >= 1).astype(int)
        return out, "a_marque_un_but"
    raise ValueError("Impossible de trouver la cible BUT.")


def find_date_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    for col in DATE_CANDIDATES:
        if col in df.columns:
            out = df.copy()
            out[col] = pd.to_datetime(out[col], errors="coerce")
            out = out[out[col].notna()].copy()
            out = out.sort_values(col).reset_index(drop=True)
            return out, col
    raise ValueError("Impossible de trouver une colonne date.")


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


def compute_sample_weights(y: pd.Series) -> np.ndarray:
    y_array = y.astype(int).to_numpy()
    positives = int(y_array.sum())
    negatives = int(len(y_array) - positives)
    if positives == 0 or negatives == 0:
        return np.ones(len(y_array), dtype=float)
    pos_weight = negatives / positives
    return np.where(y_array == 1, pos_weight, 1.0).astype(float)


def build_temporal_splits(df: pd.DataFrame, date_col: str) -> Dict[str, Any]:
    unique_dates = sorted(df[date_col].dropna().dt.strftime("%Y-%m-%d").unique().tolist())
    n = len(unique_dates)
    if n < 12:
        raise ValueError(f"Pas assez de dates uniques: {n}")
    train_end = max(1, int(n * 0.70))
    valid_end = max(train_end + 1, int(n * 0.85))
    if valid_end >= n:
        valid_end = n - 1
    train_dates = set(unique_dates[:train_end])
    valid_dates = set(unique_dates[train_end:valid_end])
    test_dates = set(unique_dates[valid_end:])
    df_l = df.copy()
    df_l["_dk"] = df_l[date_col].dt.strftime("%Y-%m-%d")
    return {
        "train": df_l[df_l["_dk"].isin(train_dates)].drop(columns=["_dk"]),
        "validation": df_l[df_l["_dk"].isin(valid_dates)].drop(columns=["_dk"]),
        "test": df_l[df_l["_dk"].isin(test_dates)].drop(columns=["_dk"]),
        "meta": {
            "train_start": min(train_dates), "train_end": max(train_dates),
            "validation_start": min(valid_dates), "validation_end": max(valid_dates),
            "test_start": min(test_dates), "test_end": max(test_dates),
        },
    }


def safe_roc_auc(y_true, proba):
    if len(np.unique(y_true)) < 2: return float("nan")
    return float(roc_auc_score(y_true, proba))


def safe_ap(y_true, proba):
    if len(np.unique(y_true)) < 2: return float("nan")
    return float(average_precision_score(y_true, proba))


def evaluate(variant, split_name, y_true, proba):
    y = y_true.astype(int).to_numpy()
    return pd.DataFrame([{
        "model_variant": variant, "split": split_name,
        "rows": int(len(y)), "positive_rate": float(np.mean(y)),
        "mean_predicted_proba": float(np.mean(proba)),
        "brier_score": float(brier_score_loss(y, proba)),
        "log_loss": float(log_loss(y, proba, labels=[0, 1])),
        "roc_auc": safe_roc_auc(y, proba),
        "average_precision": safe_ap(y, proba),
    }])


def main() -> None:
    require_file(FEATURES_PATH)
    ensure_output_dir()

    print("04_train_goal_model.py")
    df = pd.read_csv(FEATURES_PATH, low_memory=False)
    for col in df.columns:
        if pd.api.types.is_bool_dtype(df[col]):
            df[col] = df[col].astype(int)

    df, target_col = find_target_column(df)
    df, date_col = find_date_column(df)

    feature_cols = [c for c in FEATURE_WHITELIST
                    if c in df.columns and c not in FORBIDDEN_FEATURE_COLUMNS
                    and c != target_col and c != date_col]
    print(f"Features disponibles : {len(feature_cols)}")

    splits = build_temporal_splits(df, date_col)
    train_df, valid_df, test_df = splits["train"], splits["validation"], splits["test"]

    X_train = to_numeric_frame(train_df, feature_cols)
    X_valid = to_numeric_frame(valid_df, feature_cols)
    X_test = to_numeric_frame(test_df, feature_cols)

    # Remove all-nan / constant cols
    ok_mask = ~X_train.isna().all(axis=0) & (X_train.nunique(dropna=True) > 1)
    kept = X_train.columns[ok_mask].tolist()
    X_train, X_valid, X_test = X_train[kept], X_valid[kept], X_test[kept]

    y_train = train_df[target_col].astype(int)
    y_valid = valid_df[target_col].astype(int)
    y_test = test_df[target_col].astype(int)

    model = HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.05, max_iter=300, max_depth=5,
        min_samples_leaf=80, l2_regularization=1.5, early_stopping=False,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train, sample_weight=compute_sample_weights(y_train))

    valid_proba = model.predict_proba(X_valid)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    metrics = pd.concat([
        evaluate("but_enrichi", "validation", y_valid, valid_proba),
        evaluate("but_enrichi", "test", y_test, test_proba),
    ], ignore_index=True)

    metrics_path = OUTPUTS_DIR / "metrics_modele_but_v2.csv"
    metrics.to_csv(metrics_path, index=False)

    # Predictions
    for split_name, split_df, proba in [
        ("validation", valid_df, valid_proba),
        ("test", test_df, test_proba),
    ]:
        keep_cols = [c for c in META_CANDIDATES if c in split_df.columns]
        pred = split_df[keep_cols].copy()
        pred["target_but_1p"] = split_df[target_col].astype(int).values
        pred["proba_but_1p_raw"] = proba
        pred["split"] = split_name
        pred.to_csv(OUTPUTS_DIR / f"predictions_{split_name}_but_v2.csv", index=False)

    summary = {
        "status": "ok",
        "input_file": str(FEATURES_PATH),
        "target_column": target_col,
        "features_kept": len(kept),
        "split_meta": splits["meta"],
        "metrics": metrics.to_dict(orient="records"),
    }
    write_json(OUTPUTS_DIR / "04_train_goal_model_summary.json", summary)

    print(f"Train: {len(train_df)} | Valid: {len(valid_df)} | Test: {len(test_df)}")
    print(f"Features: {len(kept)}")
    print(metrics.to_string(index=False))
    print("Terminé.")


if __name__ == "__main__":
    main()
