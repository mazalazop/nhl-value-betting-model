#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model/02b_ablation_point_model.py

Objectif
--------
Mesurer d'où vient réellement le gain du modèle POINT via une ablation simple,
sans toucher au pipeline officiel d'entraînement/calibration.

Principe
--------
- Charge base_features_context_v2.csv
- Construit l'univers POINT leak-free
- Fait un split temporel propre sur dates uniques :
    train / validation / test
- Entraîne le même type de modèle sur plusieurs familles de features
- Compare les résultats sur validation et test

Sorties
-------
- outputs/ablation_point_model_v2.csv
- outputs/ablation_point_model_summary.json
- outputs/ablation_point_feature_sets_v2.json

Important
---------
- Script d'audit méthodologique
- Ne remplace pas 02_train_point_model.py
- Ne remplace pas 03_calibrate_point_model.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

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

INPUT_CANDIDATES = [
    FINAL_DIR / "base_features_context_v2.csv",
    PROJECT_ROOT / "base_features_context_v2.csv",
]

ABLATION_OUTPUT_PATH = OUTPUTS_DIR / "ablation_point_model_v2.csv"
SUMMARY_OUTPUT_PATH = OUTPUTS_DIR / "ablation_point_model_summary.json"
FEATURE_SETS_OUTPUT_PATH = OUTPUTS_DIR / "ablation_point_feature_sets_v2.json"

RANDOM_STATE = 42
EPSILON = 1e-6

# =========================
# FEATURES PAR FAMILLE
# =========================

CORE_FEATURES = [
    "is_home_player",
    "jours_repos",
    "nb_matchs_avant_match",
    "nb_matchs_joues_10",
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
    "tirs_par_60_5",
    "points_par_60_5",
    "buts_par_60_5",
    "hist_ok_5",
    "hist_ok_10",
    "is_premier_match_joueur",
]

TEAM_CONTEXT_FEATURES = [
    "jours_repos_team",
    "team_back_to_back",
    "team_back_to_back_away",
    "consecutive_away_games",
    "team_winrate_5",
    "team_gf_moy_5",
    "team_ga_moy_5",
    "team_context_found",
]

VS_ADV_FEATURES = [
    "nb_matchs_vs_adv_avant",
    "points_vs_adv_shrunk",
    "buts_vs_adv_shrunk",
    "tirs_vs_adv_shrunk",
]

RETURN_ABSENCE_FEATURES = [
    "jours_absence_pre_match",
    "absence_longue_flag",
    "matchs_depuis_retour_avant_match",
    "ratio_toi_retour_vs_pre_absence",
    "ratio_pp_retour_vs_pre_absence",
    "eligible_post_retour",
]

TARGET_COL = "a_marque_un_point"
DATE_COL = "date_match"

POINT_UNIVERSE_FILTERS = {
    "hist_ok_5": 1,
    "toi_moy_5_min": 10.0,
}

# =========================
# OUTILS
# =========================

def ensure_outputs_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def require_columns(df: pd.DataFrame, columns: List[str], label: str = "") -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        prefix = f"{label} - " if label else ""
        raise ValueError(f"{prefix}Colonnes manquantes : {missing}")


def clip_proba(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.clip(arr, EPSILON, 1.0 - EPSILON)


def compute_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
    y_true = pd.Series(y_true).astype(int).to_numpy()
    proba = clip_proba(proba)

    out = {
        "rows": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "mean_predicted_proba": float(np.mean(proba)),
        "logloss": float(log_loss(y_true, proba, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, proba)),
    }

    if len(np.unique(y_true)) >= 2:
        out["auc"] = float(roc_auc_score(y_true, proba))
        out["avg_precision"] = float(average_precision_score(y_true, proba))
    else:
        out["auc"] = float("nan")
        out["avg_precision"] = float("nan")

    return out


def find_input_file() -> Path:
    for path in INPUT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Impossible de trouver base_features_context_v2.csv.\n"
        f"Chemins testés : {[str(p) for p in INPUT_CANDIDATES]}"
    )


def build_point_universe(df: pd.DataFrame) -> pd.DataFrame:
    required = [DATE_COL, TARGET_COL, "hist_ok_5", "toi_moy_5"] + CORE_FEATURES
    require_columns(df, required, "build_point_universe")

    temp = df.copy()
    temp[DATE_COL] = pd.to_datetime(temp[DATE_COL], errors="coerce")
    temp = temp.dropna(subset=[DATE_COL]).copy()

    temp = temp[
        (pd.to_numeric(temp["hist_ok_5"], errors="coerce") == POINT_UNIVERSE_FILTERS["hist_ok_5"]) &
        (pd.to_numeric(temp["toi_moy_5"], errors="coerce") >= POINT_UNIVERSE_FILTERS["toi_moy_5_min"])
    ].copy()

    temp = temp.sort_values([DATE_COL, "id_match", "id_joueur"], na_position="last").reset_index(drop=True)
    return temp


def split_dates_train_val_test(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    test_ratio: float = 0.20,
    val_ratio_within_train: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col]).copy()

    unique_dates = sorted(pd.Series(temp[date_col].dt.normalize().unique()).tolist())
    if len(unique_dates) < 10:
        raise ValueError("Pas assez de dates uniques pour une ablation temporelle robuste.")

    n_dates = len(unique_dates)

    idx_test_start = int(np.floor(n_dates * (1 - test_ratio)))
    idx_test_start = max(1, min(idx_test_start, n_dates - 1))
    test_start_date = pd.Timestamp(unique_dates[idx_test_start])

    train_val_df = temp[temp[date_col] < test_start_date].copy()
    test_df = temp[temp[date_col] >= test_start_date].copy()

    train_val_dates = sorted(pd.Series(train_val_df[date_col].dt.normalize().unique()).tolist())
    if len(train_val_dates) < 5:
        raise ValueError("Pas assez de dates uniques dans train+validation.")

    idx_val_start = int(np.floor(len(train_val_dates) * (1 - val_ratio_within_train)))
    idx_val_start = max(1, min(idx_val_start, len(train_val_dates) - 1))
    val_start_date = pd.Timestamp(train_val_dates[idx_val_start])

    train_df = train_val_df[train_val_df[date_col] < val_start_date].copy()
    val_df = train_val_df[train_val_df[date_col] >= val_start_date].copy()

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError("Split train/validation/test invalide.")

    split_info = {
        "train_end_before": str(val_start_date.date()),
        "validation_start": str(val_start_date.date()),
        "validation_end_before": str(test_start_date.date()),
        "test_start": str(test_start_date.date()),
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_unique_dates": int(train_df[date_col].dt.normalize().nunique()),
        "validation_unique_dates": int(val_df[date_col].dt.normalize().nunique()),
        "test_unique_dates": int(test_df[date_col].dt.normalize().nunique()),
    }

    return train_df, val_df, test_df, split_info


def build_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    all_candidates = CORE_FEATURES + TEAM_CONTEXT_FEATURES + VS_ADV_FEATURES + RETURN_ABSENCE_FEATURES
    available = {c for c in all_candidates if c in df.columns}

    def keep(cols: List[str]) -> List[str]:
        return [c for c in cols if c in available]

    feature_sets = {
        "core": keep(CORE_FEATURES),
        "core_plus_team": keep(CORE_FEATURES + TEAM_CONTEXT_FEATURES),
        "core_plus_vs_adv": keep(CORE_FEATURES + VS_ADV_FEATURES),
        "core_plus_return_absence": keep(CORE_FEATURES + RETURN_ABSENCE_FEATURES),
        "core_plus_team_vs_adv": keep(CORE_FEATURES + TEAM_CONTEXT_FEATURES + VS_ADV_FEATURES),
        "full": keep(CORE_FEATURES + TEAM_CONTEXT_FEATURES + VS_ADV_FEATURES + RETURN_ABSENCE_FEATURES),
    }

    for name, cols in feature_sets.items():
        if len(cols) == 0:
            raise ValueError(f"Le jeu de features '{name}' est vide.")

    return feature_sets


def default_fill_map() -> Dict[str, float]:
    return {
        # core
        "is_home_player": 0,
        "jours_repos": 7,
        "nb_matchs_avant_match": 0,
        "nb_matchs_joues_10": 0,
        "tirs_moy_5": 0.0,
        "toi_moy_5": 0.0,
        "pp_moy_5": 0.0,
        "points_moy_5": 0.0,
        "buts_moy_5": 0.0,
        "passes_moy_5": 0.0,
        "tirs_moy_10": 0.0,
        "toi_moy_10": 0.0,
        "points_moy_10": 0.0,
        "buts_moy_10": 0.0,
        "tirs_par_60_5": 0.0,
        "points_par_60_5": 0.0,
        "buts_par_60_5": 0.0,
        "hist_ok_5": 0,
        "hist_ok_10": 0,
        "is_premier_match_joueur": 0,
        # team
        "jours_repos_team": 3,
        "team_back_to_back": 0,
        "team_back_to_back_away": 0,
        "consecutive_away_games": 0,
        "team_winrate_5": 0.5,
        "team_gf_moy_5": 2.8,
        "team_ga_moy_5": 2.8,
        "team_context_found": 0,
        # vs adv
        "nb_matchs_vs_adv_avant": 0,
        "points_vs_adv_shrunk": 0.0,
        "buts_vs_adv_shrunk": 0.0,
        "tirs_vs_adv_shrunk": 0.0,
        # retour absence
        "jours_absence_pre_match": 3,
        "absence_longue_flag": 0,
        "matchs_depuis_retour_avant_match": 0,
        "ratio_toi_retour_vs_pre_absence": 0.0,
        "ratio_pp_retour_vs_pre_absence": 0.0,
        "eligible_post_retour": 0,
    }


def prepare_xy(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    target_col: str = TARGET_COL,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    require_columns(train_df, features + [target_col], "prepare_xy train")
    require_columns(val_df, features + [target_col], "prepare_xy val")
    require_columns(test_df, features + [target_col], "prepare_xy test")

    fill_map = default_fill_map()

    x_train = train_df[features].copy()
    x_val = val_df[features].copy()
    x_test = test_df[features].copy()

    y_train = pd.to_numeric(train_df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    y_val = pd.to_numeric(val_df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()
    y_test = pd.to_numeric(test_df[target_col], errors="coerce").fillna(0).astype(int).to_numpy()

    for col in features:
        x_train[col] = pd.to_numeric(x_train[col], errors="coerce")
        x_val[col] = pd.to_numeric(x_val[col], errors="coerce")
        x_test[col] = pd.to_numeric(x_test[col], errors="coerce")

        if col in fill_map:
            fill_value = fill_map[col]
        else:
            train_non_na = x_train[col].dropna()
            fill_value = float(train_non_na.median()) if not train_non_na.empty else 0.0

        x_train[col] = x_train[col].fillna(fill_value)
        x_val[col] = x_val[col].fillna(fill_value)
        x_test[col] = x_test[col].fillna(fill_value)

    return x_train, x_val, x_test, y_train, y_val, y_test


def build_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=0.05,
        max_iter=400,
        max_depth=4,
        min_samples_leaf=50,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


def run_one_ablation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_set_name: str,
    features: List[str],
) -> List[dict]:
    x_train, x_val, x_test, y_train, y_val, y_test = prepare_xy(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        features=features,
        target_col=TARGET_COL,
    )

    model = build_model()
    model.fit(x_train, y_train)

    p_val = clip_proba(model.predict_proba(x_val)[:, 1])
    p_test = clip_proba(model.predict_proba(x_test)[:, 1])

    val_metrics = compute_metrics(y_val, p_val)
    test_metrics = compute_metrics(y_test, p_test)

    rows = [
        {
            "feature_set": feature_set_name,
            "split": "validation",
            "n_features": len(features),
            "features": " | ".join(features),
            **val_metrics,
        },
        {
            "feature_set": feature_set_name,
            "split": "test",
            "n_features": len(features),
            "features": " | ".join(features),
            **test_metrics,
        },
    ]
    return rows


def main() -> None:
    ensure_outputs_dir()

    input_path = find_input_file()
    df = pd.read_csv(input_path)

    required_min = [
        "id_joueur",
        "id_match",
        DATE_COL,
        TARGET_COL,
        "hist_ok_5",
        "toi_moy_5",
    ] + CORE_FEATURES
    require_columns(df, required_min, "main input")

    df_point = build_point_universe(df)
    train_df, val_df, test_df, split_info = split_dates_train_val_test(df_point)

    feature_sets = build_feature_sets(df_point)

    print("02b_ablation_point_model.py")
    print(f"Input utilisé : {input_path}")
    print(f"Rows après univers POINT : {len(df_point)}")
    print("")
    print("=== SPLIT TEMPOREL ===")
    for k, v in split_info.items():
        print(f"{k}: {v}")

    all_rows: List[dict] = []
    for feature_set_name, features in feature_sets.items():
        print("")
        print(f"=== ABLATION : {feature_set_name} ===")
        print(f"Nombre de features : {len(features)}")
        rows = run_one_ablation(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            feature_set_name=feature_set_name,
            features=features,
        )
        all_rows.extend(rows)

    results_df = pd.DataFrame(all_rows)

    results_df["sort_logloss"] = results_df["logloss"]
    results_df["sort_brier"] = results_df["brier"]

    validation_ranking = (
        results_df[results_df["split"] == "validation"]
        .sort_values(["sort_logloss", "sort_brier"], ascending=[True, True])
        .reset_index(drop=True)
    )
    best_feature_set = str(validation_ranking.iloc[0]["feature_set"])

    results_df["best_on_validation"] = (results_df["feature_set"] == best_feature_set).astype(int)
    results_df = results_df.drop(columns=["sort_logloss", "sort_brier"])
    results_df = results_df.sort_values(["split", "logloss", "brier"], ascending=[True, True, True]).reset_index(drop=True)

    results_df.to_csv(ABLATION_OUTPUT_PATH, index=False)

    feature_sets_payload = {
        "input_file": str(input_path),
        "feature_sets": feature_sets,
    }
    write_json(FEATURE_SETS_OUTPUT_PATH, feature_sets_payload)

    summary = {
        "status": "ok",
        "input_file": str(input_path),
        "rows_loaded": int(len(df)),
        "rows_point_universe": int(len(df_point)),
        "split_info": split_info,
        "best_feature_set_on_validation": best_feature_set,
        "outputs": {
            "ablation_csv": str(ABLATION_OUTPUT_PATH),
            "feature_sets_json": str(FEATURE_SETS_OUTPUT_PATH),
            "summary_json": str(SUMMARY_OUTPUT_PATH),
        },
    }
    write_json(SUMMARY_OUTPUT_PATH, summary)

    print("")
    print("=== TOP VALIDATION ===")
    print(validation_ranking[[
        "feature_set",
        "n_features",
        "logloss",
        "brier",
        "auc",
        "avg_precision",
    ]])

    print("")
    print(f"Meilleur jeu de features sur validation : {best_feature_set}")
    print("")
    print("=== SORTIES ===")
    print(f"- {ABLATION_OUTPUT_PATH}")
    print(f"- {FEATURE_SETS_OUTPUT_PATH}")
    print(f"- {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
