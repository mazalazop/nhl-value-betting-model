#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model/02_train_point_model.py

Objectif
--------
Entraîner le modèle POINT (joueur marque au moins 1 point) de façon robuste
à partir de :
- data/final/base_features_context_v2.csv

Principes de sécurité
---------------------
- validation temporelle uniquement
- aucune feature du match courant
- aucune cible ou quasi-cible dans les features
- listes de features explicites (whitelist), pas de collecte automatique fragile
- intégration des nouvelles features pré-match issues de 01_build_base_features.py
  (hit rates récents, streaks, retours d'absence stabilisés, contexte standings/playoffs)

Sorties
-------
- outputs/metrics_modele_point_v2.csv
- outputs/metrics_modele_point_enrichi_v2.csv
- outputs/predictions_validation_point_v2.csv
- outputs/predictions_test_point_v2.csv
- outputs/predictions_validation_point_enrichi_v2.csv
- outputs/predictions_test_point_enrichi_v2.csv
- outputs/02_train_point_model_summary.json
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

META_CANDIDATES = [
    "date_match",
    "id_match",
    "id_joueur",
    "nom",
    "player_name",
    "team_player_match",
    "adversaire_match",
    "team_name_match",
    "opponent_name_match",
    "position",
    "home_road_flag",
    "is_home_player",
    "saison",
    "season_source",
]

# Features validées comme connues avant match.
# Le PP du match courant ne doit jamais entrer directement dans le modèle :
# seul son historique pré-match (ex: pp_moy_5) est autorisé.
BASELINE_FEATURE_WHITELIST = [
    # Contexte joueur / volume / usage
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
    # Historique vs adversaire
    "nb_matchs_vs_adv_avant",
    "points_vs_adv_5",
    "buts_vs_adv_5",
    "tirs_vs_adv_5",
    "points_vs_adv_shrunk",
    "buts_vs_adv_shrunk",
    "tirs_vs_adv_shrunk",
    # Retour d'absence / stabilisation
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
    # Hit rates / streaks / sous-performance récente
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
]

ENRICHED_EXTRA_WHITELIST = [
    # Contexte équipe
    "is_home_team",
    "jours_repos_team",
    "team_back_to_back",
    "team_back_to_back_away",
    "consecutive_away_games",
    "team_winrate_5",
    "team_gf_moy_5",
    "team_ga_moy_5",
    "team_games_played_pre_approx",
    # Standings / fin de saison / pression playoffs
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

# Colonnes interdites, même si elles existent dans la base.
FORBIDDEN_FEATURE_COLUMNS = {
    # cible / quasi-cible
    "target_point_1p",
    "target_points_1p",
    "target_point",
    "label_point_1p",
    "y_point",
    "point_1p",
    "a_marque_un_point",
    "a_marque_un_but",
    # résultats du match courant
    "buts",
    "passes",
    "points",
    "tirs",
    "temps_de_glace",
    "temps_pp",
    "plus_moins",
    "penalty_minutes",
    "buts_domicile",
    "buts_exterieur",
    "buts_match_equipe",
    "buts_match_adversaire",
    "victoire_equipe",
    "defaite_equipe",
    "diff_buts_equipe",
    "status",
    # qualité / fusion
    "_merge",
    "match_trouve",
    "team_attendue_match",
    "adversaire_attendu_match",
    "check_team_ok",
    "check_opp_ok",
    "team_context_found",
    "check_home_team_context_ok",
    # identifiants / dates / textes
    "id_match",
    "id_joueur",
    "date_match",
    "date_match_match",
    "date",
    "match_date",
    "game_date",
    "date_game",
    "team_player_match",
    "adversaire_match",
    "team_name_match",
    "opponent_name_match",
    "nom",
    "player_name",
    "position",
    "home_road_flag",
    "season_source",
    "standings_lookup_date_pre",
    # règles métier de sélection, interdites dans l'entraînement
    "hard_exclude_hot_streak_pre",
    "is_value_bet",
    "value_gap",
    # ids équipes bruts
    "id_equipe_domicile",
    "id_equipe_exterieur",
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

    if "points" in df.columns:
        out = df.copy()
        out["points"] = pd.to_numeric(out["points"], errors="coerce")
        out = out[out["points"].notna()].copy()
        out["target_point_1p"] = (out["points"] >= 1).astype(int)
        return out, "target_point_1p"

    raise ValueError(
        "Impossible de trouver la cible POINT. "
        "Aucune colonne target reconnue, et la colonne 'points' est absente."
    )


def find_date_column(df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    for col in DATE_CANDIDATES:
        if col in df.columns:
            out = df.copy()
            out[col] = pd.to_datetime(out[col], errors="coerce")
            out = out[out[col].notna()].copy()
            out = out.sort_values(col).reset_index(drop=True)
            return out, col

    raise ValueError(
        "Impossible de trouver une colonne date exploitable. "
        "Colonnes attendues possibles : "
        + ", ".join(DATE_CANDIDATES)
    )


def normalize_boolean_like_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_bool_dtype(out[col]):
            out[col] = out[col].astype(int)
    return out


def resolve_feature_list(
    df: pd.DataFrame,
    requested_cols: List[str],
    target_col: str,
    date_col: str,
) -> Tuple[List[str], List[str], List[str]]:
    kept: List[str] = []
    missing: List[str] = []
    forbidden_requested: List[str] = []

    for col in requested_cols:
        if col in FORBIDDEN_FEATURE_COLUMNS or col == target_col or col == date_col:
            forbidden_requested.append(col)
            continue
        if col not in df.columns:
            missing.append(col)
            continue
        kept.append(col)

    return kept, missing, forbidden_requested


def assert_no_forbidden_features(feature_cols: List[str], target_col: str, date_col: str) -> None:
    bad = sorted(
        {
            col for col in feature_cols
            if col in FORBIDDEN_FEATURE_COLUMNS or col == target_col or col == date_col
        }
    )
    if bad:
        raise ValueError(
            "Features interdites détectées dans la liste finale : "
            + ", ".join(bad)
        )


def build_temporal_splits(df: pd.DataFrame, date_col: str) -> Dict[str, Any]:
    unique_dates = sorted(df[date_col].dropna().dt.strftime("%Y-%m-%d").unique().tolist())

    if len(unique_dates) < 12:
        raise ValueError(
            "Pas assez de dates uniques pour faire un split temporel propre "
            f"(trouvé : {len(unique_dates)})."
        )

    n_dates = len(unique_dates)
    train_end = max(1, int(n_dates * 0.70))
    valid_end = max(train_end + 1, int(n_dates * 0.85))

    if valid_end >= n_dates:
        valid_end = n_dates - 1

    train_dates = set(unique_dates[:train_end])
    valid_dates = set(unique_dates[train_end:valid_end])
    test_dates = set(unique_dates[valid_end:])

    if len(valid_dates) == 0 or len(test_dates) == 0:
        raise ValueError("Split temporel invalide : validation ou test vide.")

    df_local = df.copy()
    df_local["_date_key_for_split"] = df_local[date_col].dt.strftime("%Y-%m-%d")

    train_df = df_local[df_local["_date_key_for_split"].isin(train_dates)].copy()
    valid_df = df_local[df_local["_date_key_for_split"].isin(valid_dates)].copy()
    test_df = df_local[df_local["_date_key_for_split"].isin(test_dates)].copy()

    for subset_name, subset_df in [("train", train_df), ("validation", valid_df), ("test", test_df)]:
        if subset_df.empty:
            raise ValueError(f"Split temporel invalide : subset vide pour {subset_name}.")

    return {
        "train": train_df.drop(columns=["_date_key_for_split"]),
        "validation": valid_df.drop(columns=["_date_key_for_split"]),
        "test": test_df.drop(columns=["_date_key_for_split"]),
        "meta": {
            "train_start": min(train_dates),
            "train_end": max(train_dates),
            "validation_start": min(valid_dates),
            "validation_end": max(valid_dates),
            "test_start": min(test_dates),
            "test_end": max(test_dates),
        },
    }


def to_numeric_frame(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)

    for col in cols:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            series = series.astype(int)
        else:
            series = pd.to_numeric(series, errors="coerce")
        out[col] = series

    out = out.replace([np.inf, -np.inf], np.nan).astype(float)
    return out


def keep_train_valid_features_only(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    not_all_nan = ~X_train.isna().all(axis=0)
    variable = X_train.nunique(dropna=True) > 1
    kept_cols = X_train.columns[not_all_nan & variable].tolist()
    dropped_cols = [c for c in X_train.columns if c not in kept_cols]

    if len(kept_cols) < 5:
        raise ValueError(
            f"Trop peu de features utilisables après nettoyage train-only : {len(kept_cols)}"
        )

    return (
        X_train[kept_cols].copy(),
        X_valid[kept_cols].copy(),
        X_test[kept_cols].copy(),
        kept_cols,
        dropped_cols,
    )


def compute_sample_weights(y: pd.Series) -> np.ndarray:
    y_array = y.astype(int).to_numpy()
    positives = int(y_array.sum())
    negatives = int(len(y_array) - positives)

    if positives == 0 or negatives == 0:
        return np.ones(len(y_array), dtype=float)

    pos_weight = negatives / positives
    return np.where(y_array == 1, pos_weight, 1.0).astype(float)


def safe_roc_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, proba))


def safe_average_precision(y_true: np.ndarray, proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, proba))


def precision_and_lift_top_k(
    y_true: np.ndarray,
    proba: np.ndarray,
    top_ratio: float = 0.10,
) -> Tuple[float, float, int]:
    if len(y_true) == 0:
        return float("nan"), float("nan"), 0

    k = max(1, int(np.ceil(len(y_true) * top_ratio)))
    order = np.argsort(-proba)
    top_idx = order[:k]

    precision_top_k = float(np.mean(y_true[top_idx]))
    base_rate = float(np.mean(y_true))
    lift_top_k = float(precision_top_k / base_rate) if base_rate > 0 else float("nan")

    return precision_top_k, lift_top_k, k


def evaluate_split(
    model_variant: str,
    split_name: str,
    y_true: pd.Series,
    proba: np.ndarray,
) -> pd.DataFrame:
    y_array = y_true.astype(int).to_numpy()

    precision_top_10pct, lift_top_10pct, top_k = precision_and_lift_top_k(
        y_true=y_array,
        proba=proba,
        top_ratio=0.10,
    )

    metrics = {
        "model_variant": model_variant,
        "split": split_name,
        "rows": int(len(y_array)),
        "positive_rate": float(np.mean(y_array)),
        "mean_predicted_proba": float(np.mean(proba)),
        "brier_score": float(brier_score_loss(y_array, proba)),
        "log_loss": float(log_loss(y_array, proba, labels=[0, 1])),
        "roc_auc": safe_roc_auc(y_array, proba),
        "average_precision": safe_average_precision(y_array, proba),
        "precision_top_10pct": precision_top_10pct,
        "lift_top_10pct": lift_top_10pct,
        "top_k_count": int(top_k),
    }

    return pd.DataFrame([metrics])


def build_prediction_frame(
    df_subset: pd.DataFrame,
    date_col: str,
    target_col: str,
    proba: np.ndarray,
    model_variant: str,
    split_name: str,
) -> pd.DataFrame:
    keep_cols = [col for col in META_CANDIDATES if col in df_subset.columns]

    if date_col not in keep_cols:
        keep_cols = [date_col] + keep_cols

    keep_cols = list(dict.fromkeys(keep_cols))

    pred = df_subset[keep_cols].copy()
    pred["target_point_1p"] = df_subset[target_col].astype(int).values
    pred["proba_point_1p_raw"] = proba
    pred["model_variant"] = model_variant
    pred["split"] = split_name

    if date_col in pred.columns:
        pred["rank_proba_sur_date"] = (
            pred.groupby(date_col)["proba_point_1p_raw"]
            .rank(method="first", ascending=False)
            .astype(int)
        )

    pred = pred.sort_values([date_col, "proba_point_1p_raw"], ascending=[True, False]).reset_index(drop=True)
    return pred


def train_one_variant(
    variant_name: str,
    feature_cols: List[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    date_col: str,
) -> Dict[str, Any]:
    assert_no_forbidden_features(feature_cols, target_col=target_col, date_col=date_col)

    X_train = to_numeric_frame(train_df, feature_cols)
    X_valid = to_numeric_frame(valid_df, feature_cols)
    X_test = to_numeric_frame(test_df, feature_cols)

    X_train, X_valid, X_test, kept_cols, dropped_cols = keep_train_valid_features_only(
        X_train, X_valid, X_test
    )

    assert_no_forbidden_features(kept_cols, target_col=target_col, date_col=date_col)

    y_train = train_df[target_col].astype(int)
    y_valid = valid_df[target_col].astype(int)
    y_test = test_df[target_col].astype(int)

    if y_train.nunique() < 2:
        raise ValueError(f"Le train ne contient qu'une seule classe pour {variant_name}.")

    sample_weight = compute_sample_weights(y_train)

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

    model.fit(X_train, y_train, sample_weight=sample_weight)

    valid_proba = model.predict_proba(X_valid)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    metrics_valid = evaluate_split(variant_name, "validation", y_valid, valid_proba)
    metrics_test = evaluate_split(variant_name, "test", y_test, test_proba)
    metrics_all = pd.concat([metrics_valid, metrics_test], ignore_index=True)

    pred_valid = build_prediction_frame(
        df_subset=valid_df,
        date_col=date_col,
        target_col=target_col,
        proba=valid_proba,
        model_variant=variant_name,
        split_name="validation",
    )
    pred_test = build_prediction_frame(
        df_subset=test_df,
        date_col=date_col,
        target_col=target_col,
        proba=test_proba,
        model_variant=variant_name,
        split_name="test",
    )

    return {
        "feature_cols_kept": kept_cols,
        "feature_cols_dropped_train_only": dropped_cols,
        "metrics": metrics_all,
        "pred_valid": pred_valid,
        "pred_test": pred_test,
    }


def main() -> None:
    require_file(FEATURES_PATH)
    ensure_output_dir()

    print("02_train_point_model.py")
    print(f"Input : {FEATURES_PATH}")

    df = pd.read_csv(FEATURES_PATH, low_memory=False)
    df = normalize_boolean_like_columns(df)

    print(f"Lignes chargées : {len(df)}")
    print(f"Colonnes chargées : {len(df.columns)}")

    df, target_col = find_target_column(df)
    df, date_col = find_date_column(df)

    baseline_requested = list(BASELINE_FEATURE_WHITELIST)
    enriched_requested = list(dict.fromkeys(BASELINE_FEATURE_WHITELIST + ENRICHED_EXTRA_WHITELIST))

    baseline_cols, baseline_missing, baseline_forbidden_requested = resolve_feature_list(
        df, baseline_requested, target_col=target_col, date_col=date_col
    )
    enriched_cols, enriched_missing, enriched_forbidden_requested = resolve_feature_list(
        df, enriched_requested, target_col=target_col, date_col=date_col
    )

    if len(baseline_cols) < 8:
        raise ValueError(
            "Trop peu de features baseline disponibles après filtrage sûr. "
            f"Disponibles : {len(baseline_cols)}"
        )

    if len(enriched_cols) < len(baseline_cols):
        raise ValueError("Le set enrichi ne doit pas être plus petit que le baseline.")

    splits = build_temporal_splits(df, date_col=date_col)
    train_df = splits["train"]
    valid_df = splits["validation"]
    test_df = splits["test"]
    split_meta = splits["meta"]

    print("")
    print("=== SPLIT TEMPOREL ===")
    print(f"Train      : {len(train_df)} lignes")
    print(f"Validation : {len(valid_df)} lignes")
    print(f"Test       : {len(test_df)} lignes")
    print(f"Période train      : {split_meta['train_start']} -> {split_meta['train_end']}")
    print(f"Période validation : {split_meta['validation_start']} -> {split_meta['validation_end']}")
    print(f"Période test       : {split_meta['test_start']} -> {split_meta['test_end']}")

    print("")
    print("=== FEATURES VALIDÉES ===")
    print(f"Baseline demandées : {len(baseline_requested)}")
    print(f"Baseline gardées   : {len(baseline_cols)}")
    print(f"Enrichi demandé    : {len(enriched_requested)}")
    print(f"Enrichi gardé      : {len(enriched_cols)}")

    if baseline_missing:
        print("")
        print("Colonnes baseline absentes (non bloquant) :")
        for c in baseline_missing:
            print(f"- {c}")

    if enriched_missing:
        print("")
        print("Colonnes enrichies absentes (non bloquant) :")
        for c in enriched_missing:
            print(f"- {c}")

    if baseline_forbidden_requested or enriched_forbidden_requested:
        raise ValueError(
            "La whitelist contient une colonne interdite. "
            f"Baseline interdites : {baseline_forbidden_requested} | "
            f"Enrichi interdites : {enriched_forbidden_requested}"
        )

    baseline_result = train_one_variant(
        variant_name="baseline",
        feature_cols=baseline_cols,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        target_col=target_col,
        date_col=date_col,
    )

    enriched_result = train_one_variant(
        variant_name="enrichi",
        feature_cols=enriched_cols,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        target_col=target_col,
        date_col=date_col,
    )

    metrics_baseline_path = OUTPUTS_DIR / "metrics_modele_point_v2.csv"
    metrics_enriched_path = OUTPUTS_DIR / "metrics_modele_point_enrichi_v2.csv"
    pred_valid_baseline_path = OUTPUTS_DIR / "predictions_validation_point_v2.csv"
    pred_test_baseline_path = OUTPUTS_DIR / "predictions_test_point_v2.csv"
    pred_valid_enriched_path = OUTPUTS_DIR / "predictions_validation_point_enrichi_v2.csv"
    pred_test_enriched_path = OUTPUTS_DIR / "predictions_test_point_enrichi_v2.csv"
    summary_path = OUTPUTS_DIR / "02_train_point_model_summary.json"

    baseline_result["metrics"].to_csv(metrics_baseline_path, index=False)
    enriched_result["metrics"].to_csv(metrics_enriched_path, index=False)
    baseline_result["pred_valid"].to_csv(pred_valid_baseline_path, index=False)
    baseline_result["pred_test"].to_csv(pred_test_baseline_path, index=False)
    enriched_result["pred_valid"].to_csv(pred_valid_enriched_path, index=False)
    enriched_result["pred_test"].to_csv(pred_test_enriched_path, index=False)

    summary = {
        "status": "ok",
        "input_file": str(FEATURES_PATH),
        "rows_after_target_and_date_filter": int(len(df)),
        "columns_loaded": int(len(df.columns)),
        "target_column_used": target_col,
        "date_column_used": date_col,
        "split_meta": split_meta,
        "feature_sets": {
            "baseline_requested": int(len(baseline_requested)),
            "enriched_requested": int(len(enriched_requested)),
            "baseline_available_after_safe_filter": int(len(baseline_cols)),
            "enriched_available_after_safe_filter": int(len(enriched_cols)),
            "baseline_kept_train_only": int(len(baseline_result["feature_cols_kept"])),
            "enriched_kept_train_only": int(len(enriched_result["feature_cols_kept"])),
            "baseline_missing": baseline_missing,
            "enriched_missing": enriched_missing,
            "baseline_features": baseline_result["feature_cols_kept"],
            "enriched_features": enriched_result["feature_cols_kept"],
            "baseline_dropped_train_only": baseline_result["feature_cols_dropped_train_only"],
            "enriched_dropped_train_only": enriched_result["feature_cols_dropped_train_only"],
        },
        "forbidden_feature_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "outputs": {
            "metrics_modele_point_v2.csv": str(metrics_baseline_path),
            "metrics_modele_point_enrichi_v2.csv": str(metrics_enriched_path),
            "predictions_validation_point_v2.csv": str(pred_valid_baseline_path),
            "predictions_test_point_v2.csv": str(pred_test_baseline_path),
            "predictions_validation_point_enrichi_v2.csv": str(pred_valid_enriched_path),
            "predictions_test_point_enrichi_v2.csv": str(pred_test_enriched_path),
        },
        "notes": [
            "Validation temporelle utilisée",
            "Aucun split aléatoire",
            "Calibration laissée au script 03",
            "Features choisies via whitelist explicite connue avant match",
            "Colonnes du match courant et cible exclues explicitement",
        ],
    }

    write_json(summary_path, summary)

    print("")
    print("=== SORTIES ===")
    print(f"- {metrics_baseline_path}")
    print(f"- {metrics_enriched_path}")
    print(f"- {pred_valid_baseline_path}")
    print(f"- {pred_test_baseline_path}")
    print(f"- {pred_valid_enriched_path}")
    print(f"- {pred_test_enriched_path}")
    print(f"- {summary_path}")
    print("")
    print("Terminé.")


if __name__ == "__main__":
    main()
