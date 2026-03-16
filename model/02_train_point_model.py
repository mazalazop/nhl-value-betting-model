#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model/02_train_point_model.py

Objectif
--------
Entraîner un premier modèle POINT robuste à partir de :
- data/final/base_features_context_v2.csv

Sorties
-------
- outputs/metrics_modele_point_v2.csv
- outputs/metrics_modele_point_enrichi_v2.csv
- outputs/predictions_validation_point_v2.csv
- outputs/predictions_test_point_v2.csv
- outputs/predictions_validation_point_enrichi_v2.csv
- outputs/predictions_test_point_enrichi_v2.csv
- outputs/02_train_point_model_summary.json

Principe
--------
- cible = joueur marque au moins 1 point
- validation temporelle obligatoire
- pas de split aléatoire
- exclusion prudente des colonnes de fuite évidente
- baseline = sous-ensemble plus simple de features
- enrichi = ensemble plus large de features numériques sûres

Important
---------
Cette version est volontairement robuste et prudente.
Elle ne prétend pas être une copie exacte du notebook historique.
Elle sert à remettre le pipeline GitHub sur des rails propres, sans fuite de données.
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

LEAKAGE_EXACT_COLUMNS = {
    # identifiants / dates
    "id_match",
    "date_match",
    "date_match_match",
    "date",
    "match_date",
    "game_date",
    "date_game",
    # résultats du match courant / fusion
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
    "status",
    "_merge",
    "match_trouve",
    "team_attendue_match",
    "adversaire_attendu_match",
    "check_team_ok",
    "check_opp_ok",
    # textes / labels bruts
    "team_player_match",
    "adversaire_match",
    "team_name_match",
    "opponent_name_match",
    "nom",
    "player_name",
}

LEAKAGE_KEYWORDS = [
    "target",
    "label",
    "implied_prob",
    "probabilite_implicite",
    "odds",
    "cote",
    "edge",
]

CONTEXT_KEYWORDS = [
    "opponent",
    "adversaire",
    "opp_",
    "team_",
    "is_home",
    "home_road",
    "domicile",
    "exterieur",
    "rest",
    "days_rest",
    "back_to_back",
    "b2b",
    "fatigue",
    "travel",
    "confront",
    "context",
]


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


def is_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return True
    if pd.api.types.is_bool_dtype(series):
        return True
    return False


def has_leakage_keyword(col: str) -> bool:
    col_low = col.lower()
    return any(keyword in col_low for keyword in LEAKAGE_KEYWORDS)


def has_context_keyword(col: str) -> bool:
    col_low = col.lower()
    return any(keyword in col_low for keyword in CONTEXT_KEYWORDS)


def collect_candidate_feature_columns(df: pd.DataFrame, target_col: str, date_col: str) -> List[str]:
    candidate_cols: List[str] = []

    protected_exclusions = set(LEAKAGE_EXACT_COLUMNS)
    protected_exclusions.add(target_col)
    protected_exclusions.add(date_col)

    for col in df.columns:
        if col in protected_exclusions:
            continue
        if has_leakage_keyword(col):
            continue
        if col.startswith("Unnamed:"):
            continue
        if not is_numeric_like(df[col]):
            continue
        candidate_cols.append(col)

    if not candidate_cols:
        raise ValueError("Aucune feature numérique candidate trouvée après filtrage prudent.")

    return candidate_cols


def split_feature_sets(all_feature_cols: List[str]) -> Tuple[List[str], List[str]]:
    enriched = list(all_feature_cols)

    baseline = [
        col for col in all_feature_cols
        if not has_context_keyword(col)
    ]

    # Sécurité : si le baseline devient trop petit, on retombe sur l'ensemble enrichi.
    if len(baseline) < 8:
        baseline = list(enriched)

    return baseline, enriched


def build_temporal_splits(df: pd.DataFrame, date_col: str) -> Dict[str, pd.DataFrame]:
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
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    not_all_nan = ~X_train.isna().all(axis=0)
    variable = X_train.nunique(dropna=True) > 1
    kept_cols = X_train.columns[not_all_nan & variable].tolist()

    if len(kept_cols) < 5:
        raise ValueError(
            f"Trop peu de features utilisables après nettoyage train-only : {len(kept_cols)}"
        )

    return (
        X_train[kept_cols].copy(),
        X_valid[kept_cols].copy(),
        X_test[kept_cols].copy(),
        kept_cols,
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


def precision_and_lift_top_k(y_true: np.ndarray, proba: np.ndarray, top_ratio: float = 0.10) -> Tuple[float, float, int]:
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
    X_train = to_numeric_frame(train_df, feature_cols)
    X_valid = to_numeric_frame(valid_df, feature_cols)
    X_test = to_numeric_frame(test_df, feature_cols)

    X_train, X_valid, X_test, kept_cols = keep_train_valid_features_only(X_train, X_valid, X_test)

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
        "metrics": metrics_all,
        "pred_valid": pred_valid,
        "pred_test": pred_test,
    }


def main() -> None:
    require_file(FEATURES_PATH)
    ensure_output_dir()

    print("02_train_point_model.py")
    print(f"Input : {FEATURES_PATH}")

    df = pd.read_csv(FEATURES_PATH)
    df = normalize_boolean_like_columns(df)

    print(f"Lignes chargées : {len(df)}")
    print(f"Colonnes chargées : {len(df.columns)}")

    df, target_col = find_target_column(df)
    df, date_col = find_date_column(df)

    candidate_feature_cols = collect_candidate_feature_columns(df, target_col=target_col, date_col=date_col)
    baseline_cols, enriched_cols = split_feature_sets(candidate_feature_cols)

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
    print("=== FEATURES ===")
    print(f"Candidats sûrs total : {len(candidate_feature_cols)}")
    print(f"Baseline            : {len(baseline_cols)}")
    print(f"Enrichi             : {len(enriched_cols)}")

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
            "candidate_total": int(len(candidate_feature_cols)),
            "baseline_requested": int(len(baseline_cols)),
            "enriched_requested": int(len(enriched_cols)),
            "baseline_kept_train_only": int(len(baseline_result["feature_cols_kept"])),
            "enriched_kept_train_only": int(len(enriched_result["feature_cols_kept"])),
            "baseline_features": baseline_result["feature_cols_kept"],
            "enriched_features": enriched_result["feature_cols_kept"],
        },
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
            "Version robuste GitHub, pas copie garantie du notebook historique",
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
