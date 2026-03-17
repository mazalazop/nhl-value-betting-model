#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model/03_calibrate_point_model.py

Objectif
--------
Calibrer les probabilités du modèle POINT enrichi à partir des sorties de :
- outputs/predictions_validation_point_enrichi_v2.csv
- outputs/predictions_test_point_enrichi_v2.csv

Principe
--------
- on part des prédictions déjà produites par model/02_train_point_model.py
- on crée un split temporel interne dans la validation :
    - calib_fit
    - calib_eval
- on compare 3 méthodes sur calib_eval :
    - raw
    - sigmoid
    - isotonic
- on choisit la meilleure selon :
    - logloss croissant
    - puis brier croissant
- on refit la méthode retenue sur toute la validation
- on applique ensuite au test, sans jamais apprendre sur le test

Sorties
-------
- outputs/comparaison_calibration_point_enrichi_v2.csv
- outputs/selection_calibrateur_point_enrichi_v2.csv
- outputs/predictions_validation_point_enrichi_calibre_v2.csv
- outputs/predictions_test_point_enrichi_calibre_v2.csv
- outputs/03_calibrate_point_model_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

VALIDATION_INPUT_PATH = OUTPUTS_DIR / "predictions_validation_point_enrichi_v2.csv"
TEST_INPUT_PATH = OUTPUTS_DIR / "predictions_test_point_enrichi_v2.csv"

COMPARAISON_OUTPUT_PATH = OUTPUTS_DIR / "comparaison_calibration_point_enrichi_v2.csv"
SELECTION_OUTPUT_PATH = OUTPUTS_DIR / "selection_calibrateur_point_enrichi_v2.csv"
VALIDATION_OUTPUT_PATH = OUTPUTS_DIR / "predictions_validation_point_enrichi_calibre_v2.csv"
TEST_OUTPUT_PATH = OUTPUTS_DIR / "predictions_test_point_enrichi_calibre_v2.csv"
SUMMARY_OUTPUT_PATH = OUTPUTS_DIR / "03_calibrate_point_model_summary.json"

EPSILON = 1e-6
RANDOM_STATE = 42

TARGET_CANDIDATES = [
    "target_point_1p",          # format actuel du script GitHub 02
    "a_marque_un_point",        # alias historique notebook Colab
    "target_point",
    "point_1p",
]

PROBA_CANDIDATES = [
    "proba_point_1p_raw",               # format actuel du script GitHub 02
    "proba_point_modele_enrichi",       # alias historique notebook Colab
    "proba_point",
    "proba",
]

DATE_CANDIDATES = [
    "date_match",
    "date",
    "match_date",
    "game_date",
    "date_game",
]

MATCH_ID_CANDIDATES = [
    "id_match",
    "match_id",
    "game_id",
]

PLAYER_ID_CANDIDATES = [
    "id_joueur",
    "player_id",
    "id_player",
]


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")


def ensure_output_dir() -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str], label: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Aucune colonne {label} trouvée. Colonnes disponibles : {list(df.columns)}")


def first_existing_column_or_none(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def clip_proba(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.clip(arr, EPSILON, 1.0 - EPSILON)


def logit_clip(p: np.ndarray) -> np.ndarray:
    p = clip_proba(p)
    return np.log(p / (1.0 - p))


def calculer_metrics(y_true: np.ndarray, proba: np.ndarray) -> Dict[str, float]:
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


def split_calibration_temporel(
    df: pd.DataFrame,
    date_col: str,
    fit_ratio: float = 0.50,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    temp = df.copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col]).copy()

    if temp.empty:
        raise ValueError("Aucune date exploitable pour la calibration.")

    sort_cols: List[str] = [date_col]
    match_id_col = first_existing_column_or_none(temp, MATCH_ID_CANDIDATES)
    player_id_col = first_existing_column_or_none(temp, PLAYER_ID_CANDIDATES)

    if match_id_col is not None:
        sort_cols.append(match_id_col)
    if player_id_col is not None:
        sort_cols.append(player_id_col)

    temp = temp.sort_values(sort_cols).reset_index(drop=True)

    dates_uniques = sorted(pd.Series(temp[date_col].dt.normalize().unique()).tolist())
    if len(dates_uniques) < 3:
        raise ValueError("Pas assez de dates uniques pour réaliser une calibration temporelle propre.")

    idx_cut = int(np.floor(len(dates_uniques) * fit_ratio))
    idx_cut = max(1, min(idx_cut, len(dates_uniques) - 1))
    date_cut = pd.Timestamp(dates_uniques[idx_cut])

    calib_fit = temp[temp[date_col] < date_cut].copy()
    calib_eval = temp[temp[date_col] >= date_cut].copy()

    if calib_fit.empty or calib_eval.empty:
        raise ValueError("Split de calibration invalide : calib_fit ou calib_eval est vide.")

    return calib_fit, calib_eval, date_cut


def fit_sigmoid_calibrator(proba_fit: np.ndarray, y_fit: np.ndarray) -> LogisticRegression:
    x_fit = logit_clip(proba_fit).reshape(-1, 1)
    y_fit = pd.Series(y_fit).astype(int).to_numpy()

    clf = LogisticRegression(
        C=1e6,
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_STATE,
    )
    clf.fit(x_fit, y_fit)
    return clf


def predict_sigmoid_calibrator(calibrator: LogisticRegression, proba: np.ndarray) -> np.ndarray:
    x = logit_clip(proba).reshape(-1, 1)
    pred = calibrator.predict_proba(x)[:, 1]
    return clip_proba(pred)


def fit_isotonic_calibrator(proba_fit: np.ndarray, y_fit: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
    iso.fit(np.asarray(proba_fit, dtype=float), pd.Series(y_fit).astype(int).to_numpy())
    return iso


def predict_isotonic_calibrator(calibrator: IsotonicRegression, proba: np.ndarray) -> np.ndarray:
    pred = calibrator.predict(np.asarray(proba, dtype=float))
    return clip_proba(pred)


def standardize_prediction_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, str, str]:
    temp = df.copy()

    target_col = first_existing_column(temp, TARGET_CANDIDATES, "cible")
    proba_col = first_existing_column(temp, PROBA_CANDIDATES, "probabilité")
    date_col = first_existing_column(temp, DATE_CANDIDATES, "date")

    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp = temp.dropna(subset=[date_col]).copy()

    temp[target_col] = pd.to_numeric(temp[target_col], errors="coerce")
    temp = temp.dropna(subset=[target_col]).copy()
    temp[target_col] = temp[target_col].astype(int)

    target_values = set(temp[target_col].unique().tolist())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"La colonne cible '{target_col}' n'est pas binaire 0/1 : {sorted(target_values)}")

    temp[proba_col] = pd.to_numeric(temp[proba_col], errors="coerce")
    temp = temp.dropna(subset=[proba_col]).copy()
    temp[proba_col] = clip_proba(temp[proba_col].to_numpy())

    return temp.reset_index(drop=True), target_col, proba_col, date_col


def build_selection_df(
    y_eval: np.ndarray,
    p_eval_raw: np.ndarray,
    p_eval_sigmoid: np.ndarray,
    p_eval_isotonic: np.ndarray,
) -> pd.DataFrame:
    selection_df = pd.DataFrame([
        {"method": "raw", **calculer_metrics(y_eval, p_eval_raw)},
        {"method": "sigmoid", **calculer_metrics(y_eval, p_eval_sigmoid)},
        {"method": "isotonic", **calculer_metrics(y_eval, p_eval_isotonic)},
    ])
    selection_df = selection_df.sort_values(["logloss", "brier"], ascending=[True, True]).reset_index(drop=True)
    return selection_df


def add_calibrated_columns(
    df: pd.DataFrame,
    target_col: str,
    raw_proba_col: str,
    calibrated_proba: np.ndarray,
    chosen_method: str,
) -> pd.DataFrame:
    out = df.copy()
    calibrated_proba = clip_proba(calibrated_proba)

    # Colonnes standardisées utiles pour le pipeline GitHub actuel
    out["target_point_1p"] = out[target_col].astype(int)
    out["proba_point_1p_raw"] = clip_proba(out[raw_proba_col].to_numpy())
    out["proba_point_1p_calibree"] = calibrated_proba
    out["cote_theorique_point_1p_calibree"] = 1.0 / calibrated_proba

    # Alias de compatibilité avec le notebook Colab historique
    out["a_marque_un_point"] = out["target_point_1p"]
    out["proba_point_modele_enrichi"] = out["proba_point_1p_raw"]
    out["proba_point_modele_enrichi_calibree"] = out["proba_point_1p_calibree"]
    out["cote_theorique_point_calibree"] = out["cote_theorique_point_1p_calibree"]

    out["methode_calibration_point"] = chosen_method

    if "rank_proba_sur_date" in out.columns:
        date_col = first_existing_column_or_none(out, DATE_CANDIDATES)
        if date_col is not None:
            out["rank_proba_sur_date_calibree"] = (
                out.groupby(date_col)["proba_point_1p_calibree"]
                .rank(method="first", ascending=False)
                .astype(int)
            )

    return out


def main() -> None:
    require_file(VALIDATION_INPUT_PATH)
    require_file(TEST_INPUT_PATH)
    ensure_output_dir()

    print("03_calibrate_point_model.py")
    print(f"Input validation : {VALIDATION_INPUT_PATH}")
    print(f"Input test       : {TEST_INPUT_PATH}")

    validation_df_raw = pd.read_csv(VALIDATION_INPUT_PATH)
    test_df_raw = pd.read_csv(TEST_INPUT_PATH)

    validation_df, target_col_val, proba_col_val, date_col_val = standardize_prediction_frame(validation_df_raw)
    test_df, target_col_test, proba_col_test, date_col_test = standardize_prediction_frame(test_df_raw)

    print("")
    print("=== COLONNES DETECTEES ===")
    print(f"Validation cible : {target_col_val}")
    print(f"Validation proba : {proba_col_val}")
    print(f"Validation date  : {date_col_val}")
    print(f"Test cible       : {target_col_test}")
    print(f"Test proba       : {proba_col_test}")
    print(f"Test date        : {date_col_test}")

    calib_fit, calib_eval, date_cut = split_calibration_temporel(
        validation_df,
        date_col=date_col_val,
        fit_ratio=0.50,
    )

    print("")
    print("=== SPLIT CALIBRATION INTERNE ===")
    print(f"calib_fit  : {calib_fit.shape}")
    print(f"calib_eval : {calib_eval.shape}")
    print(f"date_cut   : {date_cut.date()}")

    y_fit = calib_fit[target_col_val].to_numpy()
    p_fit = clip_proba(calib_fit[proba_col_val].to_numpy())

    y_eval = calib_eval[target_col_val].to_numpy()
    p_eval_raw = clip_proba(calib_eval[proba_col_val].to_numpy())

    y_val_full = validation_df[target_col_val].to_numpy()
    p_val_full = clip_proba(validation_df[proba_col_val].to_numpy())

    y_test = test_df[target_col_test].to_numpy()
    p_test_raw = clip_proba(test_df[proba_col_test].to_numpy())

    sigmoid_cal = fit_sigmoid_calibrator(p_fit, y_fit)
    isotonic_cal = fit_isotonic_calibrator(p_fit, y_fit)

    p_eval_sigmoid = predict_sigmoid_calibrator(sigmoid_cal, p_eval_raw)
    p_eval_isotonic = predict_isotonic_calibrator(isotonic_cal, p_eval_raw)

    selection_df = build_selection_df(
        y_eval=y_eval,
        p_eval_raw=p_eval_raw,
        p_eval_sigmoid=p_eval_sigmoid,
        p_eval_isotonic=p_eval_isotonic,
    )

    chosen_method = str(selection_df.iloc[0]["method"])

    print("")
    print("=== SELECTION CALIBRATEUR SUR CALIB_EVAL ===")
    print(selection_df)
    print("")
    print(f"Méthode choisie : {chosen_method}")

    if chosen_method == "sigmoid":
        calibrator_final = fit_sigmoid_calibrator(p_val_full, y_val_full)
        p_val_cal = predict_sigmoid_calibrator(calibrator_final, p_val_full)
        p_test_cal = predict_sigmoid_calibrator(calibrator_final, p_test_raw)
    elif chosen_method == "isotonic":
        calibrator_final = fit_isotonic_calibrator(p_val_full, y_val_full)
        p_val_cal = predict_isotonic_calibrator(calibrator_final, p_val_full)
        p_test_cal = predict_isotonic_calibrator(calibrator_final, p_test_raw)
    else:
        p_val_cal = p_val_full.copy()
        p_test_cal = p_test_raw.copy()

    metrics_val_raw = calculer_metrics(y_val_full, p_val_full)
    metrics_val_cal = calculer_metrics(y_val_full, p_val_cal)
    metrics_test_raw = calculer_metrics(y_test, p_test_raw)
    metrics_test_cal = calculer_metrics(y_test, p_test_cal)

    comparaison_df = pd.DataFrame([
        {"split": "validation_raw", "method_applied": "raw", **metrics_val_raw},
        {"split": "validation_calibrated", "method_applied": chosen_method, **metrics_val_cal},
        {"split": "test_raw", "method_applied": "raw", **metrics_test_raw},
        {"split": "test_calibrated", "method_applied": chosen_method, **metrics_test_cal},
    ])

    validation_out = add_calibrated_columns(
        df=validation_df,
        target_col=target_col_val,
        raw_proba_col=proba_col_val,
        calibrated_proba=p_val_cal,
        chosen_method=chosen_method,
    )
    test_out = add_calibrated_columns(
        df=test_df,
        target_col=target_col_test,
        raw_proba_col=proba_col_test,
        calibrated_proba=p_test_cal,
        chosen_method=chosen_method,
    )

    comparaison_df.to_csv(COMPARAISON_OUTPUT_PATH, index=False)
    selection_df.to_csv(SELECTION_OUTPUT_PATH, index=False)
    validation_out.to_csv(VALIDATION_OUTPUT_PATH, index=False)
    test_out.to_csv(TEST_OUTPUT_PATH, index=False)

    summary = {
        "status": "ok",
        "inputs": {
            "validation": str(VALIDATION_INPUT_PATH),
            "test": str(TEST_INPUT_PATH),
        },
        "detected_columns": {
            "validation": {
                "target": target_col_val,
                "proba": proba_col_val,
                "date": date_col_val,
            },
            "test": {
                "target": target_col_test,
                "proba": proba_col_test,
                "date": date_col_test,
            },
        },
        "calibration": {
            "fit_ratio": 0.50,
            "date_cut": str(date_cut.date()),
            "method_selected_on_calib_eval": chosen_method,
        },
        "row_counts": {
            "validation_loaded": int(len(validation_df_raw)),
            "validation_used_after_cleaning": int(len(validation_df)),
            "test_loaded": int(len(test_df_raw)),
            "test_used_after_cleaning": int(len(test_df)),
            "calib_fit": int(len(calib_fit)),
            "calib_eval": int(len(calib_eval)),
        },
        "outputs": {
            "comparaison": str(COMPARAISON_OUTPUT_PATH),
            "selection": str(SELECTION_OUTPUT_PATH),
            "validation_calibrated": str(VALIDATION_OUTPUT_PATH),
            "test_calibrated": str(TEST_OUTPUT_PATH),
        },
        "notes": [
            "Aucun apprentissage de calibration sur le test",
            "Choix du calibrateur sur un split temporel interne de la validation",
            "Compatibilité maintenue avec les noms de colonnes du notebook historique et du script GitHub actuel",
        ],
    }
    write_json(SUMMARY_OUTPUT_PATH, summary)

    print("")
    print("=== METRICS VALIDATION RAW ===")
    for k, v in metrics_val_raw.items():
        print(f"{k}: {v:.6f}" if pd.notna(v) else f"{k}: nan")

    print("")
    print("=== METRICS VALIDATION CALIBRE ===")
    for k, v in metrics_val_cal.items():
        print(f"{k}: {v:.6f}" if pd.notna(v) else f"{k}: nan")

    print("")
    print("=== METRICS TEST RAW ===")
    for k, v in metrics_test_raw.items():
        print(f"{k}: {v:.6f}" if pd.notna(v) else f"{k}: nan")

    print("")
    print("=== METRICS TEST CALIBRE ===")
    for k, v in metrics_test_cal.items():
        print(f"{k}: {v:.6f}" if pd.notna(v) else f"{k}: nan")

    print("")
    print("=== SORTIES ===")
    print(f"- {COMPARAISON_OUTPUT_PATH}")
    print(f"- {SELECTION_OUTPUT_PATH}")
    print(f"- {VALIDATION_OUTPUT_PATH}")
    print(f"- {TEST_OUTPUT_PATH}")
    print(f"- {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
