#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model/10_compare_point_model_to_baseline.py

Objectif
--------
Comparer les métriques du modèle POINT actuellement relancé
(issues de outputs/comparaison_calibration_point_enrichi_v2.csv)
à la baseline validée du projet.

Baseline officielle retenue
--------------------------
Référence projet = modèle POINT enrichi + calibration sigmoid.

Validation brut :
- brier = 0.222318
- logloss = 0.636226
- auc = 0.686842
- avg_precision = 0.523736

Test brut :
- brier = 0.224014
- logloss = 0.640154
- auc = 0.678702
- avg_precision = 0.522698

Validation calibré :
- brier = 0.202690
- logloss = 0.592299
- auc = 0.686842
- avg_precision = 0.523736

Test calibré :
- brier = 0.205518
- logloss = 0.599639
- auc = 0.678702
- avg_precision = 0.522698

Sorties
-------
- outputs/10_compare_point_model_to_baseline.csv
- outputs/10_compare_point_model_to_baseline.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

INPUT_PATH = OUTPUTS_DIR / "comparaison_calibration_point_enrichi_v2.csv"
CSV_OUTPUT_PATH = OUTPUTS_DIR / "10_compare_point_model_to_baseline.csv"
JSON_OUTPUT_PATH = OUTPUTS_DIR / "10_compare_point_model_to_baseline.json"

BASELINE = {
    "validation_raw": {
        "brier": 0.222318,
        "logloss": 0.636226,
        "auc": 0.686842,
        "avg_precision": 0.523736,
    },
    "test_raw": {
        "brier": 0.224014,
        "logloss": 0.640154,
        "auc": 0.678702,
        "avg_precision": 0.522698,
    },
    "validation_calibrated": {
        "brier": 0.202690,
        "logloss": 0.592299,
        "auc": 0.686842,
        "avg_precision": 0.523736,
    },
    "test_calibrated": {
        "brier": 0.205518,
        "logloss": 0.599639,
        "auc": 0.678702,
        "avg_precision": 0.522698,
    },
}

LOWER_IS_BETTER = {"logloss", "brier"}
HIGHER_IS_BETTER = {"auc", "avg_precision"}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")


def load_current_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"split", "method_applied", "logloss", "brier", "auc", "avg_precision"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {sorted(missing)}")
    return df.copy()


def judge(metric: str, current: float, baseline: float, tol: float = 1e-12) -> str:
    if pd.isna(current) or pd.isna(baseline):
        return "na"
    delta = current - baseline
    if metric in LOWER_IS_BETTER:
        if delta < -tol:
            return "better"
        if delta > tol:
            return "worse"
        return "equal"
    if metric in HIGHER_IS_BETTER:
        if delta > tol:
            return "better"
        if delta < -tol:
            return "worse"
        return "equal"
    return "na"


def summarize_split(rows: pd.DataFrame, split_name: str) -> Dict[str, str]:
    split_rows = rows[rows["split"] == split_name].copy()
    if split_rows.empty:
        return {"split": split_name, "status": "missing"}

    judgments = split_rows["judgment"].tolist()
    better = judgments.count("better")
    worse = judgments.count("worse")

    if worse == 0 and better > 0:
        status = "improved"
    elif better == 0 and worse > 0:
        status = "degraded"
    elif better == 0 and worse == 0:
        status = "unchanged"
    else:
        status = "mixed"

    return {
        "split": split_name,
        "status": status,
        "better_metrics": better,
        "worse_metrics": worse,
        "equal_metrics": judgments.count("equal"),
    }


def main() -> None:
    require_file(INPUT_PATH)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    current_df = load_current_metrics(INPUT_PATH)

    rows: List[Dict] = []
    for split_name, metrics in BASELINE.items():
        current_split = current_df[current_df["split"] == split_name]
        if current_split.empty:
            for metric_name, baseline_value in metrics.items():
                rows.append({
                    "split": split_name,
                    "metric": metric_name,
                    "baseline_value": baseline_value,
                    "current_value": pd.NA,
                    "delta_current_minus_baseline": pd.NA,
                    "judgment": "missing_current_split",
                    "method_applied": pd.NA,
                })
            continue

        row = current_split.iloc[0]
        method_applied = row.get("method_applied", pd.NA)
        for metric_name, baseline_value in metrics.items():
            current_value = pd.to_numeric(pd.Series([row.get(metric_name)]), errors="coerce").iloc[0]
            delta = current_value - baseline_value if pd.notna(current_value) else pd.NA
            rows.append({
                "split": split_name,
                "metric": metric_name,
                "baseline_value": baseline_value,
                "current_value": current_value,
                "delta_current_minus_baseline": delta,
                "judgment": judge(metric_name, current_value, baseline_value),
                "method_applied": method_applied,
            })

    compare_df = pd.DataFrame(rows)
    compare_df.to_csv(CSV_OUTPUT_PATH, index=False)

    split_summaries = [
        summarize_split(compare_df, "validation_raw"),
        summarize_split(compare_df, "validation_calibrated"),
        summarize_split(compare_df, "test_raw"),
        summarize_split(compare_df, "test_calibrated"),
    ]

    overall_test_calibrated = next((x for x in split_summaries if x["split"] == "test_calibrated"), None)
    if overall_test_calibrated is None:
        overall_status = "missing_test_calibrated"
    else:
        overall_status = overall_test_calibrated["status"]

    payload = {
        "status": "ok",
        "baseline_reference": "point enrichi + calibration sigmoid validee dans les memos projet",
        "input_metrics_file": str(INPUT_PATH),
        "outputs": {
            "csv": str(CSV_OUTPUT_PATH),
            "json": str(JSON_OUTPUT_PATH),
        },
        "split_summaries": split_summaries,
        "overall_focus_split": "test_calibrated",
        "overall_focus_status": overall_status,
        "recommendation": (
            "keep_new_version" if overall_status == "improved" else
            "review_new_version" if overall_status == "mixed" else
            "rollback_or_debug"
        ),
    }

    JSON_OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("10_compare_point_model_to_baseline.py")
    print(f"Input  : {INPUT_PATH}")
    print(f"Output : {CSV_OUTPUT_PATH}")
    print(f"Output : {JSON_OUTPUT_PATH}")
    print("")
    print("=== SPLIT SUMMARIES ===")
    for item in split_summaries:
        print(item)
    print("")
    print(f"FOCUS test_calibrated : {overall_status}")
    print(f"Recommendation        : {payload['recommendation']}")


if __name__ == "__main__":
    main()
