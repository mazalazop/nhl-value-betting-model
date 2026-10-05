"""Reproduce the locked POINT depth-6/depth-4 comparison; never selects a model.

Outputs must not exist. --final scores the already-consumed final block: it must
not be used as another tuning set. No NHL, bookmaker or Sheets network access.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
import pandas as pd
from henachel.evaluation import metrics
from henachel.manifest import manifest
from henachel.point import POINT_PARAMS

SPEC = importlib.util.spec_from_file_location('point_production', ROOT / 'model/05_predict_upcoming_games.py')
PROD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROD)
BASE_PARAMS = dict(loss='log_loss', learning_rate=.05, max_iter=300, max_depth=6,
                   min_samples_leaf=50, l2_regularization=1., early_stopping=False, random_state=42)


def fit_depth(train, evaluation, target, date, cutoff, depth):
    """Keep the historical fit contract and restore process-global parameters."""
    original = POINT_PARAMS.copy()
    try:
        POINT_PARAMS.clear()
        POINT_PARAMS.update(BASE_PARAMS, max_depth=depth)
        model, calibrator, metadata = PROD.fit_point_model_and_calibrator(train, target, date, cutoff)
        assert pd.Timestamp(metadata['fit_calibration_split']['calib_end']) < cutoff
        matrix = PROD.to_numeric_frame(evaluation, metadata['feature_cols_kept'])
        probabilities = calibrator.predict(model.predict_proba(matrix)[:, 1])
        return probabilities, metadata
    finally:
        POINT_PARAMS.clear()
        POINT_PARAMS.update(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--final', action='store_true', help='Reproduce the locked final comparison, never tune on it')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite an existing experiment')
    PROD.FEATURES_HISTORY_PATH = args.features.resolve()
    frame, target, date = PROD.load_history()
    # Preserve the row orders frozen in the original two experiments.
    if not args.final:
        frame = frame.sort_values([date, 'id_match', 'id_joueur']).reset_index(drop=True)
    dates = sorted(frame[date].unique())
    reserved = int(len(dates) * .85)
    if reserved <= 160:
        raise ValueError('Insufficient history for the frozen 160/60-date protocol')
    final_start = pd.Timestamp(dates[reserved])
    windows = [dates[reserved:]] if args.final else [dates[i:min(i+60, reserved)] for i in range(160, reserved, 60)]
    args.output.mkdir(parents=True)
    reports = []
    scored = []
    for fold, window in enumerate(windows):
        cutoff = pd.Timestamp(window[0])
        train = frame[frame[date] < cutoff]
        evaluation = frame[frame[date].isin(window)]
        rows = evaluation[[date, 'id_match', 'id_joueur', 'game_type', target]].copy()
        rows['fold'] = fold
        for depth in [6, 4]:
            probabilities, metadata = fit_depth(train, evaluation, target, date, cutoff, depth)
            rows[f'depth{depth}'] = probabilities
            strata = {}
            for label, subset in [('all', rows), ('regular', rows[rows.game_type.eq(2)]), ('playoffs', rows[rows.game_type.eq(3)])]:
                if not subset.empty:
                    strata[label] = metrics(subset[target], subset[f'depth{depth}'])
            reports.append(dict(fold=fold, depth=depth, fit=metadata, strata=strata))
        scored.append(rows)
    predictions = pd.concat(scored, ignore_index=True)
    assert not predictions.duplicated(['id_match', 'id_joueur']).any()
    predictions.to_csv(args.output / 'predictions.csv.gz', index=False)
    summary = dict(protocol='frozen 2026-10-03 comparison, not a search', final=args.final,
                   reserved_start=str(final_start), parameters=BASE_PARAMS, features=PROD.FEATURE_WHITELIST,
                   input=str(args.features.resolve()), input_file_sha256=hashlib.sha256(args.features.read_bytes()).hexdigest(),
                   git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   manifest=manifest([args.features], BASE_PARAMS), folds=reports)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(f'{len(predictions)} predictions exported to {args.output}')


if __name__ == '__main__':
    main()
