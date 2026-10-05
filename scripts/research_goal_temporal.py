"""Immutable development-only GOAL study on additional seasons. No production entry."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'model'))
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from henachel.goal import BASE_FEATURES, FEATURE_FAMILIES, GOAL_PARAMS, augment_goal_features, goal_labels, numeric_features
from henachel.calibration import select_calibrator
from henachel.evaluation import metrics, reliability_bins
from henachel.goal_experiment import paired_comparison
from henachel.data import final_mask, validate_player_games
from henachel.manifest import manifest

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / 'docs/research_20261005/protocol.json'
BASELINE = BASE_FEATURES + FEATURE_FAMILIES['drought']
FAMILIES = {
    'short_volume': ['shots_mean3_pre', 'shots_cv10_pre'],
    'interactions': ['shots_pp_interaction_pre', 'drought_shots_interaction_pre'],
    'congestion': ['team_games_previous4days_pre', 'player_team_changed_pre'],
}
FOLDS = [('2023-01-01', '2023-03-01'), ('2023-03-01', '2023-04-20'),
         ('2023-10-01', '2023-12-01'), ('2023-12-01', '2024-02-01')]
PROTOCOL = dict(target='GOAL_1_PLUS', status='research_only', seasons=[20222023, 20232024],
                development_end='2024-02-01', folds=FOLDS, baseline=BASELINE, families=FAMILIES,
                parameters=GOAL_PARAMS, calibration='unchanged temporal auto selection',
                retrospective_holdout=['2024-02-01', '2024-04-20'], holdout_access=False,
                prospective_confirmation='2026-27 still required; older holdout is retrospective',
                selection='all folds negative logloss/Brier deltas, Bonferroni 3 CI upper < 0, ECE delta <= 0.005',
                smoothing_prior=10, seed=42, no_complexity_search=True)


def frozen_protocol():
    expected = json.loads(json.dumps(PROTOCOL))
    actual = json.loads(PROTOCOL_PATH.read_text())
    if actual != expected:
        raise ValueError('Frozen research protocol differs from implementation')
    return actual


def development_only(frame):
    """Remove quarantined dates before feature construction or outcome validation."""
    frame = frame.reset_index(drop=True)
    dates = pd.to_datetime(frame.date_match, errors='raise')
    seasons = pd.to_numeric(frame.season_source, errors='raise')
    if not seasons.isin(PROTOCOL['seasons']).all():
        raise ValueError('Only additional 2022-23 and 2023-24 seasons allowed')
    result = frame.loc[dates < pd.Timestamp(PROTOCOL['development_end'])].copy()
    result['date_match'] = dates.loc[result.index]
    if not result.game_type.eq(2).all():
        raise ValueError('Research universe requires regular season only')
    if result.duplicated(['id_match', 'id_joueur']).any():
        raise ValueError('Duplicate player/game')
    return result.sort_values(['date_match', 'id_match', 'id_joueur']).reset_index(drop=True)


def augment_research_features(frame, prior_history=None):
    current = frame.copy()
    current['_requested_research'] = True
    if prior_history is not None:
        past = prior_history.copy()
        if not past.empty and pd.to_datetime(past.date_match).max() >= pd.to_datetime(current.date_match).min():
            raise ValueError('History must be strictly prior')
        past['_requested_research'] = False
        current['tirs'] = np.nan
        work = pd.concat([past, current], ignore_index=True)
    else:
        work = current.reset_index(drop=True)
    if work.duplicated(['id_match', 'id_joueur']).any():
        raise ValueError('Duplicate player/game')
    work.date_match = pd.to_datetime(work.date_match, errors='raise')
    work = work.sort_values(['id_joueur', 'date_match', 'id_match']).copy()
    shots = pd.to_numeric(work.tirs, errors='raise')
    if (shots.dropna() < 0).any() or np.isinf(shots).any():
        raise ValueError('Invalid shots')
    work['tirs'] = shots
    groups = work.groupby('id_joueur', sort=False)
    work['shots_mean3_pre'] = groups.tirs.transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    mean = groups.tirs.transform(lambda x: x.shift(1).rolling(10, min_periods=5).mean())
    sd = groups.tirs.transform(lambda x: x.shift(1).rolling(10, min_periods=5).std())
    work['shots_cv10_pre'] = sd / mean.where(mean > 0)
    previous = groups.team_player_match.shift(1)
    work['player_team_changed_pre'] = work.team_player_match.ne(previous).astype(float).where(previous.notna() & work.team_player_match.notna())
    team_games = pd.concat([
        work[['date_match', 'id_match', 'team_player_match']].rename(columns={'team_player_match': 'team'}),
        work[['date_match', 'id_match', 'adversaire_match']].rename(columns={'adversaire_match': 'team'})
    ]).drop_duplicates()
    if team_games.duplicated(['id_match', 'team']).any():
        raise ValueError('Inconsistent game dates')
    counts = {}
    for team, games in team_games.dropna(subset=['team']).groupby('team'):
        dates = games.date_match.sort_values().to_numpy()
        for date in games.date_match.unique():
            counts[(team, pd.Timestamp(date))] = int(((dates < date) & (dates >= date - np.timedelta64(4, 'D'))).sum())
    work['team_games_previous4days_pre'] = [counts.get((team, date), np.nan) for team, date in zip(work.team_player_match, work.date_match)]
    work['shots_pp_interaction_pre'] = work.tirs_moy_5 * work.pp_moy_5
    work['drought_shots_interaction_pre'] = work.goal_relative_drought_pre * work.tirs_moy_5
    return work[work._requested_research].drop(columns='_requested_research').sort_values(['date_match', 'id_match', 'id_joueur']).reset_index(drop=True)


def fit(history, cutoff, features):
    allowed = BASELINE + sum(FAMILIES.values(), [])
    if not features or len(features) != len(set(features)) or set(features) - set(allowed):
        raise ValueError('Undeclared research features')
    past = history[pd.to_datetime(history.date_match) < pd.Timestamp(cutoff)].copy()
    if pd.Timestamp(cutoff) > pd.Timestamp(PROTOCOL['development_end']):
        raise ValueError('Holdout inaccessible')
    past = development_only(past)
    past['a_marque_un_but'] = goal_labels(past)
    dates = sorted(past.date_match.unique())
    if len(dates) < 20:
        raise ValueError('Insufficient training dates')
    cut = dates[int(.85 * len(dates))]
    train, cal = past[past.date_match < cut], past[past.date_match >= cut]
    x = numeric_features(train, features)
    kept = x.columns[x.notna().any() & x.nunique().gt(1)].tolist()
    y = train.a_marque_un_but
    if not kept or y.nunique() != 2:
        raise ValueError('Insufficient variable features/classes')
    model = HistGradientBoostingClassifier(**GOAL_PARAMS)
    model.fit(x[kept], y, sample_weight=np.where(y.eq(1), y.eq(0).sum()/y.eq(1).sum(), 1.))
    raw = model.predict_proba(numeric_features(cal, kept))[:, 1]
    calibrator, info = select_calibrator(raw, cal.a_marque_un_but, cal.date_match)
    return model, calibrator, dict(kept=kept, fit_end=str(train.date_match.max()), calibration_end=str(cal.date_match.max()), calibration=info)


def summary(rows):
    bins = reliability_bins(rows.a_marque_un_but, rows.probability)
    return dict(metrics=metrics(rows.a_marque_un_but, rows.probability), calibration=bins,
                ece=sum(b['count'] * abs(b['mean_probability'] - b['observed_rate']) for b in bins if b['count']) / len(rows))


def run(source, output):
    frozen_protocol()
    if output.exists():
        raise FileExistsError('Immutable experiment directory already exists')
    frame = development_only(pd.read_csv(source, low_memory=False))
    validate_player_games(frame)
    if not final_mask(frame).all():
        raise ValueError('Research history requires explicit final status')
    frame = augment_research_features(augment_goal_features(frame))
    frame['a_marque_un_but'] = goal_labels(frame)
    output.mkdir(parents=True)
    results, reports = {}, {}
    for name, extra in [('baseline', [])] + list(FAMILIES.items()):
        predictions, reports[name] = [], {'folds': []}
        for fold, (start, end) in enumerate(FOLDS):
            train = frame[frame.date_match < start]
            test = frame[(frame.date_match >= start) & (frame.date_match < end)]
            if test.empty:
                raise ValueError(f'Empty preregistered fold {fold}')
            model, cal, meta = fit(train, start, BASELINE + extra)
            rows = test[['date_match', 'id_match', 'id_joueur', 'game_type', 'a_marque_un_but']].copy()
            rows['fold'] = fold
            rows['probability'] = cal.predict(model.predict_proba(numeric_features(test, meta['kept']))[:, 1])
            prevalence = float(train.a_marque_un_but.mean())
            rows['prevalence_baseline'] = prevalence
            rows['player_smoothed_baseline'] = (test.season_goal_hits_before_match + 10 * prevalence) / (test.season_games_before_match + 10)
            reports[name]['folds'].append(dict(fold=fold, fit=meta, **summary(rows)))
            predictions.append(rows)
        rows = pd.concat(predictions, ignore_index=True)
        if rows.duplicated(['id_match', 'id_joueur']).any():
            raise ValueError('Duplicate OOF identities')
        rows.to_csv(output / f'{name}.csv.gz', index=False)
        results[name] = rows
        reports[name].update(summary(rows))
        reports[name]['benchmarks'] = {column: metrics(rows.a_marque_un_but, rows[column]) for column in ['prevalence_baseline', 'player_smoothed_baseline']}
    comparisons = {}
    for name in FAMILIES:
        comparison = paired_comparison(results['baseline'], results[name], family_size=3)
        comparison['development_only_selected'] = comparison['passes'] and reports[name]['ece'] <= reports['baseline']['ece'] + .005
        benchmark = results[name].copy()
        benchmark['probability'] = benchmark.player_smoothed_baseline
        comparison['smoothed_benchmark_diagnostic'] = paired_comparison(benchmark, results[name], family_size=1)
        comparisons[name] = comparison
    report = dict(protocol=PROTOCOL, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  protocol_sha256=hashlib.sha256(PROTOCOL_PATH.read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), reports=reports,
                  comparisons=comparisons, production_approved=False, holdout_used=False,
                  manifest=manifest([source, PROTOCOL_PATH, Path(__file__)], GOAL_PARAMS))
    (output / 'summary.json').write_text(json.dumps(report, indent=2, default=str, allow_nan=False))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.features, args.output)
