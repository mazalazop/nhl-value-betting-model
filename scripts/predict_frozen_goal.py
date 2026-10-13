"""Score prepared future GOAL features with a trusted frozen local bundle; never fit.

This research-only adapter reuses existing GOAL feature functions when history is
provided. It never fits, selects or promotes a model, or calls old weights current.
"""
import argparse
import hashlib
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'model'))
from henachel.goal import BASE_FEATURES, FEATURE_FAMILIES, numeric_features, augment_goal_features
from henachel.matching import utc


def predict(bundle, future, model_hash, now=None):
    clock = utc(now or pd.Timestamp.now(tz='UTC'))
    allowed = set(BASE_FEATURES + FEATURE_FAMILIES['drought'] + ['team_games_previous4days_pre', 'player_team_changed_pre'])
    features = bundle['features']
    if not features or len(features) != len(set(features)) or set(features)-allowed:
        raise ValueError('Not a frozen GOAL17/C19 feature contract')
    required = ['id_match', 'id_joueur', 'nom', 'team_player_match', 'date_match', 'start_time_utc']
    if set(required)-set(future): raise ValueError('Missing future metadata')
    if future.duplicated(['id_match', 'id_joueur']).any(): raise ValueError('Duplicate future player/game')
    if any(c in future and future[c].notna().any() for c in ['buts', 'points', 'a_marque_un_but', 'a_marque_un_point']):
        raise ValueError('Current results forbidden in future input')
    for value in future.start_time_utc:
        if utc(value) <= clock: raise ValueError('Puck drop passed')
    cutoff = pd.Timestamp(bundle['cutoff'])
    if future.empty: return pd.DataFrame(columns=required+['proba_goal_1p_calibree', 'model_sha256'])
    if cutoff >= pd.to_datetime(future.date_match).min(): raise ValueError('Model cutoff not strictly prior')
    x = numeric_features(future, features)
    raw = bundle['model'].predict_proba(x)[:, 1]
    probabilities = bundle['calibrator'].predict(raw)
    if not np.isfinite(probabilities).all() or ((probabilities<0)|(probabilities>1)).any(): raise ValueError('Invalid GOAL probabilities')
    out = future[required].copy()
    out['proba_goal_1p_calibree'] = probabilities
    out['model_sha256'] = model_hash
    out['model_cutoff'] = str(cutoff.date())
    out['model_status'] = 'research_only_not_production_approved'
    out['generated_at_utc'] = clock.isoformat()
    return out


def prepare(future, history, features):
    # Reuse the existing scientifically tested inference definition; no new feature.
    if any(c in future and future[c].notna().any() for c in ['buts', 'points', 'a_marque_un_but', 'a_marque_un_point']):
        raise ValueError('Current outcomes forbidden')
    if future.empty: return future
    if pd.to_datetime(history.date_match).max() >= pd.to_datetime(future.date_match).min():
        raise ValueError('History must be strictly earlier than the slate')
    result = augment_goal_features(future, prior_history=history)
    if {'team_games_previous4days_pre', 'player_team_changed_pre'} & set(features):
        from research_goal_temporal import augment_research_features
        result = augment_research_features(result, prior_history=augment_goal_features(history))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', required=True, type=Path)
    p.add_argument('--expected-sha256', required=True)
    p.add_argument('--future-features', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--prior-history', type=Path, help='Existing production historical feature file; strictly earlier than slate')
    a = p.parse_args()
    digest = hashlib.sha256(a.bundle.read_bytes()).hexdigest()
    if digest != a.expected_sha256: raise ValueError('Frozen artifact hash changed')
    # joblib is executable serialization: only explicitly trusted local artifacts.
    bundle = joblib.load(a.bundle)
    future = pd.read_json(a.future_features, orient='table') if a.future_features.suffix == '.json' else pd.read_csv(a.future_features)
    if a.prior_history:
        future = prepare(future, pd.read_csv(a.prior_history, low_memory=False), bundle['features'])
    rows = predict(bundle, future, digest)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as f: rows.to_csv(f, index=False)
    print(f'GOAL research only: {len(rows)} predictions; model SHA {digest}')


if __name__ == '__main__': main()
