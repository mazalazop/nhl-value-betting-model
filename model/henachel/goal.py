"""Separate GOAL 1+ research model; POINT configuration is never mutated."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from henachel.calibration import select_calibrator
from henachel.data import final_mask, validate_player_games

GOAL_PARAMS = dict(loss='log_loss', learning_rate=.05, max_iter=300, max_depth=4,
                   min_samples_leaf=50, l2_regularization=1., early_stopping=False, random_state=42)
BASE_FEATURES = ['is_home_player','nb_matchs_avant_match','jours_repos',
                 'tirs_moy_5','tirs_moy_10','tirs_par_60_5','toi_moy_5','toi_moy_10','pp_moy_5',
                 'buts_moy_5','buts_moy_10','buts_par_60_5',
                 'goal_hit_rate_last_10','goal_hit_rate_season_pre','goal_hit_rate_prev_season']
FEATURE_FAMILIES = {
    'conversion': ['goal_conversion_10_pre','goal_shots_trend_pre'],
    'usage': ['goal_toi_trend_pre','goal_pp_share_pre'],
    'context': ['team_gf_moy_5','team_ga_moy_5','team_back_to_back','jours_repos_team','goal_opponent_ga_5_pre'],
    'return': ['absence_longue_flag','matchs_depuis_retour_avant_match','ratio_toi_retour_vs_pre_absence','ratio_pp_retour_vs_pre_absence'],
    'drought': ['goal_drought_pre','goal_relative_drought_pre'],
}


def goal_labels(frame):
    if 'buts' not in frame:
        raise ValueError('Missing goal target source: buts')
    goals = pd.to_numeric(frame.buts, errors='coerce')
    if goals.isna().any() or not np.isfinite(goals).all() or (goals < 0).any() or (goals % 1 != 0).any():
        raise ValueError('Invalid goal counts')
    labels = goals.ge(1).astype(int)
    if 'a_marque_un_but' in frame:
        supplied = pd.to_numeric(frame.a_marque_un_but, errors='coerce')
        if not supplied.isin([0, 1]).all() or not supplied.eq(labels).all():
            raise ValueError('Goal label disagrees with buts')
    return labels


def load_goal_history(path):
    frame = pd.read_csv(path, low_memory=False)
    validate_player_games(frame)
    if not final_mask(frame).all():
        raise ValueError('GOAL history contains non-final games')
    if 'game_type' not in frame or not frame.game_type.isin([2,3]).all():
        raise ValueError('GOAL universe requires explicit regular/playoff game types; no silent preseason filtering')
    frame['a_marque_un_but'] = goal_labels(frame)
    frame['date_match'] = pd.to_datetime(frame.date_match, errors='raise')
    return frame.sort_values(['date_match','id_match','id_joueur']).reset_index(drop=True)


def augment_goal_features(frame, prior_history=None):
    """Derived features consume only pregame columns or strictly prior goals.

    For inference, pass finalized prior_history and upcoming feature rows with
    unknown outcomes. A future row never advances the observed drought.
    """
    current = frame.copy(); current['_goal_requested'] = True
    if prior_history is not None:
        past = prior_history.copy(); past['_goal_requested'] = False
        if not past.empty and pd.to_datetime(past.date_match).max() >= pd.to_datetime(current.date_match).min():
            raise ValueError('Future GOAL features require strictly earlier history')
        current['buts'] = np.nan
        work = pd.concat([past, current], ignore_index=True)
    else:
        work = current.reset_index(drop=True)
    if work.duplicated(['id_match','id_joueur']).any():
        raise ValueError('Duplicate GOAL player/game identities')
    work['date_match'] = pd.to_datetime(work.date_match, errors='raise')
    work = work.sort_values(['id_joueur','date_match','id_match']).copy()
    work['goal_conversion_10_pre'] = work.buts_moy_10 / work.tirs_moy_10.where(work.tirs_moy_10 > 0)
    work['goal_shots_trend_pre'] = work.tirs_moy_5 - work.tirs_moy_10
    work['goal_toi_trend_pre'] = work.toi_moy_5 - work.toi_moy_10
    work['goal_pp_share_pre'] = work.pp_moy_5 / work.toi_moy_5.where(work.toi_moy_5 > 0)
    drought = pd.Series(index=work.index, dtype=float)
    for _, group in work.groupby(['id_joueur','season_source'], sort=False):
        count = 0
        for index, goals in zip(group.index, group.buts):
            drought.loc[index] = count
            if pd.notna(goals):
                count = 0 if goals >= 1 else count + 1
    work['goal_drought_pre'] = drought
    p = work.goal_hit_rate_season_pre.where(work.goal_hit_rate_season_pre.between(0,1,inclusive='neither'))
    work['goal_relative_drought_pre'] = drought * p / (1-p)
    keys = ['id_match','team_player_match']
    if work.groupby(keys).team_ga_moy_5.nunique(dropna=False).gt(1).any():
        raise ValueError('Inconsistent pregame team defence')
    opponents = work.drop_duplicates(keys)[keys+['team_ga_moy_5']].rename(
        columns={'team_player_match':'adversaire_match','team_ga_moy_5':'goal_opponent_ga_5_pre'})
    work = work.drop(columns='goal_opponent_ga_5_pre', errors='ignore').merge(
        opponents, on=['id_match','adversaire_match'], how='left', validate='many_to_one')
    return work[work._goal_requested].drop(columns='_goal_requested').sort_values(
        ['date_match','id_match','id_joueur']).reset_index(drop=True)


def numeric_features(frame, columns):
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f'Missing GOAL features: {sorted(missing)}')
    return frame[columns].apply(pd.to_numeric, errors='raise').replace([np.inf,-np.inf], np.nan)


def fit_goal_model(history, cutoff, features=None, parameters=None):
    features = list(BASE_FEATURES if features is None else features)
    allowed = set(BASE_FEATURES + sum(FEATURE_FAMILIES.values(), []) + ['goal_opponent_pk_10_pre'])
    if not features or len(set(features)) != len(features) or set(features) - allowed:
        raise ValueError('GOAL feature list must contain only approved pregame fields')
    cutoff = pd.Timestamp(cutoff)
    past = history[pd.to_datetime(history.date_match) < cutoff].copy()
    past['date_match'] = pd.to_datetime(past.date_match)
    past = past.sort_values(['date_match','id_match','id_joueur']).reset_index(drop=True)
    past['a_marque_un_but'] = goal_labels(past)
    dates = sorted(past.date_match.unique())
    if len(dates) < 20:
        raise ValueError('Insufficient historical dates for GOAL calibration')
    cut = dates[min(len(dates)-1, max(1,int(len(dates)*.85)))]
    train = past[past.date_match < cut]; calibration = past[past.date_match >= cut]
    x = numeric_features(train, features)
    kept = x.columns[x.notna().any() & x.nunique(dropna=True).gt(1)].tolist()
    if not kept:
        raise ValueError('No variable GOAL training features')
    y = train.a_marque_un_but
    if y.nunique() != 2:
        raise ValueError('GOAL training needs both classes')
    params = GOAL_PARAMS.copy()
    if parameters:
        if set(parameters) - {'max_depth','min_samples_leaf'}:
            raise ValueError('Only preregistered GOAL complexity overrides are supported')
        params.update(parameters)
    model = HistGradientBoostingClassifier(**params)
    weights = np.where(y.eq(1), y.eq(0).sum()/y.eq(1).sum(), 1.)
    model.fit(x[kept], y, sample_weight=weights)
    raw = model.predict_proba(numeric_features(calibration, kept))[:,1]
    cal, info = select_calibrator(raw, calibration.a_marque_un_but, calibration.date_match)
    meta = dict(target='GOAL_1_PLUS', parameters=params, feature_cols_kept=kept,
                feature_cols_dropped_train_only=[c for c in features if c not in kept],
                fit_rows=len(train), calib_rows=len(calibration), fit_start=str(train.date_match.min()),
                fit_end=str(train.date_match.max()), calib_start=str(calibration.date_match.min()),
                calib_end=str(calibration.date_match.max()), calibration=info)
    return model, cal, meta
