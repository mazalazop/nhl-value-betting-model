"""Fixed GOAL17 operational refit. Research only; no feature/model selection."""
import hashlib
import json
import numpy as np
import pandas as pd
from henachel.goal import BASE_FEATURES, FEATURE_FAMILIES, GOAL_PARAMS, augment_goal_features, fit_goal_model, goal_labels, numeric_features
from henachel.data import final_mask, validate_player_games
from henachel.matching import utc

FEATURES = BASE_FEATURES + FEATURE_FAMILIES['drought']
STATUS = 'research_only_operational'
MESSAGE = 'Operational research model; not prospectively production-approved.'
CONFIG_SHA = hashlib.sha256(json.dumps(dict(features=FEATURES, parameters=GOAL_PARAMS, calibration='existing temporal select_calibrator'),sort_keys=True).encode()).hexdigest()


def upcoming_slate(matches,target_date,now=None):
    """NHL PRE is admissible only while the actual scheduled puck drop is future."""
    clock=utc(now or pd.Timestamp.now(tz='UTC'))
    dates=pd.to_datetime(matches.date_match,errors='raise')
    candidates=matches.loc[dates.eq(pd.Timestamp(target_date)) & matches.status.isin(['FUT','PRE'])].copy()
    if candidates.empty:return candidates
    if 'schedule_state' not in candidates:raise ValueError('Missing schedule state')
    starts=candidates.start_time_utc.map(utc)
    return candidates.loc[candidates.schedule_state.eq('OK') & starts.gt(clock)].sort_values(['date_match','id_match'])


def admissible_history(frame, target_date):
    cutoff = pd.Timestamp(target_date)
    if cutoff.tzinfo is not None or cutoff != cutoff.normalize(): raise ValueError('Expected NHL date without time')
    dates = pd.to_datetime(frame.date_match, errors='raise')
    if dates.isna().any(): raise ValueError('Missing history dates')
    if 'status' not in frame: raise ValueError('Final status required')
    # Quarantine current/future rows before reading outcomes or choosing features.
    past = frame.loc[(dates < cutoff) & final_mask(frame)].copy()
    past['date_match'] = dates.loc[past.index]
    if past.empty: raise ValueError('No finalized history strictly before target')
    validate_player_games(past)
    if 'game_type' not in past or not past.game_type.isin([2,3]).all():
        raise ValueError('Explicit regular/playoff history required; preseason not allowed')
    past['a_marque_un_but'] = goal_labels(past)
    return past.sort_values(['date_match','id_match','id_joueur']).reset_index(drop=True)


def refit(frame, target_date):
    past = admissible_history(frame, target_date)
    derived = augment_goal_features(past)
    model, calibrator, fit = fit_goal_model(derived, pd.Timestamp(target_date), FEATURES)
    summary = dict(date_min=str(past.date_match.min().date()),date_max=str(past.date_match.max().date()),
                   observations=len(past),games=int(past.id_match.nunique()),players=int(past.id_joueur.nunique()),
                   seasons=sorted(past.season_source.astype(str).unique().tolist()),positive_rate=float(past.a_marque_un_but.mean()),
                   excluded_rows=len(frame)-len(past),fit=fit,config_sha=CONFIG_SHA,
                   training_cutoff=str(pd.Timestamp(target_date).date()),model_status=STATUS,
                   model_name='GOAL17_OPERATIONAL_REFIT',message=MESSAGE)
    return dict(model=model,calibrator=calibrator,features=fit['feature_cols_kept'],cutoff=target_date,summary=summary),past


def score(bundle, past, future, target_date, now=None):
    clock = utc(now or pd.Timestamp.now(tz='UTC')); cutoff=pd.Timestamp(target_date)
    if not pd.to_datetime(future.date_match).eq(cutoff).all(): raise ValueError('Wrong slate date')
    if future.duplicated(['id_match','id_joueur']).any(): raise ValueError('Duplicate future identity')
    if any(k in future and future[k].notna().any() for k in ['points','buts','a_marque_un_but','a_marque_un_point']):
        raise ValueError('Current outcomes forbidden')
    if any(utc(v)<=clock for v in future.start_time_utc): raise ValueError('Puck drop passed')
    if pd.Timestamp(bundle['cutoff']) != cutoff: raise ValueError('Wrong model cutoff')
    derived=augment_goal_features(future,prior_history=past)
    raw=bundle['model'].predict_proba(numeric_features(derived,bundle['features']))[:,1]
    probability=bundle['calibrator'].predict(raw)
    if not np.isfinite(probability).all() or not ((probability>=0)&(probability<=1)).all(): raise ValueError('Invalid probability')
    out=derived[['date_match','id_match','start_time_utc','id_joueur','nom','team_player_match','adversaire_match','position']].copy()
    out['probability_goal']=probability;out['proba_goal_1p_calibree']=probability
    aliases=dict(date_match='game_date',id_match='game_id',start_time_utc='event_start_utc',id_joueur='player_id',nom='player_name',team_player_match='team',adversaire_match='opponent')
    for original,canonical in aliases.items():out[canonical]=out[original]
    out['run_date']=str(cutoff.date());out['training_cutoff']=str(cutoff.date())
    out['model_status']=STATUS;out['model_name']='GOAL17_OPERATIONAL_REFIT';out['message']=MESSAGE
    out['calibration_method']=bundle['calibrator'].method;out['config_sha']=CONFIG_SHA
    out['generated_at_utc']=clock.isoformat();out['pp_status']='UNKNOWN';out['evidence_status']='UNKNOWN'
    for col in ['toi_moy_5','toi_moy_10','pp_moy_5','pp_moy_10']:out[col]=derived[col] if col in derived else np.nan
    out=out.sort_values(['probability_goal','game_id','player_id'],ascending=[False,True,True]).reset_index(drop=True)
    out['goal_probability_rank']=np.arange(1,len(out)+1)
    return out,derived


def top_ranking(frame, top=10, pp_status=None):
    if top<1: raise ValueError('top must be positive')
    if pp_status not in {None,'PP1','PP2','UNKNOWN'}:raise ValueError('Invalid PP filter')
    chosen=frame
    if pp_status is not None:
        chosen=chosen[chosen.pp_status.eq(pp_status) & chosen.evidence_status.eq('VERIFIED_FRESH')]
    return chosen.sort_values('goal_probability_rank').head(top).copy()
