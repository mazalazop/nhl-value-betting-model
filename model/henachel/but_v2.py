"""Independent BUT research model. Never imported by POINT or GOAL V1."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from henachel.goal import goal_labels
from henachel.calibration import select_calibrator

KEYS = ['id_match', 'id_joueur']
FAMILIES = {
    'volume': [f'shots_mean_{w}' for w in (3, 5, 10)] + [f'shots60_{w}' for w in (3, 5, 10)],
    'usage': [f'{x}_mean_{w}' for x in ('toi', 'pp') for w in (3, 5, 10)],
    'offense': ['goals_mean_10', 'conversion_10', 'prior_hit_rate'],
    'team_context': ['team_goals_5', 'opponent_goals_against_5'],
    'calendar': ['rest_days', 'home'],
    'drought': ['drought'],
    'hot_streak': ['hot_streak'],
}
FEATURES = sum(FAMILIES.values(), [])

def config(path=None):
    return json.loads(Path(path or Path(__file__).resolve().parents[2] / 'configs/but_v2.json').read_text())

def validate(frame, outcomes=True):
    required=KEYS+['date_match','season_source','team_player_match']
    if set(required)-set(frame):raise ValueError('Missing identities/date/season/team')
    if frame[required].isna().any().any() or frame.duplicated(KEYS).any():
        raise ValueError('Missing or duplicate identity')
    dates=pd.to_datetime(frame.date_match,errors='raise')
    if dates.isna().any() or not dates.eq(dates.dt.normalize()).all():raise ValueError('Expected NHL dates')
    if frame.assign(_date=dates).duplicated(['id_joueur','_date']).any():
        raise ValueError('Multiple player games same date: ordering not proven')
    if outcomes:
        if not frame.status.isin(['FINAL','OFF']).all():raise ValueError('Final outcomes required')
        if not frame.game_type.isin([2,3]).all():raise ValueError('Regular/playoff only')
        goal_labels(frame)
    return dates

def build_features(frame):
    """Reconstruct from raw observations, excluding each target date.

    Historical replay only: original availability timestamps are not inferred.
    Unknown measurements remain NaN. Rolling windows reset each season.
    Future rows must have empty outcome/measurement cells and be after history.
    """
    dates=validate(frame,outcomes=False)
    f=frame.copy();f['date_match']=dates
    f=f.sort_values(['id_joueur','date_match','id_match']).reset_index(drop=True)
    for col in ['buts','tirs','temps_de_glace','temps_pp']:
        f[col]=pd.to_numeric(f[col],errors='raise')
        if ((f[col].dropna()<0) | ~np.isfinite(f[col].dropna())).any():raise ValueError('Invalid measurement')
    f['hit']=f.buts.ge(1).where(f.buts.notna()).astype(float)
    grouping=['id_joueur','season_source']
    g=f.groupby(grouping,sort=False)
    for source,label in [('tirs','shots'),('temps_de_glace','toi'),('temps_pp','pp'),('buts','goals')]:
        for w in (3,5,10):
            f[f'{label}_mean_{w}']=g[source].transform(lambda s:s.shift().rolling(w,min_periods=1).mean())
    for w in (3,5,10):
        f[f'shots60_{w}']=60*f[f'shots_mean_{w}']/f[f'toi_mean_{w}'].where(f[f'toi_mean_{w}']>0)
    f['conversion_10']=f.goals_mean_10/f.shots_mean_10.where(f.shots_mean_10>0)
    f['prior_hit_rate']=g.hit.transform(lambda s:s.shift().expanding().mean())
    f['rest_days']=g.date_match.diff().dt.days
    f['home']=pd.to_numeric(f.is_home_player,errors='raise')
    f['drought']=np.nan;f['hot_streak']=np.nan
    for _,group in g:
        drought=hot=0;observed=False
        for idx,goals in zip(group.index,group.buts):
            if observed:f.loc[idx,['drought','hot_streak']]=[drought,hot]
            if pd.notna(goals):
                drought=0 if goals>=1 else drought+1
                hot=hot+1 if goals>=1 else 0
                observed=True
    f['team_goals_5']=np.nan;f['opponent_goals_against_5']=np.nan
    if {'buts_match_equipe','buts_match_adversaire','adversaire_match'} <= set(f):
        keys=['id_match','team_player_match']
        cols=['buts_match_equipe','buts_match_adversaire']
        if f.groupby(keys)[cols].nunique(dropna=False).gt(1).any().any():
            raise ValueError('Conflicting team scores')
        teams=f.drop_duplicates(keys).sort_values(['date_match','id_match']).copy()
        groups=teams.groupby(['team_player_match','season_source'])
        teams['team_goals_5']=groups.buts_match_equipe.transform(lambda s:s.shift().rolling(5,min_periods=1).mean())
        teams['opponent_goals_against_5']=groups.buts_match_adversaire.transform(lambda s:s.shift().rolling(5,min_periods=1).mean())
        f=f.drop(columns=['team_goals_5','opponent_goals_against_5']).merge(
            teams[keys+['team_goals_5']],on=keys,how='left',validate='many_to_one')
        opponent=teams[keys+['opponent_goals_against_5']].rename(columns={'team_player_match':'adversaire_match'})
        f=f.merge(opponent,on=['id_match','adversaire_match'],how='left',validate='many_to_one')
    return f.sort_values(['date_match']+KEYS).reset_index(drop=True)

def matrix(frame, features):
    if not features or set(features)-set(FEATURES):raise ValueError('Unapproved features')
    x=frame[features].apply(pd.to_numeric,errors='raise')
    if np.isinf(x.to_numpy()).any():raise ValueError('Infinite feature')
    return x

def raw_probability(model, kind, x):
    if kind=='poisson':return -np.expm1(-model.predict(x))
    return model.predict_proba(x)[:,1]

def fit(history, cutoff, kind='logistic', features=None, settings=None):
    settings=settings or config();features=list(FEATURES if features is None else features)
    past=history[pd.to_datetime(history.date_match)<pd.Timestamp(cutoff)].copy()
    validate(past)
    dates=sorted(pd.to_datetime(past.date_match).unique())
    if len(dates)<20:raise ValueError('Insufficient training dates')
    boundary=dates[int(len(dates)*(1-settings['calibration_fraction']))]
    train=past[pd.to_datetime(past.date_match)<boundary]
    cal=past[pd.to_datetime(past.date_match)>=boundary]
    x=matrix(train,features)
    kept=[c for c in features if x[c].notna().any() and x[c].nunique()>1]
    if not kept:raise ValueError('No variable training features')
    y=goal_labels(train)
    if y.nunique()!=2:raise ValueError('Two classes required')
    if kind=='logistic':estimator=LogisticRegression(C=settings['logistic_C'],max_iter=2000,random_state=settings['seed'])
    elif kind=='poisson':estimator=PoissonRegressor(alpha=settings['poisson_alpha'],max_iter=2000)
    elif kind=='hgb':estimator=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,min_samples_leaf=50,l2_regularization=1,early_stopping=False,random_state=settings['seed'])
    else:raise ValueError('Unknown model')
    # Median and missing indicators learned ONLY from fitting rows; raw unknowns retained.
    model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),estimator)
    model.fit(x[kept],train.buts if kind=='poisson' else y)
    raw=raw_probability(model,kind,matrix(cal,kept))
    calibrator,info=select_calibrator(raw,goal_labels(cal),cal.date_match)
    return dict(model=model,kind=kind,calibrator=calibrator,features=kept,
                metadata=dict(fit_end=str(train.date_match.max()),calibration_end=str(cal.date_match.max()),
                cutoff=str(cutoff),train_rows=len(train),calibration_rows=len(cal),calibration=info,
                dropped_features=[c for c in features if c not in kept],version=settings['version']))

def predict(bundle, frame):
    if (pd.to_datetime(frame.date_match)<pd.Timestamp(bundle['metadata']['cutoff'])).any():
        raise ValueError('Prediction predates model cutoff')
    raw=raw_probability(bundle['model'],bundle['kind'],matrix(frame,bundle['features']))
    p=bundle['calibrator'].predict(raw)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probability')
    return p
