"""Explicit NHL data contracts. Unknown is never equivalent to zero or final."""
import numpy as np
import pandas as pd

FINAL_STATES = frozenset({'OFF','FINAL'})
TEAM_ALIASES = {'ARI':'UTA','PHX':'UTA','PHO':'UTA'}


def normalize_team(value):
    if pd.isna(value): return None
    code=str(value).strip().upper()
    return TEAM_ALIASES.get(code,code) or None


def game_state(status, schedule_state=None):
    state='' if pd.isna(status) else str(status).upper()
    schedule='' if pd.isna(schedule_state) else str(schedule_state).upper()
    if state in {'PPD','POSTPONED'} or schedule in {'PPD','POSTPONED'}: return 'postponed'
    if state in {'CANC','CANCELLED','CANCELED'} or schedule in {'CANC','CANCELLED','CANCELED'}: return 'cancelled'
    if state in FINAL_STATES: return 'final'
    if state in {'LIVE','CRIT'}: return 'live'
    if state in {'FUT','PRE'}: return 'upcoming'
    return 'unknown'


def final_mask(frame):
    if 'status' not in frame: return pd.Series(False,index=frame.index)
    schedule=frame.get('schedule_state',pd.Series('',index=frame.index)).fillna('')
    return pd.Series([game_state(a,b)=='final' for a,b in zip(frame.status.fillna(''),schedule)],index=frame.index)


def validate_player_games(frame, require_outcomes=True):
    required=['id_match','id_joueur','date_match','team_player_match','adversaire_match',
              'is_home_player','id_equipe_domicile','id_equipe_exterieur']
    missing=set(required)-set(frame)
    if missing: raise ValueError(f'Missing player/game fields: {sorted(missing)}')
    if frame[required].isna().any().any(): raise ValueError('Missing critical player/game fields')
    if pd.to_datetime(frame.date_match,errors='coerce').isna().any(): raise ValueError('Invalid game date')
    if frame.duplicated(['id_match','id_joueur']).any(): raise ValueError('Duplicate player/game keys')
    for col in ['id_match','id_joueur']:
        ids=pd.to_numeric(frame[col],errors='coerce')
        if ids.isna().any() or (ids<=0).any() or (ids%1!=0).any(): raise ValueError(f'Invalid {col}')
    if not frame.is_home_player.isin([0,1]).all(): raise ValueError('Invalid home flag')
    home=frame.id_equipe_domicile.map(normalize_team)
    away=frame.id_equipe_exterieur.map(normalize_team)
    expected=home.where(frame.is_home_player==1,away)
    opponent=away.where(frame.is_home_player==1,home)
    if (home==away).any() or not frame.team_player_match.map(normalize_team).eq(expected).all() or not frame.adversaire_match.map(normalize_team).eq(opponent).all():
        raise ValueError('Inconsistent team/opponent')
    if require_outcomes:
        for col in ['points','buts','passes']:
            if col not in frame: raise ValueError(f'Missing target source: {col}')
            x=pd.to_numeric(frame[col],errors='coerce')
            if x.isna().any() or not np.isfinite(x).all() or (x<0).any() or (x%1!=0).any(): raise ValueError(f'Invalid outcome: {col}')
        if not (frame.points == frame.buts+frame.passes).all(): raise ValueError('points != goals + assists')
    return frame
