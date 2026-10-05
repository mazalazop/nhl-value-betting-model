"""Current roster evidence takes priority; historical fallback never guesses a transfer."""
import warnings
import pandas as pd
from henachel.data import normalize_team


def choose_players(history,roster,teams,target_date,now=None,lookback_days=45,include_goalies=False):
    now=pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if now.tzinfo is None: raise ValueError('Roster clock must be timezone-aware')
    hist=history[pd.to_datetime(history.date_match)<pd.Timestamp(target_date)].copy()
    latest=hist.sort_values(['date_match','id_match']).groupby('id_joueur').tail(1).copy()
    latest['team_player_match']=latest.team_player_match.map(normalize_team)
    output=[];covered=set()
    if roster is not None and len(roster):
        required={'id_joueur','nom','position','id_equipe','observed_at'}
        if not required.issubset(roster): raise ValueError('Incomplete roster contract')
        r=roster.copy();r['team_player_match']=r.id_equipe.map(normalize_team)
        observed=pd.to_datetime(r.observed_at,utc=True,errors='coerce');age=now-observed
        r=r[age.between(pd.Timedelta(0),pd.Timedelta(days=2)) & r.team_player_match.isin(teams)]
        # A current snapshot cannot be used to replay an old date: that would leak transfers.
        if pd.Timestamp(target_date).date()<now.tz_convert('America/New_York').date():
            raise ValueError('Current roster cannot reconstruct historical membership')
        if r.id_joueur.duplicated().any(): raise ValueError('Player assigned to multiple current roster entries')
        covered=set(r.team_player_match);r['roster_source']='nhl_current'
        last_dates=latest.set_index('id_joueur').date_match
        r['days_since_last_game']=(pd.Timestamp(target_date)-pd.to_datetime(r.id_joueur.map(last_dates))).dt.days
        output.append(r)
    missing=set(teams)-covered
    if missing:
        warnings.warn(f'Roster unavailable/stale for {sorted(missing)}; historical team fallback, max {lookback_days} days',UserWarning)
        fallback=latest[latest.team_player_match.isin(missing)].copy()
        fallback['days_since_last_game']=(pd.Timestamp(target_date)-pd.to_datetime(fallback.date_match)).dt.days
        fallback=fallback[fallback.days_since_last_game.between(0,lookback_days)]
        # A proven transfer must not also reappear on its historical team's fallback roster.
        proven=set(output[0].id_joueur) if output else set()
        fallback=fallback[~fallback.id_joueur.isin(proven)]
        fallback['roster_source']='historical_fallback';output.append(fallback)
    pool=pd.concat(output,ignore_index=True) if output else latest.iloc[:0]
    if not include_goalies: pool=pool[pool.position.ne('G')]
    return pool.reset_index(drop=True)
