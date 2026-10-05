"""Fail-closed POINT 1+ matching. All timestamps in the adapter contract are UTC."""
import hashlib
import math
import re
import unicodedata
from collections import Counter

import pandas as pd

IMPLIED_TOLERANCE = 1e-6
MAX_ODDS_AGE = pd.Timedelta(hours=6)
COLUMNS = '''bet_id run_date date_match id_match id_joueur player_name player_name_bookmaker position team opponent is_home bookmaker market stat threshold outcome_key outcome_label event_id event_url event_slug home_team away_team start_time_utc captured_at odds_decimal implied_probability model_probability_raw model_probability fair_odds_model edge_probability edge_probability_pct_points ev_per_unit kelly_fraction is_positive_ev recommended_flag recommendation_rank bet_status result actual_stat_value settled_at rank_proba_sur_date rank_proba_sur_match hard_exclude_hot_streak_pre match_method fuzzy_score'''.split()


def name(value):
    if value is None or pd.isna(value): return ''
    text=unicodedata.normalize('NFKD', str(value)).encode('ascii','ignore').decode().lower()
    return ' '.join(re.sub(r'[^a-z0-9 ]', ' ', text.replace("'",'')).split())


def names_compatible(a,b):
    a,b=name(a).split(),name(b).split()
    if len(a)<2 or len(a)!=len(b): return False
    # Surnames (including compound names) must agree; only explicit initials abbreviate.
    return a[-1]==b[-1] and all(x==y or (min(len(x),len(y))==1 and x[0]==y[0]) for x,y in zip(a[:-1],b[:-1]))


def utc(value):
    result=pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is None: raise ValueError('missing_or_naive_timestamp')
    return result.tz_convert('UTC')


def value_metrics(probability,odds,supplied_implied=None):
    p,c=float(probability),float(odds)
    if not math.isfinite(p) or not 0<=p<=1: raise ValueError('invalid_model_probability')
    if not math.isfinite(c) or c<=1: raise ValueError('invalid_odds')
    implied=1/c
    if supplied_implied is not None:
        given=float(supplied_implied)
        if not math.isfinite(given) or abs(given-implied)>IMPLIED_TOLERANCE: raise ValueError('inconsistent_implied_probability')
    ev=p*c-1
    return dict(odds_decimal=c,implied_probability=implied,model_probability=p,
                fair_odds_model=1/p if p else None,edge_probability=p-implied,
                edge_probability_pct_points=100*(p-implied),ev_per_unit=ev,
                kelly_fraction=max(0,ev/(c-1)),is_positive_ev=ev>0)


def stable_bet_id(game,player,bookmaker):
    return hashlib.sha256(f'{int(game)}|{int(player)}|{str(bookmaker).lower()}|points|1|at_least'.encode()).hexdigest()[:32]


def match_point_rows(model,odds,run_date,team_names,now=None):
    now=utc(now if now is not None else pd.Timestamp.now(tz='UTC'))
    aliases={name(alias):code for code,variants in team_names.items() for alias in [code]+variants}
    aliases.update({'utah':'UTA','utah hockey club':'UTA','utah mammoth':'UTA'})
    def team(x):
        code=aliases.get(name(x))
        if code is None: raise ValueError('unknown_team')
        return code
    model=model.reset_index(drop=True);odds=odds.reset_index(drop=True)
    candidates={}; reasons={}; valid={}
    if model.duplicated(['id_match','id_joueur']).any(): raise ValueError('Duplicate model player/game')
    for oi,o in odds.iterrows():
        try:
            if o.get('market')!='player_points' or o.get('stat')!='points' or float(o.get('threshold',0))!=1:
                raise ValueError('unsupported_market')
            if str(o.get('outcome_key','')).lower() not in {'1_plus','points_1_plus','1+','over_0.5'}:
                raise ValueError('unsupported_outcome')
            if any(pd.isna(o.get(k)) or not str(o.get(k,'')).strip() for k in ['bookmaker','event_id']): raise ValueError('missing_event_identity')
            start,captured=utc(o.get('event_start_utc')),utc(o.get('captured_at'))
            if now>=start: raise ValueError('event_already_started')
            if not pd.Timedelta(0)<=now-captured<=MAX_ODDS_AGE: raise ValueError('stale_or_future_odds')
            value_metrics(.5,o.get('odds_decimal'),o.get('implied_probability'))
            home,away,player_team=team(o.home_team),team(o.away_team),team(o.team)
            if home==away or player_team not in {home,away}: raise ValueError('inconsistent_teams')
            possible=[]
            for mi,m in model.iterrows():
                if pd.Timestamp(m.date_match).date()!=pd.Timestamp(o.get('date_match')).date(): continue
                if pd.Timestamp(m.date_match).date()!=pd.Timestamp(run_date).date(): continue
                if utc(m.get('start_time_utc'))!=start: continue
                if pd.notna(o.get('nhl_game_id')) and int(o.nhl_game_id)!=int(m.id_match): continue
                mt,op=team(m.team_player_match),team(m.adversaire_match)
                if player_team!=mt or {mt,op}!={home,away}: continue
                if m.is_home_player not in (0,1) or (mt if m.is_home_player==1 else op)!=home: continue
                if names_compatible(m.nom,o.player_name): possible.append(mi)
            exact=[mi for mi in possible if name(model.loc[mi,'nom'])==name(o.player_name)]
            if exact: possible=exact
            if len(possible)!=1: raise ValueError('ambiguous_player_or_event' if possible else 'no_verified_identity')
            mi=possible[0]; m=model.loc[mi]
            if any(not math.isfinite(float(m[k])) or float(m[k])<=0 or float(m[k])!=int(m[k]) for k in ['id_match','id_joueur']): raise ValueError('invalid_nhl_identity')
            metrics=value_metrics(m.proba_point_1p_calibree,o.odds_decimal,o.get('implied_probability'))
            value_metrics(m.proba_point_1p_raw,o.odds_decimal)
            candidates[oi]=mi;valid[oi]=metrics
        except (ValueError,TypeError,KeyError,AttributeError,OverflowError) as exc:
            reasons[oi]=str(exc)
    multiplicity=Counter(candidates.values()); matched=[];used=set()
    for oi,mi in candidates.items():
        if multiplicity[mi]!=1: reasons[oi]='multiple_bookmaker_candidates';continue
        o,m=odds.loc[oi],model.loc[mi];used.add(mi)
        row={k:o.get(k) for k in ['bookmaker','market','stat','threshold','outcome_key','outcome_label','event_id','event_url','event_slug','home_team','away_team','captured_at']}
        row.update(valid[oi]);row.update(bet_id=stable_bet_id(m.id_match,m.id_joueur,o.bookmaker),run_date=run_date,
            date_match=pd.Timestamp(m.date_match).date().isoformat(),id_match=int(m.id_match),id_joueur=int(m.id_joueur),
            player_name=m.nom,player_name_bookmaker=o.player_name,position=m.get('position'),team=team(m.team_player_match),opponent=team(m.adversaire_match),is_home=int(m.is_home_player),
            start_time_utc=m.start_time_utc,model_probability_raw=float(m.proba_point_1p_raw),
            recommended_flag=False,recommendation_rank=None,bet_status='pending',result='',actual_stat_value=None,settled_at='',
            rank_proba_sur_date=m.get('rank_proba_sur_date'),rank_proba_sur_match=m.get('rank_proba_sur_match'),
            hard_exclude_hot_streak_pre=m.get('hard_exclude_hot_streak_pre'),match_method='verified_name',fuzzy_score=None)
        matched.append(row)
    unmatched_model=model.loc[~model.index.isin(used)].copy();unmatched_model['reason']='no_unique_verified_odds'
    rejected=odds.loc[list(reasons)].copy();rejected['reason']=[reasons[i] for i in rejected.index]
    summary=dict(model_rows_count=len(model),bookmaker_rows_count=len(odds),matched_rows_count=len(matched),unmatched_model_rows_count=len(unmatched_model),unmatched_bookmaker_rows_count=len(rejected),exact_match_count=len(matched),fuzzy_match_count=0,duplicate_candidate_count=sum(v=='multiple_bookmaker_candidates' for v in reasons.values()))
    return pd.DataFrame(matched,columns=COLUMNS),unmatched_model,rejected,summary
