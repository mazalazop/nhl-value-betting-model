"""Adapt an immutable production artifact; never rerun or modify V1."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from henachel.but_v2_shadow import utc
from henachel.but_v2_validation import write


def prepare(root, output):
    root=Path(root);output=Path(output)
    hp=root/'data/final/base_features_context_v2.csv'
    vp=root/'outputs/predictions_upcoming_goal_research.csv'
    mp=root/'data/raw/matchs.csv'
    ref=pd.read_csv(vp);matches=pd.read_csv(mp)
    if ref.empty:raise ValueError('No V1 goal universe in upstream artifact')
    keys=['id_match','id_joueur']
    if ref.duplicated(keys).any() or matches.duplicated('id_match').any():raise ValueError('Duplicate upstream identity')
    if not ref.model_name.eq('GOAL17_OPERATIONAL_REFIT').all():raise ValueError('Unexpected V1 algorithm')
    if ref.date_match.nunique()!=1 or not ref.training_cutoff.eq(ref.date_match).all():raise ValueError('Unexpected V1 target date')
    columns=keys+['nom','date_match','team_player_match','adversaire_match','start_time_utc']
    slate=ref[columns].merge(matches[['id_match','saison','id_equipe_domicile','id_equipe_exterieur','status','schedule_state','game_type',
                                    'date_match','start_time_utc']],on='id_match',suffixes=('','_schedule'),validate='many_to_one')
    if slate.saison.isna().any() or not slate.date_match.eq(slate.date_match_schedule).all():raise ValueError('Schedule identity/date mismatch')
    if not (utc(slate.start_time_utc)==utc(slate.start_time_utc_schedule)).all():raise ValueError('Schedule kickoff mismatch')
    home=slate.team_player_match.eq(slate.id_equipe_domicile)
    away=slate.team_player_match.eq(slate.id_equipe_exterieur)
    expected=np.where(home,slate.id_equipe_exterieur,slate.id_equipe_domicile)
    if not (home|away).all() or not slate.adversaire_match.eq(expected).all():raise ValueError('Schedule team mismatch')
    slate['season_source']=slate.saison.astype(int);slate['is_home_player']=home.astype(int)
    for col in ['buts','tirs','temps_de_glace','temps_pp','buts_match_equipe','buts_match_adversaire']:slate[col]=np.nan
    odds_path=root/'outputs/structured_bookmaker/normalized_goals_odds.json'
    if odds_path.exists():
        payload=json.loads(odds_path.read_text());odds=pd.DataFrame(payload.get('rows',[]))
        if len(odds):
            if not (odds.market.eq('player_goals') & odds.stat.eq('goals') & odds.threshold.eq(1) & odds.outcome_key.eq('1_plus')).all():raise ValueError('Wrong odds contract')
            odds=odds.rename(columns={'nhl_game_id':'id_match','nhl_player_id':'id_joueur'})
            if odds.duplicated(keys).any():raise ValueError('Ambiguous odds for player/game')
            joined=ref[keys+['team_player_match','start_time_utc']].merge(odds,on=keys,how='inner',validate='one_to_one')
            if not joined.team_player_match.eq(joined.team).all() or not (utc(joined.start_time_utc)==utc(joined.event_start_utc)).all():raise ValueError('Odds identity mismatch')
            odds=odds[keys+['odds_decimal','captured_at','event_url']].rename(columns={'captured_at':'odds_captured_at_utc','event_url':'odds_source'})
            odds['odds_market']='GOAL_1_PLUS'
            ref=ref.merge(odds,on=keys,how='left',validate='one_to_one')
    sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [hp,vp,mp]+([odds_path] if odds_path.exists() else [])}
    output.mkdir(parents=True,exist_ok=False)
    slate.to_csv(output/'slate.csv',index=False);ref.to_csv(output/'v1.csv',index=False)
    # A byte-for-byte source copy, not recomputed history or historical picks.
    import shutil
    shutil.copyfile(hp,output/'history.csv')
    write(output/'provenance.json',dict(source_files_sha256=sources,n=len(ref),date=str(ref.date_match.iloc[0]),
          v1_generated_at_utc=sorted(ref.generated_at_utc.unique()),source='downloaded production artifact'))
    return slate,ref
