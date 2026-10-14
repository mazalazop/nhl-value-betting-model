"""Incrementally refresh an existing public NHL artifact for a research slate.

Uses production parsers, final boxscore validation, public cache and atomic writes.
Never writes betting history or credentials. Rebuild features with 01 afterwards.
"""
import argparse
import importlib.util
from pathlib import Path
import sys
import pandas as pd
import requests
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'model'))
from henachel.history import atomic_csv
from henachel.data import final_mask
from henachel.nhl_cache import BoxscoreCache
from henachel.prospective_http import get_json


def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'model'/filename)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result


def replace_games(old,fresh):
    if fresh.empty:return old.copy()
    if fresh.id_match.isna().any():raise ValueError('Missing game identity')
    # Explicit records preserve unknown cells without pandas all-NA concat inference.
    return pd.DataFrame(old.loc[~old.id_match.isin(fresh.id_match)].to_dict('records')+fresh.to_dict('records'))


def schedule_rows(payload,source):
    rows={}
    for week in payload['gameWeek']:
        for game in week['games']:
            if game['gameType'] in {2,3}:
                rows[game['id']]=source.build_match_row(dict(game,gameDate=week['date']),game['season'])
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target-date',required=True)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();raw=args.root/'data/raw';target=pd.Timestamp(args.target_date)
    source=module('source_refresh','00_refresh_sources.py')
    box=module('box_refresh','00b_build_base_match_fusionnee.py')
    pp=module('pp_refresh','00a_refresh_pp_stats.py')
    stats=pd.read_csv(raw/'stats.csv',low_memory=False)
    matches=pd.read_csv(raw/'matchs.csv',low_memory=False)
    start=min(target-pd.Timedelta(days=2),pd.to_datetime(stats.date_match).max()).normalize()
    session=requests.Session();cache=raw/'goal_slate_public_cache'
    def fetch(url,ttl=3600):return get_json(session,url,cache,ttl)
    games={}
    for day in pd.date_range(start,target,freq='7D').union(pd.DatetimeIndex([target])):
        payload=fetch(f'https://api-web.nhle.com/v1/schedule/{day.date()}',60)['document']
        games.update(schedule_rows(payload,source))
    matches=replace_games(matches,pd.DataFrame(games.values())).sort_values(['date_match','id_match'])
    slate=matches[matches.date_match.eq(str(target.date()))]
    if slate.empty:raise ValueError('No NHL slate returned; existing inputs preserved')
    teams=sorted(set(slate.id_equipe_domicile)|set(slate.id_equipe_exterieur));rosters=[]
    for team in teams:
        saved=fetch(f'https://api-web.nhle.com/v1/roster/{team}/current')
        for row in source.parse_roster_payload(saved['document'],team,int(slate.saison.max())):
            row['observed_at']=saved['observed_at'];rosters.append(row)
    roster=pd.DataFrame(rosters)
    if roster.empty:raise ValueError('Current rosters unavailable')
    finals=matches[final_mask(matches)&(pd.to_datetime(matches.date_match)<target)]
    refresh=finals[(pd.to_datetime(finals.date_match)>=start)|~finals.id_match.isin(stats.id_match)]
    boxcache=BoxscoreCache(raw/'boxscores');new=[]
    for game in refresh.itertuples():
        payload=boxcache.get(game.id_match,game.date_match,lambda gid:fetch(f'https://api-web.nhle.com/v1/gamecenter/{gid}/boxscore')['document'])
        rows=box.parse_game_to_stats_rows(payload,int(game.id_match),game.date_match,str(game.saison))
        if not rows:raise ValueError(f'No player stats for final {game.id_match}')
        new.extend(rows)
    stats=replace_games(stats,pd.DataFrame(new))
    if not set(finals.id_match)<=set(stats.id_match):raise ValueError('Incomplete finalized history')
    base=box.build_base_match_fusionnee(stats,matches)
    old_pp=pd.read_csv(raw/'pp_stats_game.csv',low_memory=False);pp_rows=[]
    for game_type in sorted(refresh.game_type.unique()):
        result=fetch(pp.build_url(start.date(),(target-pd.Timedelta(days=1)).date(),int(game_type)))['document']
        pp_rows.extend(result['data'])
    fresh_pp=pp.normalize_dataframe(pd.DataFrame(pp_rows)) if pp_rows else pd.DataFrame()
    merged_pp=replace_games(old_pp,fresh_pp)
    if merged_pp.duplicated(['id_match','id_joueur']).any():raise ValueError('Duplicate PP identity')
    players=pd.read_csv(raw/'joueurs.csv',low_memory=False)
    lookup=roster.drop_duplicates('id_joueur').reindex(columns=players.columns)
    players=pd.DataFrame(players.loc[~players.id_joueur.isin(lookup.id_joueur)].to_dict('records')+lookup.to_dict('records'))
    for frame,name in [(matches,'matchs.csv'),(stats,'stats.csv'),(base,'base_match_fusionnee.csv'),(merged_pp,'pp_stats_game.csv'),(roster,'roster_current.csv'),(players,'joueurs.csv')]:
        atomic_csv(frame,raw/name)
    print(dict(slate_games=len(slate),roster_players=len(roster),refreshed_games=len(refresh),history_rows=len(stats),latest_history=stats.date_match.max()),flush=True)


if __name__=='__main__':main()
