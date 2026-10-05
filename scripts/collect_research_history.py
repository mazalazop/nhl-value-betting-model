"""Resumable official NHL research collection; never writes production data."""
import argparse, gzip, hashlib, json, shutil, sys, time
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
import pandas as pd
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model'))
from henachel.data import validate_player_games
TEAMS='ANA ARI BOS BUF CAR CBJ CGY CHI COL DAL DET EDM FLA LAK MIN MTL NJD NSH NYI NYR OTT PHI PIT SEA SJS STL TBL TOR VAN VGK WPG WSH'.split()
SEASONS=[20222023,20232024]
WEB='https://api-web.nhle.com/v1'
STATS='https://api.nhle.com/stats/rest/en/skater/'

class Cache:
    def __init__(self,root):
        self.root=root;root.mkdir(parents=True,exist_ok=True);self.calls=0;self.hits=0
    def get(self,url,params=None):
        full=requests.Request('GET',url,params=params).prepare().url
        path=self.root/(hashlib.sha256(full.encode()).hexdigest()+'.json.gz')
        if path.exists():
            self.hits+=1
            with gzip.open(path,'rt') as f:return json.load(f)['payload']
        if shutil.disk_usage(self.root).free<250*1024**2:raise RuntimeError('Disk reserve below 250 MiB')
        time.sleep(1.5)
        for attempt in range(3):
            self.calls+=1
            r=requests.get(full,timeout=(10,40))
            if r.status_code!=429:break
            if attempt==2:r.raise_for_status()
            delay=min(60,max(20,int(r.headers.get('Retry-After','30'))))
            print('NHL rate limit; waiting',delay,flush=True);time.sleep(delay)
        r.raise_for_status();payload=r.json()
        tmp=path.with_suffix('.tmp')
        with gzip.open(tmp,'wt') as f:json.dump({'url':full,'captured_at':datetime.now(timezone.utc).isoformat(),'payload':payload},f)
        tmp.replace(path);return payload

def report(cache,name,season,start,end):
    params={'isAggregate':'false','isGame':'true','start':0,'limit':-1,
            'cayenneExp':f'seasonId={season} and gameTypeId=2 and gameDate>="{start}" and gameDate<="{end} 23:59:59"'}
    p=cache.get(STATS+name,params)
    if not isinstance(p.get('data'),list) or len(p['data'])!=p.get('total') or p['total']>=10000:
        raise ValueError('Incomplete/capped weekly report')
    return p['data']

def validate_stats(rows,name,season):
    required={'gameId','playerId','gameDate','opponentTeamAbbrev'}
    required|={'goals','assists','points','shots','timeOnIcePerGame','teamAbbrev'} if name=='summary' else {'ppTimeOnIce'}
    keys=set()
    for r in rows:
        if required-set(r):raise ValueError(f'{name} missing fields {required-set(r)}')
        key=(r['gameId'],r['playerId'])
        if key in keys:raise ValueError('Duplicate skater/game')
        keys.add(key)
        if int(r['gameId'])//1000000!=season//10000 or (int(r['gameId'])//10000)%100!=2:raise ValueError('Wrong season/game type')
        if int(r['playerId'])<=0:raise ValueError('Invalid player identity')
        pd.Timestamp(r['gameDate'])
        if name=='summary':
            for c in ['goals','assists','points','shots']:
                x=r[c]
                if x is None or x<0 or int(x)!=x:raise ValueError('Invalid observed outcome')
            if r['goals']+r['assists']!=r['points']:raise ValueError('Inconsistent points')
        for c in ['timeOnIcePerGame'] if name=='summary' else ['ppTimeOnIce']:
            if r[c] is not None and (not pd.notna(r[c]) or r[c]<0):raise ValueError('Invalid ice time')
    return keys

def probe(cache):
    results=[]
    for season in SEASONS:
        year=season//10000;gid=year*1000000+20001
        b=cache.get(f'{WEB}/gamecenter/{gid}/boxscore')
        if b['id']!=gid or b['gameState'] not in ['OFF','FINAL'] or b['gameType']!=2:raise ValueError('Invalid boxscore')
        day=b['gameDate'];sources={}
        for name in ['summary','powerplay']:
            rows=report(cache,name,season,day,day);validate_stats(rows,name,season)
            game_rows=[r for r in rows if r['gameId']==gid]
            if not game_rows:raise ValueError('Stats missing sample game')
            sources[name]=dict(rows=len(game_rows),fields=sorted(game_rows[0]))
            if name=='summary':
                actual={p['playerId']:p for side in ['homeTeam','awayTeam'] for position in ['forwards','defense'] for p in b['playerByGameStats'][side][position]}
                if set(actual)!={r['playerId'] for r in game_rows}:raise ValueError('Boxscore/summary player universe mismatch')
                for r in game_rows:
                    p=actual[r['playerId']]
                    if any(p[x]!=r[y] for x,y in [('goals','goals'),('assists','assists'),('points','points'),('sog','shots')]):raise ValueError('Boxscore stats mismatch')
        schedule=cache.get(f'{WEB}/club-schedule-season/TOR/{season}')
        if not any(g['gameType']==2 for g in schedule['games']):raise ValueError('Missing schedule')
        results.append(dict(season=season,game_id=gid,date=day,boxscore_stats_equal=True,sources=sources))
    return results

def collect(cache,out):
    allrows=[];coverage=[]
    for season in SEASONS:
        games={}
        for team in TEAMS:
            p=cache.get(f'{WEB}/club-schedule-season/{team}/{season}')
            for g in p['games']:
                if g['gameType']!=2:continue
                if g['season']!=season or g['gameState'] not in ['OFF','FINAL']:raise ValueError('Non-final or wrong-season schedule')
                if g['id'] in games and games[g['id']]!=g:raise ValueError('Inconsistent schedule copies')
                games[g['id']]=g
        if len(games)!=1312:raise ValueError(f'Incomplete regular season schedule: {season}, {len(games)}')
        dates=sorted(g['gameDate'] for g in games.values());start=date.fromisoformat(dates[0]);last=date.fromisoformat(dates[-1]);stats=[];pp=[]
        while start<=last:
            end=min(start+timedelta(days=6),last)
            stats+=report(cache,'summary',season,start.isoformat(),end.isoformat())
            pp+=report(cache,'powerplay',season,start.isoformat(),end.isoformat())
            print(season,start,len(stats),'rows cached',flush=True);start=end+timedelta(days=1)
        skeys=validate_stats(stats,'summary',season);pkeys=validate_stats(pp,'powerplay',season)
        if {r['gameId'] for r in stats}!=set(games):raise ValueError('Incomplete stats game coverage')
        if pkeys-skeys:raise ValueError('PP has unexpected player/game')
        power={(r['gameId'],r['playerId']):r for r in pp}
        for r in stats:
            g=games[r['gameId']];home=g['homeTeam']['abbrev'];away=g['awayTeam']['abbrev'];team=r['teamAbbrev'];opp=r['opponentTeamAbbrev']
            if {team,opp}!={home,away} or team==opp or pd.Timestamp(r['gameDate']).date()!=date.fromisoformat(g['gameDate']):raise ValueError('Stats/event mismatch')
            pp_row=power.get((r['gameId'],r['playerId']))
            if pp_row and pp_row['opponentTeamAbbrev']!=opp:raise ValueError('PP opponent mismatch')
            def mins(v):return None if v is None else v/60
            allrows.append(dict(id_match=r['gameId'],id_joueur=r['playerId'],date_match=g['gameDate'],season_source=season,game_type=2,status=g['gameState'],
                team_player_match=team,adversaire_match=opp,is_home_player=int(team==home),id_equipe_domicile=home,id_equipe_exterieur=away,
                buts_domicile=g['homeTeam']['score'],buts_exterieur=g['awayTeam']['score'],buts=r['goals'],passes=r['assists'],points=r['points'],tirs=r['shots'],
                temps_de_glace=mins(r['timeOnIcePerGame']),temps_pp=mins(pp_row['ppTimeOnIce']) if pp_row else None,
                plus_moins=r.get('plusMinus'),penalty_minutes=r.get('penaltyMinutes'),nom=r.get('skaterFullName'),position=r.get('positionCode'),
                match_trouve=1,check_team_ok=1,check_opp_ok=1))
        coverage.append(dict(season=season,games=len(games),observations=len(stats),players=len({r['playerId'] for r in stats}),pp_coverage=len(pkeys)/len(skeys)))
    frame=pd.DataFrame(allrows);validate_player_games(frame)
    if frame.temps_pp.notna().mean()<.9:raise ValueError('PP coverage insufficient')
    destination=out/'historical_source.csv.gz'
    if destination.exists():raise FileExistsError(destination)
    frame.to_csv(destination,index=False)
    return dict(seasons=coverage,observations=len(frame),players=frame.id_joueur.nunique(),missing=frame.isna().sum().to_dict(),source_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                boxscore_coverage='two sampled games, stats exact; remaining stats via official game reports',standings='not used by GOAL17 or registered additions',rosters='season rosters not pregame availability; observed skaters from official game reports')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--collect',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    cache=Cache(a.output/'cache');result={'status':'incomplete'}
    try:
        result['contracts']=probe(cache);result['status']='contracts_ok'
        if a.collect:result.update(collect(cache,a.output));result['status']='collected'
    except Exception as e:
        result.update(error_type=type(e).__name__,error=str(e));raise
    finally:
        result.update(http_calls=cache.calls,cache_hits=cache.hits,checked_at=datetime.now(timezone.utc).isoformat())
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        (a.output/f'report_{stamp}.json').write_text(json.dumps(result,indent=2,default=str));print(json.dumps(result,indent=2,default=str),flush=True)
if __name__=='__main__':main()
