"""Compact official team context, validated against collected NHL event identities."""
import argparse,json,math,sys,hashlib
from pathlib import Path
import pandas as pd
from collect_research_history import Cache
REQUIRED={
 'powerplay':['ppOpportunities','powerPlayGoalsFor','ppTimeOnIcePerGame'],
 'penaltykill':['timesShorthanded','ppGoalsAgainst'],
 'penalties':['penalties','totalPenaltiesDrawn','penaltiesDrawnPer60','penaltiesTakenPer60']}
COUNTS={'ppOpportunities','powerPlayGoalsFor','timesShorthanded','ppGoalsAgainst','penalties','totalPenaltiesDrawn'}

def validate_records(rows,kind,games):
    seen=set();out=[]
    for row in rows:
        gid=row['gameId'];game=games.get(gid)
        if game is None:raise ValueError('Unexpected context game')
        if row.get('gamesPlayed')!=1:raise ValueError('Report must be per-game')
        if pd.Timestamp(row['gameDate']).date()!=pd.Timestamp(game['date_match']).date():raise ValueError('Context game date mismatch')
        opponent=row['opponentTeamAbbrev'];teams={game['id_equipe_domicile'],game['id_equipe_exterieur']}
        if opponent not in teams or len(teams)!=2:raise ValueError('Opponent identity mismatch')
        team=(teams-{opponent}).pop();key=(gid,team)
        if key in seen:raise ValueError('Duplicate context game/team')
        seen.add(key)
        values={}
        for col in REQUIRED[kind]:
            value=row.get(col)
            if value is not None:
                if not isinstance(value,(int,float)) or isinstance(value,bool) or not math.isfinite(value) or value<0:raise ValueError('Invalid context statistic')
                if col in COUNTS and int(value)!=value:raise ValueError('Noninteger count')
            values[col]=value
        out.append(dict(id_match=gid,date_match=game['date_match'],season_source=game['season_source'],team=team,opponent=opponent,nhl_team_id=row.get('teamId'),**values))
    if len(out)!=2*len(games):raise ValueError('Incomplete team/game coverage')
    return pd.DataFrame(out)

def main():
    p=argparse.ArgumentParser();p.add_argument('--history',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if (a.output/'team_context.csv.gz').exists():raise FileExistsError('Immutable team context already exists')
    cache=Cache(a.output/'cache');raw=pd.read_csv(a.history,low_memory=False);game_frame=raw.drop_duplicates('id_match');parts=[];reports=[]
    for season,group in game_frame.groupby('season_source'):
        games=group.set_index('id_match').to_dict('index');joined=None
        for kind in REQUIRED:
            payload=cache.get('https://api.nhle.com/stats/rest/en/team/'+kind,{'isAggregate':'false','isGame':'true','start':0,'limit':-1,'cayenneExp':f'seasonId={int(season)} and gameTypeId=2'})
            if payload.get('total')!=len(payload['data']) or payload['total']>=10000:raise ValueError('Incomplete/capped report')
            data=validate_records(payload['data'],kind,games)
            if joined is None:joined=data
            else:joined=joined.merge(data,on=['id_match','date_match','season_source','team','opponent','nhl_team_id'],validate='one_to_one')
            reports.append(dict(season=int(season),report=kind,rows=len(data),games=len(games),missing=data.isna().sum().to_dict()))
        if len(joined)!=2*len(games):raise ValueError('Report team IDs disagree')
        parts.append(joined)
    result=pd.concat(parts,ignore_index=True);dest=a.output/'team_context.csv.gz';result.to_csv(dest,index=False)
    report=dict(scope='regular team-game historical outcomes; must be shifted before any feature use',http_calls=cache.calls,cache_hits=cache.hits,
        reports=reports,rows=len(result),source_history_sha256=hashlib.sha256(a.history.read_bytes()).hexdigest(),output_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),pp1_available=False,experiments_run=False)
    (a.output/'coverage.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
