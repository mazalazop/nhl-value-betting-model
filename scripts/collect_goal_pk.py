"""Read official game-level NHL PK data into an isolated research file."""
import argparse
import datetime
import json
from pathlib import Path
import sys
import pandas as pd
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model'))
from henachel.data import normalize_team


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    hist=pd.read_csv(args.history,usecols=['id_match','date_match','season_source','game_type','id_equipe_domicile','id_equipe_exterieur'])
    games=hist.drop_duplicates('id_match').set_index('id_match');rows=[];calls=[]
    url='https://api.nhle.com/stats/rest/en/team/penaltykill'
    for season in sorted(hist.season_source.unique()):
        for kind in [2,3]:
            start=0
            while True:
                params={'isAggregate':'false','isGame':'true','start':start,'limit':1000,
                        'cayenneExp':f'seasonId={int(season)} and gameTypeId={kind}'}
                response=requests.get(url,params=params,timeout=45);response.raise_for_status();payload=response.json()
                captured=datetime.datetime.now(datetime.timezone.utc).isoformat()
                chunk=payload['data'];calls.append({'url':response.url,'captured_at':captured,'rows':len(chunk),'total':payload['total']})
                for row in chunk:
                    game_id=int(row['gameId'])
                    if game_id not in games.index:continue
                    game=games.loc[game_id];opponent=normalize_team(row['opponentTeamAbbrev'])
                    teams={normalize_team(game.id_equipe_domicile),normalize_team(game.id_equipe_exterieur)}
                    if opponent not in teams or len(teams)!=2:raise ValueError('PK NHL event team mismatch')
                    if pd.Timestamp(row['gameDate']).date()!=pd.Timestamp(game.date_match).date():raise ValueError('PK NHL date mismatch')
                    if int(game.season_source)!=int(season) or int(game.game_type)!=kind:raise ValueError('PK NHL season/type mismatch')
                    team=(teams-{opponent}).pop()
                    rows.append({'id_match':game_id,'date_match':game.date_match,'season_source':int(season),'team':team,
                                 'pp_goals_against':row['ppGoalsAgainst'],'times_shorthanded':row['timesShorthanded'],'captured_at':captured})
                start+=len(chunk)
                if start>=payload['total']:break
                if not chunk:raise ValueError('Incomplete PK pagination')
    out=pd.DataFrame(rows)
    if out.empty or out.duplicated(['id_match','team']).any():raise ValueError('Missing or duplicate PK records')
    coverage=len(out)/(2*len(games))
    if coverage<.95:raise ValueError(f'Insufficient PK coverage: {coverage:.3%}')
    args.output.parent.mkdir(parents=True,exist_ok=True);out.to_csv(args.output,index=False)
    args.output.with_suffix('.manifest.json').write_text(json.dumps({'coverage':coverage,'records':len(out),'games':len(games),'calls':calls},indent=2))
    print(json.dumps({'coverage':coverage,'records':len(out),'http_calls':len(calls)}))


if __name__=='__main__':main()
