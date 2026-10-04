"""Small read-only availability probes, not a historical-data collection or fit."""
import argparse
import datetime
import json
from pathlib import Path
import requests


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    results=[]
    for season,date in [(20222023,'2023-02-01'),(20232024,'2024-02-01')]:
        year=str(season)[:4]
        urls={
            'boxscore':f'https://api-web.nhle.com/v1/gamecenter/{year}020001/boxscore',
            'powerplay':f'https://api.nhle.com/stats/rest/en/skater/powerplay?isAggregate=false&isGame=true&start=0&limit=1&cayenneExp=seasonId={season}%20and%20gameTypeId=2',
            'standings':f'https://api-web.nhle.com/v1/standings/{date}',
        }
        for source,url in urls.items():
            record={'season':season,'source':source,'url':url,'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
            try:
                response=requests.get(url,timeout=30);record['http_status']=response.status_code;response.raise_for_status();payload=response.json()
                if source=='boxscore':
                    players=payload.get('playerByGameStats',{}).get('awayTeam',{}).get('forwards',[])
                    record.update(game_id=payload.get('id'),game_state=payload.get('gameState'),game_date=payload.get('gameDate'),fields=list(players[0]) if players else [])
                elif source=='powerplay':
                    records=payload.get('data',[]);record.update(total=payload.get('total'),fields=list(records[0]) if records else [])
                else:
                    records=payload.get('standings',[]);record.update(teams=len(records),api_dates=sorted({r.get('date') for r in records if r.get('date')}),fields=list(records[0]) if records else [])
            except requests.RequestException as error:record['error']=str(error)
            results.append(record)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'scope':'availability samples only; full compatibility/coverage not established','results':results},indent=2))
    print(json.dumps([{k:r.get(k) for k in ['season','source','http_status','game_state','total','teams','error']} for r in results],indent=2))


if __name__=='__main__':main()
