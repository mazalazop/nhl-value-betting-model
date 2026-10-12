"""Public official team PP/PK/penalty contract samples; no fit or production writes."""
import argparse,json
from pathlib import Path
from collect_research_history import Cache

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if (a.output/'contracts.json').exists():raise FileExistsError('Contract report exists')
    cache=Cache(a.output/'cache');records=[]
    for gid in [2022020001,2023020001]:
        for report in ['powerplay','penaltykill','penalties']:
            url='https://api.nhle.com/stats/rest/en/team/'+report
            try:
                data=cache.get(url,{'isAggregate':'false','isGame':'true','start':0,'limit':-1,'cayenneExp':f'gameId={gid}'})
                rows=data['data']
                valid=len(rows)==2 and all(r.get('gameId')==gid for r in rows)
                records.append({'game_id':gid,'report':report,'contract_ok':valid,'total':data.get('total'),'fields':sorted(rows[0]) if rows else [],'sample':rows})
            except Exception as e:records.append({'game_id':gid,'report':report,'contract_ok':False,'error_type':type(e).__name__,'error':str(e)})
    (a.output/'contracts.json').write_text(json.dumps({'scope':'two game samples only, not season coverage','records':records,'http_calls':cache.calls,'cache_hits':cache.hits},indent=2))
    print(json.dumps([{k:v for k,v in r.items() if k!='sample'} for r in records],indent=2))
if __name__=='__main__':main()
