"""Independent NHL historical PBP collector. Raw downloads are never committed.

Fixed research scope: 2024-25 and 2025-26, not selected by player outcomes.
Availability today does not establish historical point-in-time availability.
"""
import argparse,concurrent.futures,gzip,json,time,hashlib
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
import requests

def fetch(gid,root):
 path=root/(str(gid)+'.json.gz')
 if path.exists():return dict(id_match=gid,status='CACHED',sha256=hashlib.sha256(path.read_bytes()).hexdigest())
 url=f'https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play';errors=[]
 for attempt in range(3):
  try:
   r=requests.get(url,timeout=25);r.raise_for_status();d=r.json()
   if d.get('id')!=gid or d.get('gameState') not in {'OFF','FINAL'} or not d.get('rosterSpots') or not any(p.get('typeDescKey')=='game-end' for p in d.get('plays',[])):raise ValueError('Incomplete/nonfinal PBP')
   envelope=dict(source=url,observed_at_utc=datetime.now(timezone.utc).isoformat(),document=d)
   payload=gzip.compress(json.dumps(envelope,sort_keys=True).encode(),mtime=0)
   with path.open('xb') as f:f.write(payload)
   return dict(id_match=gid,status='DOWNLOADED',sha256=hashlib.sha256(payload).hexdigest(),errors=errors)
  except (requests.RequestException,ValueError) as exc:
   errors.append(str(exc))
   if attempt<2:
    delay=2**(attempt+1)
    if isinstance(exc,requests.HTTPError) and exc.response is not None:
     try:delay=max(delay,min(60,float(exc.response.headers.get('Retry-After',0))))
     except ValueError:pass
    time.sleep(delay)
 return dict(id_match=gid,status='UNAVAILABLE',errors=errors)

def main():
 p=argparse.ArgumentParser();p.add_argument('--history',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);a=p.parse_args()
 h=pd.read_csv(a.history,usecols=['id_match','season_source']);ids=sorted(h.loc[h.season_source.isin([20242025,20252026]),'id_match'].unique())
 if a.limit:ids=ids[:a.limit]
 a.output.mkdir(parents=True,exist_ok=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
  for i,result in enumerate(ex.map(lambda g:fetch(int(g),a.output),ids),1):
   with (a.output/'attempts.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
   if i%50==0 or i==len(ids):print(i,len(ids),flush=True)
if __name__=='__main__':main()
