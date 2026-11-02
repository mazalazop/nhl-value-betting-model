"""Official PBP geometric proxies, never labelled xG or confirmed high-danger chances."""
import argparse,gzip,json,math
from pathlib import Path
import numpy as np,pandas as pd
from core import save

def summarize_game(envelope,expected):
 d=envelope['document'];gid=int(d['id'])
 if envelope.get('source')!=f'https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play':raise ValueError('Untrusted source')
 if d.get('gameState') not in {'OFF','FINAL'} or not any(p.get('typeDescKey')=='game-end' for p in d.get('plays',[])):raise ValueError('Incomplete official game')
 if set(expected.id_match)!={gid}:raise ValueError('Wrong game')
 roster={p['playerId'] for p in d.get('rosterSpots',[])};shots={};previous=None;malformed=False
 for event in d['plays']:
  typ=event.get('typeDescKey');period=event.get('periodDescriptor',{})
  if period.get('periodType')=='SO':continue
  detail=event.get('details',{})
  if typ not in ['goal','shot-on-goal','missed-shot','blocked-shot']:
   previous=event;continue
  pid=detail.get('scoringPlayerId') if typ=='goal' else detail.get('shootingPlayerId')
  if pid is None:malformed=True;previous=event;continue
  info=shots.setdefault(pid,dict(attempts=0,sog=0,goals=0,unblocked=0,distances=[],angles=[],rebounds=0,close=0));info['attempts']+=1
  info['sog']+=typ in ['goal','shot-on-goal'];info['goals']+=typ=='goal'
  if typ!='blocked-shot':
   info['unblocked']+=1
   x,y=detail.get('xCoord'),detail.get('yCoord');side=event.get('homeTeamDefendingSide');owner=detail.get('eventOwnerTeamId')
   if x is not None and y is not None and side in ['left','right'] and owner in [d['homeTeam']['id'],d['awayTeam']['id']]:
    attack_right=(side=='left')==(owner==d['homeTeam']['id']);nx=x if attack_right else -x
    dist=math.hypot(89-nx,y);angle=math.degrees(math.atan2(abs(y),89-nx))
    info['distances'].append(dist);info['angles'].append(angle);info['close']+=dist<=20
   if previous and previous.get('typeDescKey')=='shot-on-goal' and previous.get('periodDescriptor',{}).get('number')==period.get('number') and previous.get('details',{}).get('eventOwnerTeamId')==owner:
    def seconds(e):
     m,s=map(int,e['timeInPeriod'].split(':'));return m*60+s
    if 0<=seconds(event)-seconds(previous)<=3:info['rebounds']+=1
  previous=event
 out=[]
 for row in expected.to_dict('records'):
  pid=int(row['id_joueur']);v=shots.get(pid,dict(attempts=0,sog=0,goals=0,unblocked=0,distances=[],angles=[],rebounds=0,close=0))
  known=not malformed and pid in roster and v['sog']==row['tirs']
  geometry=known and (not v['unblocked'] or len(v['distances'])/v['unblocked']>=.9)
  out.append(dict(id_match=gid,id_joueur=pid,official_pbp_goals=v['goals'] if not malformed and pid in roster else None,historical_goals_agree=(v['goals']==row['buts']) if 'buts' in row else None,quality_known=known,attempts=v['attempts'] if known else None,
   unblocked_close=v['close'] if geometry else None,distance=float(np.mean(v['distances'])) if geometry and v['distances'] else None,
   angle=float(np.mean(v['angles'])) if geometry and v['angles'] else None,rebounds=v['rebounds'] if known else None,
   source_observed_at_utc=envelope['observed_at_utc'],reason='OFFICIAL_FINAL_SOG_RECONCILED' if known else 'MISSING_PLAYER_OR_EVENT_OR_HISTORICAL_SOG_DISAGREEMENT'))
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--history',type=Path,required=True);p.add_argument('--pbp',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 h=pd.read_csv(a.history);h=h[h.season_source.isin([20242025,20252026])];rows=[];missing=[]
 for gid,g in h.groupby('id_match'):
  path=a.pbp/(str(gid)+'.json.gz')
  if not path.exists():missing.append(int(gid));continue
  rows.extend(summarize_game(json.loads(gzip.decompress(path.read_bytes())),g))
 result=pd.DataFrame(rows);result.to_csv(a.output,index=False)
 save(a.output.with_suffix('.coverage.json'),dict(games_expected=h.id_match.nunique(),games_downloaded=result.id_match.nunique() if len(result) else 0,missing_games=missing,rows_expected=len(h),rows_reconciled=int(result.quality_known.sum()) if len(result) else 0,geometry_is_xg=False))
if __name__=='__main__':main()
