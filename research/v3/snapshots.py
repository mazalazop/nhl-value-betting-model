"""Disabled V3 snapshot contract and separate CLASSEUR category normalization."""
import hashlib,json
from pathlib import Path
import pandas as pd

CATEGORIES={'TOP10_POINT':'POINT','TOP10_BUT':'BUT','TWO_POINT_PER_GAME':'POINT','TWO_BUT_PER_GAME':'BUT'}

def freeze(root,record):
 if record.get('study') not in {'POINT_V3_RESEARCH','BUT_V3_RESEARCH'}:raise ValueError('Research namespace required')
 if record.get('enabled',False):raise ValueError('V3 activation not authorized')
 cap=pd.Timestamp(record['captured_at_utc']);kick=pd.Timestamp(record['start_time_utc'])
 if cap.tzinfo is None or kick.tzinfo is None or cap>=kick:raise ValueError('Pregame UTC times required')
 for field in ['id_match','id_joueur','model_version','sources']:
  if not record.get(field):raise ValueError('Missing provenance/identity')
 for source in record['sources']:
  available=pd.Timestamp(source['available_at_utc'])
  if available.tzinfo is None or available>cap:raise ValueError('Future source')
 if record.get('pp_status') not in {None,'UNKNOWN','PROJECTED_LAST_GAME','PROBABLE','CONFIRMED'}:raise ValueError('Invalid PP confidence')
 payload=json.dumps(record,sort_keys=True,allow_nan=False).encode();key=hashlib.sha256(payload).hexdigest()
 root=Path(root);root.mkdir(parents=True,exist_ok=True);path=root/(key+'.json')
 if path.exists():
  if path.read_bytes()!=payload:raise ValueError('Immutable collision')
 else:
  with path.open('xb') as f:f.write(payload)
 return path

def classeur(rows):
 """One pick identity plus a separate membership table. Never CANONICAL."""
 picks={};members=[]
 for r in rows:
  if r['category'] not in CATEGORIES or r['market']!=CATEGORIES[r['category']]:raise ValueError('Invalid category/market')
  if not r.get('id_joueur') or not r.get('id_match'):raise ValueError('Stable NHL IDs required; never guess names')
  key=hashlib.sha256(json.dumps([r['date'],r['market'],r['id_match'],r['id_joueur'],r.get('bookmaker')]).encode()).hexdigest()
  pick={k:v for k,v in r.items() if k!='category'};pick.update(pick_id=key,archive_group='CLASSEUR',provenance='EXCEL_RECONSTRUCTION_NOT_CANONICAL',probability_roi_admissible=False)
  if key in picks and picks[key]!=pick:raise ValueError('Conflicting duplicate workbook pick')
  picks[key]=pick;members.append(dict(pick_id=key,category=r['category']))
 unique={json.dumps(m,sort_keys=True):m for m in members}
 return list(picks.values()),list(unique.values())
