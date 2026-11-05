import copy,json,sys
from pathlib import Path
import numpy as np,pandas as pd,pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'research/v3'))
import core as v3
from shot_features import summarize_game
from snapshots import classeur,freeze

def history(n=20):
 rows=[]
 for day in range(n):
  for pid,team,opp in [(1,'A','B'),(2,'B','A')]:
   rows.append(dict(date_match=pd.Timestamp('2024-01-01')+pd.Timedelta(days=day),id_match=100+day,id_joueur=pid,
    season_source=20232024,team_player_match=team,adversaire_match=opp,buts=day%2,points=day%3,passes=day%2,tirs=day%4,
    temps_de_glace=15.,temps_pp=2.,is_home_player=int(team=='A'),buts_match_equipe=2,buts_match_adversaire=2,status='OFF',game_type=2))
 return pd.DataFrame(rows)

def test_features_prefix_and_target_date_never_used():
 d=history();f=v3.features(d);prefix=v3.features(d[d.date_match<'2024-01-12'])
 columns=sum(v3.FAMILIES.values(),[])
 pd.testing.assert_frame_equal(prefix[columns].reset_index(drop=True),f[f.date_match<'2024-01-12'][columns].reset_index(drop=True))
 altered=d.copy();altered.loc[altered.date_match>='2024-01-12',['buts','points','tirs','temps_pp']]=9
 g=v3.features(altered)
 pd.testing.assert_frame_equal(f[f.date_match<='2024-01-12'][columns],g[g.date_match<='2024-01-12'][columns])
 assert f.loc[f.date_match.eq('2024-01-01'),'shots_10'].isna().all()

def test_unknowns_and_duplicate_guard():
 d=history();d['temps_pp']=np.nan;f=v3.features(d)
 assert f.pp_10.isna().all()
 with pytest.raises(ValueError,match='duplicate'):v3.features(pd.concat([d,d.iloc[:1]]))
 d.loc[0,'temps_de_glace']=0
 with pytest.raises(ValueError,match='participant'):v3.features(d)

def test_paired_alignment_and_date_clusters():
 d=history();d['y']=d.buts;d['probability']=.4
 assert v3.paired(d,d,4,100)['brier']['delta']==0
 b=d.copy();b.loc[0,'id_match']=999
 with pytest.raises(AssertionError):v3.paired(d,b,4,100)
 assert v3.metrics(d)['top10']['n']==40

def test_train_only_imputation():
 d=v3.features(history());cols=sum(v3.FAMILIES.values(),[])
 model,kept=v3.fit_logistic(d,cols,d.buts)
 before=model.named_steps['simpleimputer'].statistics_.copy()
 future=d[kept].astype(float).copy();future[:]=np.nan
 assert np.isfinite(model.predict_proba(future)).all()
 np.testing.assert_array_equal(before,model.named_steps['simpleimputer'].statistics_)

def envelope():
 return dict(source='https://api-web.nhle.com/v1/gamecenter/100/play-by-play',observed_at_utc='2026-11-01T00:00:00Z',document=dict(id=100,gameState='OFF',homeTeam={'id':1},awayTeam={'id':2},rosterSpots=[{'playerId':1}],plays=[
  dict(typeDescKey='shot-on-goal',homeTeamDefendingSide='left',periodDescriptor={'number':1,'periodType':'REG'},timeInPeriod='01:00',details={'shootingPlayerId':1,'eventOwnerTeamId':1,'xCoord':80,'yCoord':0}),
  dict(typeDescKey='goal',homeTeamDefendingSide='left',periodDescriptor={'number':1,'periodType':'REG'},timeInPeriod='01:02',details={'scoringPlayerId':1,'eventOwnerTeamId':1,'xCoord':85,'yCoord':0}),
  dict(typeDescKey='game-end')]))

def test_shot_proxies_reconciled_and_not_xg():
 e=envelope();expected=pd.DataFrame([dict(id_match=100,id_joueur=1,tirs=2)])
 r=summarize_game(e,expected)[0]
 assert r['quality_known'] and r['attempts']==2 and r['rebounds']==1 and r['unblocked_close']==2 and r['distance']==6.5
 e['document']['plays'][-1]={}
 with pytest.raises(ValueError,match='Incomplete'):summarize_game(e,expected)
 e=envelope();expected['tirs']=3;r=summarize_game(e,expected)[0]
 assert not r['quality_known'] and r['attempts'] is None

def test_shot_unknown_coordinates_not_zero():
 e=envelope();e['document']['plays'][0]['details'].pop('xCoord')
 r=summarize_game(e,pd.DataFrame([dict(id_match=100,id_joueur=1,tirs=2)]))[0]
 assert r['distance'] is None and r['unblocked_close'] is None

def test_classeur_categories_dedup_and_never_canonical():
 r=dict(date='2026-11-01',market='POINT',id_match=100,id_joueur=1,category='TOP10_POINT',odds=1.5)
 picks,members=classeur([r,dict(r,category='TWO_POINT_PER_GAME'),r])
 assert len(picks)==1 and len(members)==2 and picks[0]['archive_group']=='CLASSEUR'
 assert not picks[0]['probability_roi_admissible']
 with pytest.raises(ValueError,match='Conflicting'):classeur([r,dict(r,odds=2)])

def test_snapshot_identity_clock_and_immutability(tmp_path):
 r=dict(study='POINT_V3_RESEARCH',enabled=False,captured_at_utc='2027-01-01T12:00:00Z',start_time_utc='2027-01-01T20:00:00Z',id_match=1,id_joueur=1,model_version='fixture',sources=[dict(available_at_utc='2027-01-01T11:00:00Z')],pp_status='PROJECTED_LAST_GAME')
 p=freeze(tmp_path,r);assert freeze(tmp_path,r)==p
 assert freeze(tmp_path,dict(r,captured_at_utc='2027-01-01T13:00:00Z'))!=p
 with pytest.raises(ValueError,match='activation'):freeze(tmp_path,dict(r,enabled=True))
 with pytest.raises(ValueError,match='Future'):freeze(tmp_path,dict(r,sources=[dict(available_at_utc='2027-01-02T00:00:00Z')]))

def test_protected_production_files_unchanged():
 import hashlib
 root=Path(__file__).resolve().parents[1]
 manifest=json.loads((root/'research/v3/protected_files.json').read_text())
 for name,digest in manifest.items():
  assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
 assert not json.loads((root/'research/v3/protocol.json').read_text())['enabled']


def test_conversion_gap_is_shifted_and_not_xg():
 d=history(60);d['tirs']=4
 f=v3.features(d);before=f.loc[f.date_match.eq('2024-02-20'),'conversion_gap'].to_numpy()
 d.loc[d.date_match.ge('2024-02-20'),'buts']=7
 g=v3.features(d);np.testing.assert_allclose(before,g.loc[g.date_match.eq('2024-02-20'),'conversion_gap'],equal_nan=True)
 assert f.loc[f.date_match.lt('2024-01-20'),'conversion_gap'].isna().all()


def test_quality_features_use_only_prior_matches():
 d=history();q=d[['id_match','id_joueur']].copy()
 for c in ['attempts','unblocked_close','distance','angle','rebounds']:q[c]=1.
 f=v3.features(d,q);q.loc[q.id_match.ge(110),['attempts','distance']]=99
 g=v3.features(d,q)
 pd.testing.assert_frame_equal(f.loc[f.date_match.le('2024-01-11'),v3.QUALITY],g.loc[g.date_match.le('2024-01-11'),v3.QUALITY])


def test_quality_season_strings_and_reference_order():
 from run_quality import quality_history
 d=history();d['season_source']=d.season_source.astype(str)
 h=quality_history(d.sample(frac=1,random_state=42),[20232024]);f=v3.features(h)
 assert list(zip(h.id_match,h.id_joueur))==list(zip(f.id_match,f.id_joueur))
 with pytest.raises(ValueError,match='No rows'):quality_history(d,[19901991])


def test_official_collector_retries_then_caches_without_network(tmp_path,monkeypatch):
 import collect_shots as c
 calls=[]
 class Response:
  def raise_for_status(self):pass
  def json(self):return envelope()['document']
 def get(*args,**kwargs):
  calls.append(args)
  if len(calls)<3:raise c.requests.Timeout('fixture timeout')
  return Response()
 monkeypatch.setattr(c.requests,'get',get);monkeypatch.setattr(c.time,'sleep',lambda _:None)
 assert c.fetch(100,tmp_path)['status']=='DOWNLOADED'
 assert len(calls)==3
 assert c.fetch(100,tmp_path)['status']=='CACHED' and len(calls)==3


def test_official_collector_empty_response_never_archived(tmp_path,monkeypatch):
 import collect_shots as c
 class Response:
  def raise_for_status(self):pass
  def json(self):return {}
 monkeypatch.setattr(c.requests,'get',lambda *a,**k:Response());monkeypatch.setattr(c.time,'sleep',lambda _:None)
 result=c.fetch(100,tmp_path)
 assert result['status']=='UNAVAILABLE' and len(result['errors'])==3
 assert not list(tmp_path.iterdir())
