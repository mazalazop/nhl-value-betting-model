import pandas as pd
import pytest
from conftest import load_script

matcher=load_script('06_match_model_to_unibet_odds')

def rows():
    m=pd.DataFrame([dict(date_match=pd.Timestamp('2026-10-02'),id_match=1,id_joueur=42,nom='Jean-Gabriel Pagéau',position='C',team_player_match='UTA',adversaire_match='TOR',is_home_player=1,proba_point_1p_calibree=.6,proba_point_1p_raw=.65,rank_proba_sur_date=1,rank_proba_sur_match=1,start_time_utc='2026-10-03T00:00:00Z',hard_exclude_hot_streak_pre=1)])
    o=pd.DataFrame([dict(bookmaker='unibet',market='player_points',stat='points',threshold=1,outcome_key='1_plus',outcome_label='1+',event_url='https://example.test/1',event_id='bk1',event_slug='uta-tor',home_team='Utah Hockey Club',away_team='Toronto Maple Leafs',team='Utah Mammoth',player_name='Jean Gabriel Pageau',odds_decimal=2.,implied_probability=.5,event_start_utc='2026-10-03T00:00:00Z',captured_at='2026-10-02T23:00:00Z',date_match='2026-10-02')])
    return m,o

def run(m,o):
    return matcher.match_rows(m,o,'2026-10-02',now=pd.Timestamp('2026-10-02T23:30:00Z'))

def test_matching_exact_identity_and_math():
    m,o=rows(); out,_,_,_=run(m,o); assert len(out)==1
    row=out.iloc[0]
    assert row.edge_probability==pytest.approx(.1)
    assert row.ev_per_unit==pytest.approx(.2)
    assert row.kelly_fraction==pytest.approx(.2)
    assert row.fair_odds_model==pytest.approx(1/.6)
    assert row.hard_exclude_hot_streak_pre==1
    o['player_name']='J. G. Pageau'; again,_,_,_=run(m,o)
    assert again.bet_id.tolist()==out.bet_id.tolist()

@pytest.mark.parametrize('field,value', [('threshold',2),('stat','goals'),('market','player_shots'),('outcome_key','under'),('team','Toronto Maple Leafs'),('home_team','Boston Bruins'),('event_start_utc','2026-10-04T00:00:00Z'),('captured_at','2026-09-01T00:00:00Z'),('date_match','2026-10-01'),('odds_decimal',1),('odds_decimal',float('inf')),('implied_probability',.1),('captured_at',None)])
def test_reject_unsafe_odds(field,value):
    m,o=rows(); o[field]=value
    out,_,rejected,_=run(m,o); assert out.empty; assert rejected.reason.notna().all()

def test_ambiguous_initials_rejected_both_directions():
    m,o=rows();m.loc[0,'nom']='Jack Hughes'; m2=m.copy();m2['id_joueur']=43;m2['nom']='John Hughes'
    o['player_name']='J. Hughes';out,_,_,_=run(pd.concat([m,m2],ignore_index=True),o)
    assert out.empty
    o['player_name']='Jack Hughes';out,_,_,_=run(pd.concat([m,m2],ignore_index=True),o)
    assert out.id_joueur.tolist()==[42]

def test_duplicate_event_ambiguous_and_empty_safe():
    m,o=rows();out,_,_,_=run(m,pd.concat([o,o],ignore_index=True));assert out.empty
    out,_,_,_=run(m,o.iloc[:0]);assert 'bet_id' in out

@pytest.mark.parametrize('field',['event_id','bookmaker','event_start_utc','captured_at','date_match','team','player_name'])
def test_null_required_identity_fields_rejected(field):
    m,o=rows();o[field]=float('nan')
    out,_,rejected,_=run(m,o)
    assert out.empty;assert len(rejected)==1

def test_explicit_nhl_game_id_must_agree():
    m,o=rows();o['nhl_game_id']=999
    assert run(m,o)[0].empty
    o['nhl_game_id']=1
    assert len(run(m,o)[0])==1

@pytest.mark.parametrize('probability',[-.01,1.01,float('nan'),float('inf')])
def test_invalid_model_probabilities_rejected(probability):
    m,o=rows();m['proba_point_1p_calibree']=probability
    assert run(m,o)[0].empty
