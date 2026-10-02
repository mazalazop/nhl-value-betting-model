import pandas as pd
import pytest
from test_history import candidate
from henachel.settlement import settle

@pytest.mark.parametrize('points,result',[(0,'loss'),(1,'win'),(3,'win')])
def test_final_results_idempotent_and_corrections(points,result):
    history=candidate();stats=pd.DataFrame([dict(id_match=1,id_joueur=42,points=points,buts=0,passes=points)])
    matches=pd.DataFrame([dict(id_match=1,status='OFF')])
    out,changes,unresolved=settle(history,stats,matches,'2026-10-03T03:00:00Z')
    assert out.result.tolist()==[result];assert len(changes)==1;assert unresolved.empty
    again,changes,_=settle(out,stats,matches,'2026-10-04T03:00:00Z')
    pd.testing.assert_frame_equal(out,again);assert changes.empty
    stats['points']=2;stats['passes']=2
    corrected,changes,_=settle(out,stats,matches,'2026-10-05T03:00:00Z')
    assert corrected.actual_stat_value.tolist()==[2]
    assert len(changes)==(points!=2)

@pytest.mark.parametrize('state',['LIVE','CRIT','FUT','PPD','CANC','UNKNOWN'])
def test_non_final_never_settled(state):
    out,changes,unresolved=settle(candidate(),pd.DataFrame([dict(id_match=1,id_joueur=42,points=2,buts=1,passes=1)]),pd.DataFrame([dict(id_match=1,status=state)]),'2026-10-03T03:00:00Z')
    assert changes.empty;assert out.bet_status.tolist()==['pending'];assert len(unresolved)==1

@pytest.mark.parametrize('bad',['missing','scratch','stat','outcome','threshold','inconsistent'])
def test_unknown_is_not_over(bad):
    history=candidate();stats=pd.DataFrame([dict(id_match=1,id_joueur=42,points=2,buts=1,passes=1)])
    if bad=='missing': stats['points']=None
    if bad=='scratch': stats=stats.iloc[:0]
    if bad=='stat': history['stat']='goals'
    if bad=='outcome': history['outcome_key']='banana'
    if bad=='threshold': history['threshold']=2
    if bad=='inconsistent': stats['buts']=5
    out,changes,unresolved=settle(history,stats,pd.DataFrame([dict(id_match=1,status='OFF')]),'2026-10-03T03:00:00Z')
    assert changes.empty;assert len(unresolved)==1;assert out.bet_status.tolist()==['pending']

def test_missing_corrected_source_does_not_unsettle():
    history=candidate();history['result']='win';history['bet_status']='settled';history['actual_stat_value']=1;history['settled_at']='yesterday'
    out,changes,_=settle(history,pd.DataFrame(columns=['id_match','id_joueur','points']),pd.DataFrame([dict(id_match=1,status='LIVE')]),'2026-10-03T03:00:00Z')
    pd.testing.assert_frame_equal(out,history);assert changes.empty
