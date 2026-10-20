import numpy as np
import pandas as pd
import pytest
from henachel.but_v2_shadow import capture,report
from test_but_v2 import sample

def inputs():
    history=sample();slate=sample(51).tail(4).copy()
    slate[['buts','tirs','temps_de_glace','temps_pp']]=np.nan
    slate['start_time_utc']='2024-02-20T23:00Z';slate['status']='FUT';slate['schedule_state']='OK'
    ref=slate[['id_match','id_joueur','start_time_utc']].copy()
    ref['probability_goal']=.25;ref['generated_at_utc']='2024-02-20T12:00Z'
    return history,slate,ref

def test_shadow_capture_report_and_immutability(tmp_path):
    h,s,r=inputs();path=tmp_path/'capture'
    out=capture(h,s,r,path,{'fixture':True},now='2024-02-20T13:00Z')
    assert out.scope.eq('BUT_V2_SHADOW').all() and out.status.eq('UNRESOLVED').all()
    with pytest.raises(FileExistsError):capture(h,s,r,path,{},now='2024-02-20T13:00Z')
    result=s[['id_match','id_joueur']].iloc[:2].copy()
    result['buts']=[0,1];result['status']='OFF'
    result['source']='https://api-web.nhle.com/v1/gamecenter/50/boxscore'
    result['observed_at_utc']='2024-02-21T03:00Z'
    summary=report(path,result,tmp_path/'report')
    assert summary['settled']==2 and summary['unresolved']==2 and summary['roi'] is None
    assert summary['v1']['n']==summary['v2']['n']==2
    with (path/'predictions.csv').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='changed'):report(path,result,tmp_path/'report2')

@pytest.mark.parametrize('bad',['late','outcome','wrong_v1','naive'])
def test_shadow_rejects_unverifiable_inputs(tmp_path,bad):
    h,s,r=inputs();clock='2024-02-20T13:00Z'
    if bad=='late':clock='2024-02-21T13:00Z'
    if bad=='outcome':s['buts']=0
    if bad=='wrong_v1':r['generated_at_utc']='2024-02-21T13:00Z'
    if bad=='naive':s['start_time_utc']='2024-02-20T23:00'
    with pytest.raises(ValueError):capture(h,s,r,tmp_path/'capture',{},now=clock)

def test_archived_odds_simulation_and_reject_post_capture_quote(tmp_path):
    h,s,r=inputs()
    r['odds_decimal']=3.;r['odds_captured_at_utc']='2024-02-20T12:30Z'
    r['odds_source']='fixture-bookmaker';r['odds_market']='GOAL_1_PLUS'
    path=tmp_path/'capture'
    capture(h,s,r,path,{'fixture':True},now='2024-02-20T13:00Z')
    result=s[['id_match','id_joueur']].copy()
    result['buts']=[0,1,0,1];result['status']='OFF'
    result['source']='https://api-web.nhle.com/v1/gamecenter/50/boxscore'
    result['observed_at_utc']='2024-02-21T03:00Z'
    summary=report(path,result,tmp_path/'report')
    assert summary['roi']['mean']==.5 and summary['roi']['simulation_only']
    r['odds_captured_at_utc']='2024-02-20T14:00Z'
    with pytest.raises(ValueError,match='odds'):capture(h,s,r,tmp_path/'bad',{},now='2024-02-20T13:00Z')
