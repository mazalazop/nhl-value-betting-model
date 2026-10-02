import pandas as pd
import pytest
from conftest import load_script
from test_matching import rows, run

picks=load_script('07_build_daily_bets')

def candidate():
    m,o=rows();return run(m,o)[0]

def test_rerun_preserves_settled_history(tmp_path):
    path=tmp_path/'history.csv';data=candidate()
    picks.append_history(path,data)
    existing=pd.read_csv(path);existing['bet_status']='settled';existing['result']='win';existing['actual_stat_value']=2;existing['settled_at']='2026-10-03T03:00:00Z';existing.to_csv(path,index=False)
    picks.append_history(path,data)
    saved=pd.read_csv(path);assert len(saved)==1;assert saved.result.tolist()==['win'];assert saved.actual_stat_value.tolist()==[2]

def test_hot_streak_both_policies_and_negative_ev(tmp_path):
    data=candidate();data['value_gap']=data.edge_probability
    for disabled,expected in [(True,1),(False,0)]:
        result,_=picks.build_daily_bets(data,'2026-10-02',10,1.4,.9,.02,True,disabled)
        assert len(result)==expected
    data['model_probability']=.4;data['edge_probability']=-.1;data['value_gap']=-.1;data['ev_per_unit']=-.2
    result,_=picks.build_daily_bets(data,'2026-10-02',10,1.4,.9,.02,True,True)
    assert len(result)==1  # Existing business rule: EV does not gate selection.

def test_canonical_append_rejects_duplicate_and_conflicting_identity(tmp_path):
    path=tmp_path/'history.csv';data=candidate();picks.append_history(path,data)
    changed=data.copy();changed['id_joueur']=999
    with pytest.raises(ValueError): picks.append_history(path,changed)
    with pytest.raises(ValueError): picks.append_history(path,pd.concat([data,data]))

def test_empty_history_has_canonical_schema(tmp_path):
    path=tmp_path/'history.csv';picks.append_history(path,candidate().iloc[:0]);saved=pd.read_csv(path)
    assert {'id_match','id_joueur','bet_id','settled_at'}.issubset(saved)
