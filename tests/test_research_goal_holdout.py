import importlib.util,sys
from pathlib import Path
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import close_research_goal_study as h

def report():
    return dict(holdout_used=False,protocol=h.r.frozen_protocol(),comparisons={n:{'development_only_selected':False} for n in h.r.FAMILIES},reports={n:{'metrics':{'logloss':.4}} for n in h.r.FAMILIES})

def test_no_eligible_family_keeps_baseline():assert h.choose(report())=='baseline'
def test_only_eligible_family_selected():
    a=report();a['comparisons']['short_volume']['development_only_selected']=True
    a['reports']['interactions']['metrics']['logloss']=.01
    assert h.choose(a)=='short_volume'
def test_consumed_results_never_select():
    a=report();a['holdout_used']=True
    with pytest.raises(ValueError):h.choose(a)
def test_holdout_marker_created_exclusively(tmp_path):
    h.reserve_once(tmp_path)
    before=(tmp_path/'HOLDOUT_OPENED.json').read_bytes()
    with pytest.raises(FileExistsError):h.reserve_once(tmp_path)
    assert (tmp_path/'HOLDOUT_OPENED.json').read_bytes()==before
