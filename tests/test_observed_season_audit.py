import pandas as pd
import pytest
from test_runtime import module

m = module('audit_observed_season')


def test_losses_and_unit_roi_with_unsettled_excluded():
    d = pd.DataFrame(dict(status=['SETTLED','SETTLED','PENDING'],
        official_win=[1,0,None], model_probability=[.8,.6,.9], odds=[1.5,2,1.4],
        date=['2026-11-01']*3))
    r=m.metrics(d)
    assert r['proposed']==3 and r['settled']==2 and r['unresolved']==1
    assert r['brier']==pytest.approx(.2)
    assert r['roi_simulated']==pytest.approx(-.25)
    assert r['mean_ev']==pytest.approx(.2)


def test_unavailable_is_not_zero_performance():
    d=pd.DataFrame(columns=['status'])
    r=m.metrics(d)
    assert r['proposed']==0 and r['roi_simulated'] is None
