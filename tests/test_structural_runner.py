import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
import research_structural as r
from henachel.goal import fit_goal_model
from conftest import load_script


def fixture(target):
    p=r.protocol();rng=np.random.default_rng(8);n=240
    frame=pd.DataFrame(rng.normal(size=(n,len(p['baselines'][target]))),columns=p['baselines'][target])
    frame['date_match']=np.repeat(pd.date_range('2023-01-01',periods=60),4)
    frame['id_match']=np.arange(n)//4;frame['id_joueur']=np.arange(n)%4
    frame['buts']=rng.binomial(1,.3,n);frame['points']=frame.buts+rng.binomial(1,.3,n)
    frame['a_marque_un_but']=frame.buts.ge(1).astype(int);frame['target_point_1p']=frame.points.ge(1).astype(int)
    return frame


def test_protocol_fixed_and_bounded():
    p=r.protocol();assert p['fits']==52 and len(p['variants'])==5
    assert len(p['baselines']['POINT'])==81 and len(p['baselines']['GOAL'])==17
    assert p['production_changed'] is False


@pytest.mark.parametrize('target',['POINT','GOAL'])
def test_baseline_exact_production_fit_parity(target):
    frame=fixture(target);cutoff=pd.Timestamp('2023-03-01');cols=r.protocol()['baselines'][target]
    model,cal,meta=r.fit(frame,cutoff,cols,target)
    if target=='GOAL':old,old_cal,old_meta=fit_goal_model(frame,cutoff,cols)
    else:old,old_cal,old_meta=load_script('05_predict_upcoming_games').fit_point_model_and_calibrator(frame,'target_point_1p','date_match',cutoff)
    np.testing.assert_array_equal(cal.predict(model.predict_proba(frame[meta['kept']])[:,1]),old_cal.predict(old.predict_proba(frame[old_meta['feature_cols_kept']])[:,1]))


@pytest.mark.parametrize('target',['POINT','GOAL'])
def test_future_labels_cannot_change_fit_or_calibration(target):
    frame=fixture(target);cutoff=pd.Timestamp('2023-02-15');cols=r.protocol()['baselines'][target]
    model,cal,meta=r.fit(frame,cutoff,cols,target)
    frame.loc[frame.date_match.ge(cutoff),['points','buts','a_marque_un_but','target_point_1p']]=np.nan
    other,other_cal,other_meta=r.fit(frame,cutoff,cols,target)
    assert meta==other_meta
    np.testing.assert_array_equal(cal.predict(model.predict_proba(frame[meta['kept']])[:,1]),other_cal.predict(other.predict_proba(frame[meta['kept']])[:,1]))


def test_invalid_target_rejected_without_filtering():
    frame=fixture('POINT');frame.loc[0,'points']=-1
    with pytest.raises(ValueError,match='Invalid POINT'):r.fit(frame,'2023-03-01',r.protocol()['baselines']['POINT'],'POINT')
