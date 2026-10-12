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


def test_official_source_season_alias_required_by_point_baseline():
    raw=pd.DataFrame({'season_source':['20222023','20232024']})
    result=r.source_season_alias(raw)
    assert result.saison.tolist()==[20222023,20232024]
    assert 'saison' not in raw
    result.loc[0,'saison']=20232024
    with pytest.raises(ValueError,match='Conflicting'):r.source_season_alias(result)


def test_cache_compatibility_only_for_unchanged_historical_code(monkeypatch):
    code=str(r.ROOT/'model/henachel/features.py')
    saved={code:'8c58c4bed4058042a93c734b5fbcec4bcddbc288c166ab39aa7b70f8f9ebbc31','history':'same'}
    current={code:r.digest(code),'history':'same'}
    assert r.compatible_cache_inputs(saved,current)
    assert not r.compatible_cache_inputs(saved,{**current,'history':'changed'})
    monkeypatch.setattr(r,'historical_code_digest',lambda:'different')
    assert not r.compatible_cache_inputs(saved,current)


class FixedModel:
    def predict_proba(self,x):
        p=np.full(len(x),.4)
        return np.column_stack([1-p,p])


class FixedCalibrator:
    def predict(self,p):return p


def test_oof_resume_reuses_all_folds_and_retains_models(tmp_path,monkeypatch):
    rows=[]
    for fold,(start,end) in enumerate(r.protocol()['folds']):
        for player in range(4):
            rows.append(dict(date_match=start,id_match=fold+1,id_joueur=player+1,game_type=2,position='D' if player<2 else 'C',
                             points=player%2,is_home_player=player%2,season_point_hits_before_match=2,season_games_before_match=5))
    # Include observations before the first cutoff for prevalence; fake fit
    # isolates checkpoint orchestration from statistical training.
    rows.append({**rows[0],'date_match':'2023-10-01','id_match':99})
    frame=pd.DataFrame(rows);source=tmp_path/'features.csv.gz';frame.to_csv(source,index=False)
    r.write(source.with_suffix('.report.json'),dict(sha256=r.digest(source)))
    calls=[]
    def fake_fit(frame,cutoff,features,target):
        calls.append(cutoff)
        return FixedModel(),FixedCalibrator(),dict(kept=['is_home_player'],calibration={'method':'raw'})
    monkeypatch.setattr(r,'fit',fake_fit)
    out=tmp_path/'results';r.run(source,out,'POINT')
    assert len(calls)==24 and len(list(out.glob('*_frozen_fold.joblib')))==6
    r.run(source,out,'POINT')
    assert len(calls)==24
    restored=r.joblib.load(out/'V0_frozen_fold.joblib')
    assert restored['features']==['is_home_player']
    (out/'V0_frozen_fold.joblib').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='Frozen model cache changed'):r.run(source,out,'POINT')
