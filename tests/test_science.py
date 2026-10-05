import numpy as np
import pandas as pd
import pytest
from conftest import load_script
from henachel.point import binary_labels

future=load_script('05_predict_upcoming_games');train=load_script('02_train_point_model');cal=load_script('03_calibrate_point_model')

@pytest.mark.parametrize('bad',[None,.5,2,-1,float('inf')])
def test_invalid_targets_fail_instead_of_drop_or_truncate(bad):
    frame=pd.DataFrame({'target_point_1p':[0,bad],'date_match':['2026-01-01','2026-01-02']})
    for script in [train,future]:
        with pytest.raises(ValueError):script.find_target_column(frame)
    with pytest.raises(ValueError):binary_labels([0,bad])

def test_invalid_calibration_data_not_silently_removed():
    for field,value in [('target_point_1p',.5),('proba_point_1p_raw',np.nan),('date_match',None)]:
        frame=pd.DataFrame({'target_point_1p':[0.,1.],'proba_point_1p_raw':[.2,.7],'date_match':['2026-01-01','2026-01-02']});frame.loc[1,field]=value
        with pytest.raises(ValueError):cal.standardize_prediction_frame(frame)

def training_fixture():
    rng=np.random.default_rng(42);n=480
    df=pd.DataFrame({col:rng.normal(size=n) for col in future.FEATURE_WHITELIST})
    df['date_match']=np.repeat(pd.date_range('2024-01-01',periods=60),8)
    df['id_joueur']=np.tile(np.arange(1,9),60);df['id_match']=np.repeat(np.arange(60)+1,8)
    df['target_point_1p']=(rng.random(n)<.55).astype(int)
    return df

def test_test_labels_and_features_cannot_change_fit_or_calibrator():
    df=training_fixture();cut=pd.Timestamp('2024-02-20')
    a,ca,ma=future.fit_point_model_and_calibrator(df,'target_point_1p','date_match',cut)
    changed=df.copy();mask=changed.date_match.ge(cut)
    changed.loc[mask,'target_point_1p']=1-changed.loc[mask,'target_point_1p']
    changed.loc[mask,future.FEATURE_WHITELIST]=9999
    b,cb,mb=future.fit_point_model_and_calibrator(changed,'target_point_1p','date_match',cut)
    assert ma==mb
    x=df.loc[:,ma['feature_cols_kept']]
    np.testing.assert_array_equal(a.predict_proba(x),b.predict_proba(x))
    np.testing.assert_array_equal(ca.predict(a.predict_proba(x)[:,1]),cb.predict(b.predict_proba(x)[:,1]))

def test_metrics_and_reliability_bins_exact():
    from henachel.evaluation import metrics,reliability_bins
    result=metrics([0,1],[.2,.8]);assert result['brier']==pytest.approx(.04)
    assert result['logloss']==pytest.approx(-np.log(.8));assert result['auc']==1
    assert result['precision_top_10pct']==1;assert result['lift_top_10pct']==2
    bins=reliability_bins([0,1],[0,1]);assert sum(x['count'] for x in bins)==2
    assert bins[0]['observed_rate']==0;assert bins[-1]['observed_rate']==1

def test_walk_forward_reserves_final_test_and_expands():
    from henachel.evaluation import expanding_evaluation
    data=training_fixture()
    report,preds=expanding_evaluation(data,future.fit_point_model_and_calibrator,future.FEATURE_WHITELIST,window_dates=15)
    assert not report['final_test_used'];assert preds.date_match.max()<pd.Timestamp(report['final_test_start'])
    ends=[f['train_end'] for f in report['folds']];assert ends==sorted(set(ends))
    for fold in report['folds']:
        assert fold['train_end']<fold['eval_start']
        split=fold['fit']['fit_calibration_split'];assert split['fit_end']<split['calib_start']

def test_baseline_comparison_missing_is_not_unchanged():
    comparison=load_script('10_compare_point_model_to_baseline')
    assert comparison.judge('brier',float('nan'),.2)=='unavailable'
    assert comparison.summarize_split(pd.DataFrame({'split':['x'],'judgment':['unavailable']}),'x')['status']=='insufficient_data'
