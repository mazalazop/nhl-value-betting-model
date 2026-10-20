import numpy as np
import pandas as pd
import pytest
from henachel.but_v2 import build_features, fit, predict, config, FEATURES

def sample(days=50):
    return pd.DataFrame([dict(id_match=d,id_joueur=p,date_match=pd.Timestamp('2024-01-01')+pd.Timedelta(days=d),
        season_source=20232024,team_player_match='A',status='OFF',game_type=2,
        buts=int((d+p)%4==0),tirs=(d+p)%6,temps_de_glace=15+d%5,temps_pp=2+d%3,is_home_player=d%2)
        for d in range(days) for p in range(4)])

def test_prefix_invariance_current_and_future_outcomes():
    x=sample();full=build_features(x)
    date=pd.Timestamp('2024-02-01');before=x[x.date_match<=date].copy()
    before.loc[before.date_match==date,['buts','tirs','temps_pp','temps_de_glace']]=999
    changed=build_features(before)
    pd.testing.assert_frame_equal(full.loc[full.date_match==date,FEATURES].reset_index(drop=True),
                                  changed.loc[changed.date_match==date,FEATURES].reset_index(drop=True))

def test_missing_is_not_zero_and_season_reset():
    x=sample();x['temps_pp']=np.nan
    out=build_features(x)
    assert out.pp_mean_5.isna().all()
    assert out.loc[out.date_match==out.date_match.min(),'drought'].isna().all()
    x.loc[x.date_match>=pd.Timestamp('2024-02-01'),'season_source']=20242025
    out=build_features(x)
    assert out.loc[out.date_match==pd.Timestamp('2024-02-01'),'shots_mean_10'].isna().all()

@pytest.mark.parametrize('kind',['logistic','poisson','hgb'])
def test_fit_future_labels_do_not_affect_model_or_calibrator(kind):
    f=build_features(sample());cut='2024-02-10'
    a=fit(f,cut,kind);changed=f.copy()
    changed.loc[changed.date_match>=cut,'buts']=999
    b=fit(changed,cut,kind)
    ev=f[f.date_match>=cut]
    np.testing.assert_allclose(predict(a,ev),predict(b,ev))
    assert str(a['metadata']['calibration_end'])<cut
    with pytest.raises(ValueError,match='Unapproved'):fit(f,cut,features=['buts'])

def test_duplicate_identity_and_date_are_rejected():
    x=sample()
    with pytest.raises(ValueError,match='duplicate'):build_features(pd.concat([x,x.iloc[:1]]))
