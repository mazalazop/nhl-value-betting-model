import numpy as np
import pandas as pd
import pytest
from henachel.goal_pk import add_opponent_pk, FEATURE


def source():
    return pd.DataFrame({'id_match':[1,2,3],'date_match':['2025-01-01','2025-01-03','2025-01-05'],
                         'season_source':[20242025]*3,'team':['BOS']*3,
                         'pp_goals_against':[1,0,3],'times_shorthanded':[2,2,3]})


def frame(dates,season=20242025):
    return pd.DataFrame({'date_match':dates,'season_source':[season]*len(dates),'adversaire_match':['BOS']*len(dates)})


def test_pk_excludes_current_date_and_future_results():
    pk=source();requests=frame(['2025-01-01','2025-01-03','2025-01-04'])
    values=add_opponent_pk(requests,pk)[FEATURE]
    assert np.isnan(values.iloc[0]);assert values.iloc[1]==.5;assert values.iloc[2]==.75
    pk.loc[2,'pp_goals_against']=0
    pd.testing.assert_series_equal(values,add_opponent_pk(requests,pk)[FEATURE])


def test_pk_zero_opportunities_new_season_and_stale_are_unknown():
    pk=source();pk[['pp_goals_against','times_shorthanded']]=0
    assert add_opponent_pk(frame(['2025-01-04']),pk)[FEATURE].isna().all()
    assert add_opponent_pk(frame(['2025-01-04'],20252026),source())[FEATURE].isna().all()
    assert add_opponent_pk(frame(['2025-03-01']),source())[FEATURE].isna().all()


def test_pk_historical_future_prefix_parity():
    requests=frame(['2025-01-04'])
    pd.testing.assert_frame_equal(add_opponent_pk(requests,source()),add_opponent_pk(requests,source().iloc[:2]))


def test_pk_duplicate_identity_rejected():
    pk=source()
    with pytest.raises(ValueError,match='Duplicate'):
        add_opponent_pk(frame(['2025-01-04']),pd.concat([pk,pk.iloc[:1]]))
