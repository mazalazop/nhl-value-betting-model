import numpy as np
import pandas as pd
import pytest
from fixtures import games_fixture
from test_feature_parity import build,future
from henachel.goal_operational import FEATURES,refit,score,admissible_history,top_ranking,STATUS
from henachel.goal import augment_goal_features,fit_goal_model,numeric_features


def fixture():
    raw=games_fixture();raw['game_type']=2
    history=build(raw);history['game_type']=2
    cutoff=sorted(raw.date_match.unique())[37]
    return raw,history,cutoff


def test_operational_refit_exact_reference_and_future_labels_quarantined():
    raw,history,cutoff=fixture()
    bundle,past=refit(history,cutoff)
    assert past.date_match.max()<cutoff
    assert pd.Timestamp(bundle['summary']['fit']['calib_end'])<cutoff
    changed=history.copy();changed.loc[changed.date_match>=cutoff,['buts','points','a_marque_un_but']]=np.nan
    second,_=refit(changed,cutoff)
    derived=augment_goal_features(past)
    model,cal,meta=fit_goal_model(derived,cutoff,FEATURES)
    x=numeric_features(derived,bundle['features'])
    expected=cal.predict(model.predict_proba(x)[:,1])
    for b in [bundle,second]:
        np.testing.assert_array_equal(b['calibrator'].predict(b['model'].predict_proba(x)[:,1]),expected)
    assert bundle['summary']['model_status']==STATUS
    assert derived.tirs_moy_5.isna().any() or history.tirs.isna().any()


def test_operational_future_parity_probabilities_and_current_outcomes_refused():
    raw,history,cutoff=fixture();bundle,past=refit(history,cutoff)
    matches=raw[raw.date_match==cutoff].drop_duplicates('id_match').copy();matches['status']='FUT'
    matches[['buts_domicile','buts_exterieur']]=np.nan
    pool=raw[raw.date_match==cutoff][['id_joueur','team_player_match','nom','position']]
    upcoming=future.build_upcoming_universe(matches,past,raw.drop_duplicates('id_match'),pool,{})
    upcoming['start_time_utc']=(pd.Timestamp(cutoff).tz_localize('UTC')+pd.Timedelta(hours=20)).isoformat()
    out,features=score(bundle,past,upcoming,cutoff,now=pd.Timestamp(cutoff).tz_localize('UTC'))
    actual=augment_goal_features(history);actual=actual[actual.date_match==cutoff]
    np.testing.assert_allclose(features.sort_values('id_joueur')[FEATURES],actual.sort_values('id_joueur')[FEATURES],equal_nan=True)
    assert out.probability_goal.between(0,1).all() and out.model_status.eq(STATUS).all()
    assert top_ranking(out,10,'PP1').empty
    with pytest.raises(ValueError,match='Current outcomes'):
        score(bundle,past,upcoming.assign(buts=0),cutoff,now=pd.Timestamp(cutoff).tz_localize('UTC'))


def test_only_final_history_used_and_missing_not_zero():
    _,history,cutoff=fixture();history.loc[0,'status']='LIVE'
    past=admissible_history(history,cutoff)
    assert len(past)==len(history[(history.date_match<cutoff)&history.status.eq('OFF')])
    assert past.temps_pp.isna().any()


def test_pp_filter_requires_verified_evidence_and_preserves_probabilities():
    frame=pd.DataFrame(dict(goal_probability_rank=[1,2,3],probability_goal=[.6,.5,.4],
                            pp_status=['PP1','PP1','UNKNOWN'],evidence_status=['UNKNOWN','VERIFIED_FRESH','UNKNOWN']))
    filtered=top_ranking(frame,10,'PP1')
    assert filtered.goal_probability_rank.tolist()==[2]
    assert filtered.probability_goal.tolist()==[.5]
    assert frame.probability_goal.tolist()==[.6,.5,.4]


def test_future_pre_is_allowed_but_started_and_postponed_are_not():
    from henachel.goal_operational import upcoming_slate
    frame=pd.DataFrame(dict(id_match=[1,2,3,4],date_match=['2026-10-14']*4,
                            status=['FUT','PRE','PRE','FUT'],schedule_state=['OK','OK','OK','PPD'],
                            start_time_utc=['2026-10-15T00:00:00Z','2026-10-14T23:00:00Z','2026-10-14T12:00:00Z','2026-10-15T00:00:00Z']))
    got=upcoming_slate(frame,'2026-10-14',now='2026-10-14T20:00:00Z')
    assert got.id_match.tolist()==[1,2]
