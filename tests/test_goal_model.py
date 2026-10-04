import numpy as np
import pandas as pd
import pytest
from fixtures import games_fixture
from test_feature_parity import build, future
from henachel.goal import BASE_FEATURES, FEATURE_FAMILIES, augment_goal_features, goal_labels, fit_goal_model, numeric_features


def test_goal_target_is_not_point_target():
    frame = pd.DataFrame({'buts': [0, 1, 2], 'points': [1, 1, 3]})
    assert goal_labels(frame).tolist() == [0, 1, 1]
    with pytest.raises(ValueError, match='buts'):
        goal_labels(frame.drop(columns='buts'))


@pytest.mark.parametrize('bad', [np.nan, -1, .5, np.inf])
def test_invalid_goal_outcomes_rejected(bad):
    with pytest.raises(ValueError):
        goal_labels(pd.DataFrame({'buts': [bad]}))


def test_goal_label_disagreement_rejected():
    with pytest.raises(ValueError, match='disagrees'):
        goal_labels(pd.DataFrame({'buts': [0], 'a_marque_un_but': [1]}))


@pytest.mark.parametrize('index', [3, 12, 26, 37])
def test_goal_features_historical_future_parity(index):
    raw = games_fixture(); full = augment_goal_features(build(raw))
    cut = sorted(raw.date_match.unique())[index]
    before = build(raw[raw.date_match < cut])
    matches = raw[raw.date_match == cut].drop_duplicates('id_match').copy()
    matches['status'] = 'FUT'; matches[['buts_domicile','buts_exterieur']] = np.nan
    pool = raw[raw.date_match == cut][['id_joueur','team_player_match','nom','position']]
    upcoming = future.build_upcoming_universe(matches, before, raw.drop_duplicates('id_match'), pool, {})
    predicted = augment_goal_features(upcoming, prior_history=before).set_index('id_joueur')
    actual = full[full.date_match == cut].set_index('id_joueur')
    columns = BASE_FEATURES + sum(FEATURE_FAMILIES.values(), [])
    np.testing.assert_allclose(actual[columns], predicted.loc[actual.index, columns], equal_nan=True)


def test_current_and_future_goals_do_not_change_features():
    raw = games_fixture(); cut = sorted(raw.date_match.unique())[30]
    before = augment_goal_features(build(raw))
    changed = raw.copy(); mask = changed.date_match >= cut
    changed.loc[mask, 'buts'] = 20; changed.loc[mask, 'points'] = 20 + changed.loc[mask, 'passes']
    after = augment_goal_features(build(changed))
    cols = BASE_FEATURES + sum(FEATURE_FAMILIES.values(), [])
    pd.testing.assert_frame_equal(before.loc[before.date_match <= cut, cols], after.loc[after.date_match <= cut, cols])


def test_goal_fit_cannot_use_future_labels_or_modify_point_parameters():
    from henachel.point import POINT_PARAMS
    original = POINT_PARAMS.copy()
    frame = augment_goal_features(build(games_fixture()))
    frame['date_match'] = pd.to_datetime(frame.date_match)
    cutoff = sorted(frame.date_match.unique())[35]
    model1, cal1, meta1 = fit_goal_model(frame, cutoff)
    changed = frame.copy(); mask = changed.date_match >= cutoff
    changed.loc[mask, 'buts'] = 1 - changed.loc[mask, 'buts'].gt(0).astype(int)
    changed.loc[mask, 'a_marque_un_but'] = changed.loc[mask, 'buts']
    model2, cal2, meta2 = fit_goal_model(changed, cutoff)
    x = numeric_features(frame, meta1['feature_cols_kept'])
    assert meta1 == meta2 and POINT_PARAMS == original
    np.testing.assert_array_equal(cal1.predict(model1.predict_proba(x)[:,1]),cal2.predict(model2.predict_proba(x)[:,1]))


@pytest.mark.parametrize('forbidden',['buts','a_marque_un_but','points','a_marque_un_point'])
def test_current_match_fields_cannot_enter_goal_model(forbidden):
    with pytest.raises(ValueError,match='pregame'):
        fit_goal_model(pd.DataFrame(),pd.Timestamp('2025-01-01'),features=[forbidden])


def test_goal_comparison_cannot_select_an_unchanged_model():
    from henachel.goal_experiment import paired_comparison
    frame=pd.DataFrame({'date_match':['2025-01-01','2025-01-02','2025-01-03','2025-01-04'],
                        'id_match':[1,2,3,4],'id_joueur':[1]*4,'game_type':[2]*4,
                        'a_marque_un_but':[0,1,0,1],'fold':[0,0,1,1],'probability':[.2,.3,.2,.3]})
    result=paired_comparison(frame,frame.copy())
    assert result['delta_logloss']==0 and result['delta_brier']==0 and not result['passes']
    changed=frame.copy();changed.loc[0,'a_marque_un_but']=1
    with pytest.raises(AssertionError):paired_comparison(frame,changed)


def test_consumed_goal_holdout_cannot_be_used_for_more_trials(monkeypatch,tmp_path):
    import sys
    from henachel.goal_experiment import main
    (tmp_path/'decision.json').write_text('{"holdout_consumed": true}')
    monkeypatch.setattr(sys,'argv',['goal','--features','unused.csv','--output',str(tmp_path),'--phase','models'])
    with pytest.raises(ValueError,match='consumed'):main()


def test_locked_goal_candidate_cannot_be_changed(monkeypatch,tmp_path):
    import sys
    from henachel.goal_experiment import main
    (tmp_path/'candidate_locked.json').write_text('{"name": "drought"}')
    monkeypatch.setattr(sys,'argv',['goal','--features','unused.csv','--output',str(tmp_path),'--phase','families'])
    with pytest.raises(ValueError,match='locked'):main()
