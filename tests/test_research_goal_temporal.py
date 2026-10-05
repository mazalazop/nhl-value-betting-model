import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location('research_goal_temporal', Path(__file__).resolve().parents[1] / 'scripts/research_goal_temporal.py')
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


def fixture():
    n = 12
    return pd.DataFrame(dict(id_joueur=[1]*n, id_match=range(n), date_match=pd.date_range('2023-01-01', periods=n),
                             season_source=[20222023]*n, game_type=[2]*n, tirs=[0, 1, 2, 3, np.nan, 0, 6, 7, 8, 9, 10, 11],
                             team_player_match=['A']*8+['C']*4, adversaire_match=['B']*n,
                             tirs_moy_5=[2.]*n, pp_moy_5=[1.]*n, goal_relative_drought_pre=[.3]*n))


def test_frozen_protocol_and_no_production_promotion():
    p = r.frozen_protocol()
    assert p['holdout_access'] is False
    assert p['parameters']['max_depth'] == 4
    assert len(p['baseline']) == 17
    assert len(p['families']) == 3


@pytest.mark.parametrize('index', [1, 5, 8, 11])
def test_research_features_future_parity(index):
    source = fixture()
    historical = r.augment_research_features(source)
    future = source.iloc[[index]].copy()
    future['tirs'] = 999
    actual = r.augment_research_features(future, prior_history=source.iloc[:index])
    cols = sum(r.FAMILIES.values(), [])
    np.testing.assert_allclose(actual[cols], historical.iloc[[index]][cols], equal_nan=True)


def test_mutating_current_and_future_cannot_change_pregame_features():
    source = fixture()
    before = r.augment_research_features(source)
    source.loc[6:, 'tirs'] = 999
    after = r.augment_research_features(source)
    cols = sum(r.FAMILIES.values(), [])
    pd.testing.assert_frame_equal(before.loc[:6, cols], after.loc[:6, cols])


def test_missing_is_not_zero_and_cv_needs_five_observations():
    source = fixture()
    source.loc[:4, 'tirs'] = np.nan
    result = r.augment_research_features(source)
    assert np.isnan(result.loc[4, 'shots_mean3_pre'])
    assert np.isnan(result.loc[9, 'shots_cv10_pre'])
    source.loc[:4, 'tirs'] = 0
    result = r.augment_research_features(source)
    assert result.loc[4, 'shots_mean3_pre'] == 0
    assert np.isnan(result.loc[5, 'shots_cv10_pre'])


def test_congestion_unique_games_and_transfer():
    source = fixture()
    second = source.copy(); second.id_joueur = 2
    rows = r.augment_research_features(pd.concat([source, second]))
    one = rows[rows.id_joueur.eq(1)].reset_index(drop=True)
    assert one.loc[4, 'team_games_previous4days_pre'] == 4
    assert np.isnan(one.loc[0, 'player_team_changed_pre'])
    assert one.loc[8, 'player_team_changed_pre'] == 1
    assert one.loc[9, 'player_team_changed_pre'] == 0


def test_opponent_only_games_contribute_to_congestion():
    source = fixture()
    source.loc[4, 'team_player_match'] = 'B'
    source.loc[4, 'adversaire_match'] = 'C'
    result = r.augment_research_features(source)
    assert result.loc[4, 'team_games_previous4days_pre'] == 4


def test_holdout_excluded_before_invalid_labels_are_examined():
    source = fixture()
    future = source.iloc[[0]].copy()
    future.date_match = pd.Timestamp('2024-02-01'); future.season_source = 20232024
    future['buts'] = 'invalid_holdout_label'
    result = r.development_only(pd.concat([source, future]))
    assert len(result) == len(source)
    assert result.date_match.max() < pd.Timestamp('2024-02-01')


def test_consumed_old_holdout_season_refused():
    source = fixture(); source.season_source = 20252026
    with pytest.raises(ValueError, match='Only additional'):
        r.development_only(source)


def test_not_strictly_prior_history_refused():
    source = fixture()
    with pytest.raises(ValueError, match='strictly prior'):
        r.augment_research_features(source.iloc[[4]], source.iloc[:5])


def test_fit_current_labels_and_undeclared_features_refused():
    source = fixture()
    with pytest.raises(ValueError, match='Undeclared'):
        r.fit(source, '2023-02-01', ['buts'])
    with pytest.raises(ValueError, match='Holdout inaccessible'):
        r.fit(source, '2024-03-01', r.BASELINE)


def test_fit_future_labels_do_not_affect_training_or_calibration(monkeypatch):
    n = 50
    source = pd.DataFrame(dict(date_match=pd.date_range('2023-01-01', periods=n), id_match=range(n),
                               id_joueur=[1]*n, season_source=[20222023]*n, game_type=[2]*n,
                               buts=np.arange(n)%2, shots_mean3_pre=np.arange(n, dtype=float)))
    observations = []
    class FakeModel:
        def __init__(self, **parameters):
            assert parameters == r.GOAL_PARAMS
        def fit(self, x, y, sample_weight):
            observations.append((x.copy(), y.copy(), sample_weight.copy()))
        def predict_proba(self, x):
            return np.tile([.7, .3], (len(x), 1))
    def calibrate(raw, y, dates):
        observations.append((raw.copy(), y.copy(), dates.copy()))
        return object(), {'method': 'test'}
    monkeypatch.setattr(r, 'HistGradientBoostingClassifier', FakeModel)
    monkeypatch.setattr(r, 'select_calibrator', calibrate)
    r.fit(source, '2023-02-10', ['shots_mean3_pre'])
    source.loc[source.date_match >= '2023-02-10', 'buts'] = -999
    source.loc[source.date_match >= '2023-02-10', 'shots_mean3_pre'] = -999
    r.fit(source, '2023-02-10', ['shots_mean3_pre'])
    for first, second in zip(observations[:2], observations[2:]):
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)


def test_changed_protocol_refused(monkeypatch, tmp_path):
    path = tmp_path / 'protocol.json'
    path.write_text('{}')
    monkeypatch.setattr(r, 'PROTOCOL_PATH', path)
    with pytest.raises(ValueError, match='Frozen'):
        r.frozen_protocol()


def test_baseline_fit_exactly_matches_existing_goal_model():
    from fixtures import games_fixture
    from test_feature_parity import build
    from henachel.goal import augment_goal_features, fit_goal_model, numeric_features
    frame = augment_goal_features(build(games_fixture()))
    frame['date_match'] = pd.to_datetime(frame.date_match) - pd.DateOffset(years=2)
    frame['season_source'] = 20222023
    frame['game_type'] = 2
    cutoff = sorted(frame.date_match.unique())[35]
    original, original_cal, original_meta = fit_goal_model(frame, cutoff, r.BASELINE)
    research, research_cal, research_meta = r.fit(frame, cutoff, r.BASELINE)
    assert research_meta['kept'] == original_meta['feature_cols_kept']
    x = numeric_features(frame, research_meta['kept'])
    np.testing.assert_array_equal(original.predict_proba(x), research.predict_proba(x))
    np.testing.assert_array_equal(original_cal.predict(original.predict_proba(x)[:, 1]), research_cal.predict(research.predict_proba(x)[:, 1]))


def test_runtime_refuses_missing_final_status_before_fit(monkeypatch, tmp_path):
    source = fixture()
    source['id_match'] += 1
    source['is_home_player'] = 1
    source['id_equipe_domicile'] = source.team_player_match
    source['id_equipe_exterieur'] = source.adversaire_match
    source['buts'] = 0; source['points'] = 0; source['passes'] = 0
    path = tmp_path / 'source.csv.gz'; source.to_csv(path, index=False)
    with pytest.raises(ValueError, match='explicit final'):
        r.run(path, tmp_path / 'outputs')
    assert not (tmp_path / 'outputs').exists()
