import numpy as np
import pandas as pd
import pytest
from conftest import load_script
from fixtures import games_fixture, standings_fixture

historical = load_script('01_build_base_features')
future = load_script('05_predict_upcoming_games')


def build(raw, standings=None):
    return historical.enrichir_contexte_v2(
        historical.creer_features_temporelles_v2(historical.build_base_canonique(raw)),
        standings=standings)


def differences(index, with_standings):
    raw = games_fixture()
    standings = standings_fixture(raw) if with_standings else None
    full = build(raw, standings)
    target_date = sorted(raw.date_match.unique())[index]
    before = build(raw[raw.date_match < target_date], standings)
    matches = raw[raw.date_match == target_date].drop_duplicates('id_match').copy()
    matches['status'] = 'FUT'
    matches[['buts_domicile', 'buts_exterieur']] = np.nan
    pool = raw[raw.date_match == target_date][['id_joueur', 'team_player_match', 'nom', 'position']]
    by_team = {} if standings is None else dict(tuple(standings.groupby('team_abbrev')))
    predicted = future.build_upcoming_universe(matches, before, raw.drop_duplicates('id_match'), pool, by_team)
    actual = full[full.date_match == target_date].set_index('id_joueur')
    predicted = predicted.set_index('id_joueur')
    diffs = []
    for col in future.FEATURE_WHITELIST:
        if col not in actual:
            diffs.append(col + ': missing historical')
            continue
        a, b = actual[col].astype(float), predicted.loc[actual.index, col].astype(float)
        if not np.allclose(a, b, equal_nan=True, atol=1e-10):
            diffs.append(col)
    return diffs


@pytest.mark.parametrize('index', [3, 11, 12, 16, 24, 25, 26, 37, 49])
@pytest.mark.parametrize('with_standings', [False, True])
def test_historical_future_parity(index, with_standings):
    assert differences(index, with_standings) == []


def test_current_and_future_results_cannot_change_pre_match_features():
    raw = games_fixture()
    original = build(raw)
    cutoff = sorted(raw.date_match.unique())[30]
    changed = raw.copy()
    mask = changed.date_match >= cutoff
    changed.loc[mask, ['buts','passes']] = 40
    changed.loc[mask, 'points'] = 80
    changed.loc[mask, ['tirs','temps_de_glace','temps_pp','buts_domicile','buts_exterieur']] = 100
    rebuilt = build(changed)
    cols = [c for c in future.FEATURE_WHITELIST if c in original]
    pd.testing.assert_frame_equal(original.loc[original.date_match <= cutoff, cols],
                                  rebuilt.loc[rebuilt.date_match <= cutoff, cols])


def test_season_counters_and_missing_window_semantics():
    raw = games_fixture()
    result = build(raw)
    start = result[(result.season_source == '20242025') & (result.id_joueur == 1)].iloc[0]
    assert start.team_games_played_pre_approx == 0
    assert start.games_remaining_team_pre == 82
    row = result[(result.season_source == '20242025') & (result.id_joueur == 1)].iloc[10]
    assert row.team_games_played_pre_approx == 10
    assert row.games_remaining_team_pre == 72
    prior = raw[(raw.id_joueur == 1) & (raw.date_match < row.date_match)]
    assert row.tirs_moy_5 == pytest.approx(prior.tail(5).tirs.mean())
    assert row.passes_moy_10 == pytest.approx(prior.tail(10).passes.mean())


def test_prefix_equivalence():
    raw = games_fixture()
    full = build(raw)
    cutoff = sorted(raw.date_match.unique())[30]
    prefix = build(raw[raw.date_match <= cutoff])
    cols = ['id_match', 'id_joueur'] + future.FEATURE_WHITELIST
    pd.testing.assert_frame_equal(full[full.date_match <= cutoff][cols].reset_index(drop=True),
                                  prefix[cols].reset_index(drop=True))

def test_current_missing_toi_pp_cannot_change_pregame_season_statistics():
    raw=games_fixture();before=build(raw)
    changed=raw.copy();i=10;changed.loc[i,['temps_de_glace','temps_pp']]=np.nan
    after=build(changed)
    key=raw.loc[i,['id_match','id_joueur']]
    mask=(before.id_match==key.id_match)&(before.id_joueur==key.id_joueur)
    cols=['season_toi_before_match','season_pp_before_match','toi_moy_season_pre','pp_moy_season_pre']
    pd.testing.assert_frame_equal(before.loc[mask,cols],after.loc[mask,cols])

def test_unknown_pp_stays_unknown_but_observed_zero_is_zero():
    raw=games_fixture();raw['temps_pp']=np.nan
    unknown=build(raw)
    assert unknown.pp_moy_5.isna().all()
    assert unknown.pp_moy_season_pre.isna().all()
    assert unknown.loc[unknown.season_games_before_match.gt(0),'season_pp_before_match'].isna().all()
    raw['temps_pp']=0.
    known=build(raw);past=known.season_games_before_match.gt(0)
    assert known.loc[past,'pp_moy_5'].eq(0).all()
    assert known.loc[past,'pp_moy_season_pre'].eq(0).all()
