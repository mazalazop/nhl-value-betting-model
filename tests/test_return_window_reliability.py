"""Regression tests for the finite NHL player-return proxy (not medical injury)."""
import pandas as pd
import pytest
from conftest import load_script
from fixtures import games_fixture

model=load_script("01_build_base_features")
from henachel.point_return_patch import repair_point_returns

def dataset():
    raw=games_fixture()
    return repair_point_returns(model.enrichir_contexte_v2(
        model.creer_features_temporelles_v2(model.build_base_canonique(raw))
    ))

@pytest.mark.parametrize("season", ["20232024","20242025"])
def test_return_signal_expires_and_default_weight_restored(season):
    x=dataset()
    p=x[(x.id_joueur==1)&(x.season_source==season)].sort_values("date_match").reset_index(drop=True)
    assert len(p)==25
    # Fixture: 14-day schedule gap immediately before the 13th player game.
    assert p.loc[12,"absence_longue_flag"]==1
    assert p.loc[12:14,"return_from_absence_flag"].eq(1).all()
    assert p.loc[15:,"return_from_absence_flag"].eq(0).all()
    assert p.loc[12:14,"return_stabilized_flag"].eq(0).all()
    assert p.loc[21:,"return_stabilized_flag"].eq(0).all()
    assert p.loc[21:,"historical_current_weight"].eq(.60).all()
    assert p.loc[21:,"historical_prev_weight"].eq(.40).all()

def test_prior_season_absence_does_not_leak_into_new_season():
    x=dataset()
    p=x[(x.id_joueur==1)&(x.season_source=="20242025")].sort_values("date_match").reset_index(drop=True)
    assert p.loc[0,"retour_episode"]==0
    assert p.loc[0,"return_from_absence_flag"]==0
    assert p.loc[0,"return_stabilized_flag"]==0
    assert p.loc[0,"historical_current_weight"]==pytest.approx(.6)

def test_repeated_episodes_restart_in_same_season():
    raw=games_fixture()
    # A second gap is inserted well after the first one.
    mask=(raw.id_joueur==1)&(raw.season_source=="20232024")&(raw.date_match>=pd.Timestamp("2023-11-20"))
    raw.loc[mask,"date_match"] += pd.Timedelta(days=14)
    x=repair_point_returns(model.enrichir_contexte_v2(model.creer_features_temporelles_v2(model.build_base_canonique(raw))))
    p=x[(x.id_joueur==1)&(x.season_source=="20232024")].sort_values("date_match").reset_index(drop=True)
    episodes=p.loc[p.absence_longue_flag.eq(1)]
    assert len(episodes)>=2
    for _, row in episodes.iterrows():
        assert row.return_from_absence_flag==1
        assert row.matchs_depuis_retour_avant_match==0

def test_return_signal_not_changed_by_current_result():
    raw=games_fixture()
    original=repair_point_returns(model.enrichir_contexte_v2(model.creer_features_temporelles_v2(model.build_base_canonique(raw))))
    manipulated=raw.copy()
    manipulated["buts"]=20
    manipulated["passes"]=30
    manipulated["points"]=50
    manipulated["tirs"]=50
    manipulated["temps_de_glace"]=70.
    changed=repair_point_returns(model.enrichir_contexte_v2(model.creer_features_temporelles_v2(model.build_base_canonique(manipulated))))
    expected=original[["return_from_absence_flag","retour_episode","matchs_depuis_retour_avant_match"]]
    observed=changed[["return_from_absence_flag","retour_episode","matchs_depuis_retour_avant_match"]]
    pd.testing.assert_frame_equal(expected,observed)

def test_invalid_windows_fail_closed():
    raw=games_fixture()
    base=model.creer_features_temporelles_v2(model.build_base_canonique(raw))
    enriched=model.enrichir_contexte_v2(base)
    with pytest.raises(ValueError,match="finite return windows"):
        repair_point_returns(enriched,recent_games=10,stabilized_until_games=8)
