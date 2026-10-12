import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from structural_features import structural_features
from collect_structural_standings import flatten


def fixture():
    dates = list(pd.date_range('2022-10-10', periods=8)) + list(pd.date_range('2023-10-10', periods=8))
    rows=[]; context=[]
    for i, date in enumerate(dates):
        season=20222023 if i<8 else 20232024
        for player,team,opp in [(1,'BOS','MTL'),(2,'MTL','BOS')]:
            rows.append(dict(id_joueur=player,id_match=i,date_match=date,season_source=season,team_player_match=team,adversaire_match=opp,
                             points=i%3,buts=i%2,tirs=3.,temps_de_glace=18.,temps_pp=2.))
            context.append(dict(id_match=i,date_match=date,season_source=season,team=team,opponent=opp,ppTimeOnIcePerGame=240.,
                                ppOpportunities=3.,totalPenaltiesDrawn=4.,penalties=5.,timesShorthanded=4.,powerPlayGoalsFor=1.,ppGoalsAgainst=1.))
    return pd.DataFrame(rows),pd.DataFrame(context)


def columns(frame):
    return [c for c in frame if c.startswith('s_') or c.endswith('_pre') or c in ['expected_pp_opportunities','player_pp_exposure_interaction']]


@pytest.mark.parametrize('index',[0,5,8,9,15])
def test_structural_historical_future_parity(index):
    source,ctx=fixture(); date=source.date_match.unique()[index]
    expected=structural_features(source,ctx); future=source[source.date_match.eq(date)].copy()
    future[['points','buts','tirs','temps_de_glace','temps_pp']]=999.
    actual=structural_features(future,ctx,source[source.date_match.lt(date)])
    np.testing.assert_allclose(actual[columns(actual)],expected[expected.date_match.eq(date)][columns(actual)],equal_nan=True)


def test_current_future_outcomes_and_team_context_never_leak():
    source,ctx=fixture(); before=structural_features(source,ctx); date=pd.Timestamp('2023-10-13')
    source.loc[source.date_match.ge(date),['points','buts','tirs','temps_pp']]=99.
    ctx.loc[ctx.date_match.ge(date),['ppOpportunities','powerPlayGoalsFor','ppTimeOnIcePerGame']]=999.
    after=structural_features(source,ctx)
    pd.testing.assert_frame_equal(before.loc[before.date_match.le(date),columns(before)],after.loc[after.date_match.le(date),columns(after)])


def test_previous_season_blend_counts_and_seconds_conversion():
    source,ctx=fixture(); out=structural_features(source,ctx); start=out[(out.date_match.eq('2023-10-10')) & out.id_joueur.eq(1)].iloc[0]
    assert start.s_games_pre==0 and start.s_prev_games==8
    assert start.s_points_blend10==pytest.approx(source[(source.season_source==20222023)&source.id_joueur.eq(1)].points.mean())
    assert np.isnan(start.s_points_current_mean)
    assert start.s_share_prev==.5 and np.isnan(start.s_share_current)
    last=out.iloc[-1];assert last.expected_pp_opportunities==3.5 and last.player_pp_exposure_interaction==1.75


@pytest.mark.parametrize('value',[np.nan,0.])
def test_unknown_pp_different_from_real_zero(value):
    source,ctx=fixture();source.temps_pp=value
    out=structural_features(source,ctx).iloc[-1]
    if np.isnan(value):assert np.isnan(out.s_pp_current_mean) and np.isnan(out.s_share_current)
    else:assert out.s_pp_current_mean==0 and out.s_share_current==0


def test_team_unknown_not_zero_and_minimum_exposure():
    source,ctx=fixture();ctx.ppOpportunities=np.nan
    out=structural_features(source,ctx)
    assert out.expected_pp_opportunities.isna().all() and out.team_pp_efficiency_pre.isna().all()


def test_duplicate_and_inconsistent_context_refused():
    source,ctx=fixture()
    with pytest.raises(ValueError,match='Ambiguous'):structural_features(pd.concat([source,source.iloc[:1]]),ctx)
    ctx.loc[0,'opponent']='TOR'
    with pytest.raises(ValueError,match='identity'):structural_features(source,ctx)


def test_strict_prior_history():
    source,ctx=fixture()
    with pytest.raises(ValueError,match='strictly prior'):structural_features(source.iloc[-2:],ctx,source)


def test_standings_empty_explicit_and_wrong_date_refused():
    assert flatten({'standings':[]},'2022-10-06',20222023)==[]
    with pytest.raises(ValueError,match='date/season'):
        flatten({'standings':[{'date':'2022-10-05','seasonId':20222023}]},'2022-10-06',20222023)


def test_production_string_seasons_match_official_integer_seasons():
    source,ctx=fixture();expected=structural_features(source,ctx)
    source.season_source=source.season_source.astype(str)
    actual=structural_features(source,ctx)
    pd.testing.assert_frame_equal(expected,actual)
