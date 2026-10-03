"""Regression cases exposed by live NHL data, independent of network availability."""
import pandas as pd
from conftest import load_script
from fixtures import games_fixture
from test_feature_parity import build


def test_2026_27_regular_schedule_has_84_games_in_standings_context():
    refresh=load_script('00c_refresh_team_standings')
    row=refresh.flatten_team_record({'seasonId':20262027,'gamesPlayed':2,'date':'2026-10-02'},'2026-10-02')
    assert row['games_remaining']==82
    old=refresh.flatten_team_record({'seasonId':20252026,'gamesPlayed':2,'date':'2025-10-02'},'2025-10-02')
    assert old['games_remaining']==80


def test_2026_27_fallback_uses_actual_season_length():
    raw=games_fixture();raw=raw[raw.season_source.astype(str).eq('20242025')].copy()
    raw['season_source']='20262027';raw['saison']=20262027;raw['date_match']=pd.to_datetime(raw.date_match)+pd.DateOffset(years=2)
    frame=build(raw);first=frame[frame.team_games_played_pre_approx.eq(0)]
    assert first.games_remaining_team_pre.eq(84).all()
