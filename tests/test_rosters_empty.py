import pandas as pd
import pytest
from conftest import load_script
from fixtures import games_fixture
from henachel.rosters import choose_players
from henachel.features import build_future_features
future=load_script('05_predict_upcoming_games')


def test_current_roster_rookie_trade_and_long_return():
    hist=games_fixture();now=pd.Timestamp('2026-10-02T10:00:00Z')
    roster=pd.DataFrame([dict(id_joueur=i,nom=f'Player {i}',position='C',id_equipe='UTA',observed_at=now) for i in [1,99]])
    pool=choose_players(hist,roster,['UTA'],'2026-10-02',now)
    assert set(pool.id_joueur)=={1,99};assert pool.team_player_match.eq('UTA').all()
    assert pool.loc[pool.id_joueur.eq(1),'days_since_last_game'].iloc[0]>45
    match=pd.DataFrame([dict(id_match=2026020001,date_match='2026-10-02',saison=20262027,status='FUT',id_equipe_domicile='UTA',id_equipe_exterieur='TOR',start_time_utc='2026-10-03T00:00:00Z')])
    out=build_future_features(hist,match,pool)
    assert set(out.id_joueur)=={1,99};assert out.id_equipe_domicile.eq('UTA').all()
    assert out.loc[out.id_joueur.eq(99),'nb_matchs_avant_match'].iloc[0]==0

def test_fallback_does_not_override_schedule_with_last_match():
    hist=games_fixture();target=pd.Timestamp(hist.date_match.max())+pd.Timedelta(days=2)
    with pytest.warns(UserWarning):pool=choose_players(hist,None,['TOR'],target,now=pd.Timestamp(target,tz='UTC'))
    match=pd.DataFrame([dict(id_match=9999,date_match=target,saison=20242025,status='FUT',id_equipe_domicile='MTL',id_equipe_exterieur='TOR',start_time_utc='2024-12-03T00:00:00Z')])
    out=build_future_features(hist,match,pool)
    assert out.is_home_player.eq(0).all();assert out.id_equipe_domicile.eq('MTL').all()

def test_empty_slate_is_normal():
    matches=pd.DataFrame([dict(id_match=1,date_match=pd.Timestamp('2026-10-02'),status='OFF')])
    date=future.choose_target_date(matches,'2026-10-03')
    assert future.select_future_matches(matches,date).empty

def test_stale_roster_explicit_fallback():
    hist=games_fixture();r=pd.DataFrame([dict(id_joueur=99,nom='Rookie',position='C',id_equipe='TOR',observed_at='2020-01-01T00:00:00Z')])
    with pytest.warns(UserWarning):pool=choose_players(hist,r,['TOR'],'2026-10-02',pd.Timestamp('2026-10-02T10:00:00Z'))
    assert pool.empty
