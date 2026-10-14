import pandas as pd
from conftest import load_script


def test_incremental_replace_preserves_unrelated_games_and_missing_values():
    from importlib.util import spec_from_file_location,module_from_spec
    from pathlib import Path
    spec=spec_from_file_location('refresh_goal_slate',Path(__file__).parents[1]/'scripts/refresh_goal_slate.py')
    module=module_from_spec(spec);spec.loader.exec_module(module)
    old=pd.DataFrame({'id_match':[1,2],'id_joueur':[11,22],'temps_pp':[None,4.]})
    fresh=pd.DataFrame({'id_match':[2,2],'id_joueur':[22,23],'temps_pp':[None,0.]})
    got=module.replace_games(old,fresh)
    assert len(got)==3
    assert got[got.id_match==1].temps_pp.isna().all()
    assert got[got.id_joueur==22].temps_pp.isna().all()
    assert got[got.id_joueur==23].temps_pp.iloc[0]==0
    pd.testing.assert_frame_equal(module.replace_games(got,fresh),got)


def test_weekly_schedule_uses_nhl_day_not_utc_day():
    from pathlib import Path
    import importlib.util
    spec=importlib.util.spec_from_file_location('refresh',Path(__file__).parents[1]/'scripts/refresh_goal_slate.py')
    refresh=importlib.util.module_from_spec(spec);spec.loader.exec_module(refresh)
    source=refresh.module('fixture_source','00_refresh_sources.py')
    game=dict(id=2026020107,season=20262027,gameType=2,gameState='FUT',startTimeUTC='2026-10-15T00:30:00Z',homeTeam={'abbrev':'COL'},awayTeam={'abbrev':'DAL'})
    row=refresh.schedule_rows({'gameWeek':[{'date':'2026-10-14','games':[game]}]},source)[game['id']]
    assert row['date_match']=='2026-10-14'
    assert row['start_time_utc']=='2026-10-15T00:30:00Z'
