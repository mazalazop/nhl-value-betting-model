import pandas as pd
import numpy as np
import pytest
from conftest import load_script
from fixtures import games_fixture, standings_fixture
from test_feature_parity import build


def test_pp_refresh_prefers_updated_schedule(tmp_path):
    m=load_script('00a_refresh_pp_stats')
    (tmp_path/'data/raw').mkdir(parents=True)
    (tmp_path/'data/final').mkdir(parents=True)
    pd.DataFrame({'date_match':['2024-01-01']}).to_csv(tmp_path/'data/final/base_canonique_v2.csv',index=False)
    pd.DataFrame({'date_match':['2024-01-01','2024-02-01']}).to_csv(tmp_path/'data/raw/matchs.csv',index=False)
    assert str(m.infer_date_range_from_project(tmp_path)[1]) == '2024-02-01'


def test_pp_unknown_and_zero_remain_different():
    from henachel.quality import merge_pp
    raw=games_fixture().head(4)
    pp=raw[['id_match','id_joueur']].head(1).assign(temps_pp=0.)
    with pytest.warns(UserWarning): out,report=merge_pp(raw,pp,min_coverage=.9,policy='warn')
    assert out.temps_pp.iloc[0] == 0
    assert out.temps_pp.iloc[1:].isna().all()
    assert report['coverage']==.25
    with pytest.raises(ValueError,match='PP coverage'): merge_pp(raw,pp,min_coverage=.9,policy='error')


@pytest.mark.parametrize('kind', ['future','stale','wrong_season','api_mismatch'])
def test_invalid_standings_cannot_enter_features(kind):
    raw=games_fixture()
    standings=standings_fixture(raw)
    if kind=='future': standings['standings_lookup_date']=pd.Timestamp('2030-01-01'); standings['date_snapshot']=standings.standings_lookup_date
    if kind=='stale': standings=standings.iloc[:3].copy(); standings['standings_lookup_date']=pd.Timestamp('2020-01-01'); standings['date_snapshot']=standings.standings_lookup_date; standings['api_date']=standings.standings_lookup_date
    if kind=='wrong_season': standings['season_id']=20002001
    if kind=='api_mismatch': standings['api_date']=pd.Timestamp('2030-01-01')
    result=build(raw,standings)
    assert result.standings_context_found.eq(0).all()


def test_yesterday_standings_are_used_and_missing_are_flagged():
    raw=games_fixture()
    standings=standings_fixture(raw)
    result=build(raw,standings[standings.team_abbrev=='TOR'])
    assert result[result.team_player_match=='TOR'].standings_context_found.eq(1).all()
    assert result[result.team_player_match!='TOR'].standings_context_found.eq(0).all()


def test_standings_refresh_requests_previous_days(tmp_path):
    m=load_script('00c_refresh_team_standings')
    p=tmp_path/'base.csv'
    pd.DataFrame({'date_match':['2024-10-02','2024-10-04']}).to_csv(p,index=False)
    assert m.load_target_dates_from_base_match(p,None,None)==['2024-10-01','2024-10-03']

def test_invalid_complete_standings_snapshot_is_refetched():
    refresh=load_script('00c_refresh_team_standings')
    existing=pd.DataFrame({'date_snapshot':['2026-10-01']*32,'api_date':['2026-09-01']*32,'season_id':[20262027]*32,'team_abbrev':[f'T{i}' for i in range(32)]})
    missing,_=refresh.compute_missing_dates(['2026-10-01'],existing)
    assert missing==['2026-10-01']

@pytest.mark.parametrize('value',[-1,float('inf')])
def test_invalid_pp_duration_is_not_observed(value):
    from henachel.quality import merge_pp
    source=pd.DataFrame({'id_match':[1],'id_joueur':[1]})
    with pytest.raises(ValueError):merge_pp(source,source.assign(temps_pp=value))


def test_pp_pagination_recovers_all_rows(monkeypatch):
    from datetime import date
    from urllib.parse import urlparse, parse_qs
    m = load_script('00a_refresh_pp_stats')
    rows = [{"gameId": 2026020001 + i // 36, "playerId": 8470000 + i % 36,
             "ppTimeOnIce": i % 120} for i in range(237)]
    offsets = []
    def fake_get(session, url):
        q = parse_qs(urlparse(url).query)
        start = int(q["start"][0])
        assert q["limit"] == ["100"]
        offsets.append(start)
        return {"data": rows[start:start+100], "total": len(rows)}
    monkeypatch.setattr(m, "safe_json_get", fake_get)
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    assert m.fetch_one_chunk(None, date(2026, 11, 1), date(2026, 11, 25), 2) == rows
    assert offsets == [0, 100, 200]


@pytest.mark.parametrize("failure", ["repeat", "empty", "total_change"])
def test_pp_pagination_rejects_incomplete_or_duplicate_pages(monkeypatch, failure):
    from datetime import date
    from urllib.parse import urlparse, parse_qs
    m = load_script('00a_refresh_pp_stats')
    first = [{"gameId": 2026020001, "playerId": 8470000 + i, "ppTimeOnIce": 0}
             for i in range(100)]
    def fake_get(session, url):
        start = int(parse_qs(urlparse(url).query)["start"][0])
        if start == 0:
            return {"data": first, "total": 200}
        if failure == "repeat":
            return {"data": first, "total": 200}
        if failure == "empty":
            return {"data": [], "total": 200}
        return {"data": [], "total": 201}
    monkeypatch.setattr(m, "safe_json_get", fake_get)
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    with pytest.raises(ValueError):
        m.fetch_one_chunk(None, date(2026, 11, 1), date(2026, 11, 25), 2)


def test_pp_pagination_accepts_empty_period(monkeypatch):
    from datetime import date
    m = load_script('00a_refresh_pp_stats')
    monkeypatch.setattr(m, "safe_json_get", lambda *args: {"data": [], "total": 0})
    assert m.fetch_one_chunk(None, date(2026, 8, 1), date(2026, 8, 31), 2) == []
