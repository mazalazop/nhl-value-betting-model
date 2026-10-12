import importlib.util
from pathlib import Path
from unittest.mock import Mock
import pytest
spec=importlib.util.spec_from_file_location('research_collect',Path(__file__).parents[1]/'scripts/collect_research_history.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def row():
    return dict(gameId=2022020001,playerId=1,gameDate='2022-10-07',opponentTeamAbbrev='SJS',teamAbbrev='NSH',goals=1,assists=0,points=1,shots=3,timeOnIcePerGame=600)

def test_contract_identity_outcome_and_duplicates():
    r=row();assert m.validate_stats([r],'summary',20222023)=={(2022020001,1)}
    for bad in [dict(r,points=2),dict(r,shots=None),dict(r,playerId=0),dict(r,gameId=2023020001)]:
        with pytest.raises(ValueError):m.validate_stats([bad],'summary',20222023)
    with pytest.raises(ValueError):m.validate_stats([r,r],'summary',20222023)

def test_pp_unknown_stays_unknown_and_zero_is_valid():
    for value in [None,0,120]:
        r=dict(row(),ppTimeOnIce=value)
        m.validate_stats([r],'powerplay',20222023)
        assert r['ppTimeOnIce']==value
    with pytest.raises(ValueError):m.validate_stats([dict(row(),ppTimeOnIce=-1)],'powerplay',20222023)

def test_report_refuses_truncated_or_capped_result():
    for p in [{'data':[row()],'total':2},{'data':[],'total':10000}]:
        cache=Mock();cache.get.return_value=p
        with pytest.raises(ValueError):m.report(cache,'summary',20222023,'2022-10-07','2022-10-07')

def test_successful_http_is_cached_and_failed_http_is_not(tmp_path,monkeypatch):
    monkeypatch.setattr(m.time,'sleep',lambda _:None)
    response=Mock();response.status_code=200;response.json.return_value={'data':[row()]}
    get=Mock(return_value=response);monkeypatch.setattr(m.requests,'get',get)
    cache=m.Cache(tmp_path/'cache')
    assert cache.get('https://example.test/data')==cache.get('https://example.test/data')
    assert get.call_count==1 and cache.hits==1
    response.raise_for_status.side_effect=m.requests.HTTPError('failed')
    with pytest.raises(m.requests.HTTPError):cache.get('https://example.test/failure')
    assert len(list(cache.root.glob('*.gz')))==1


def schedule_game(**overrides):
    game=dict(id=2022020001,season=20222023,gameType=2,gameDate='2022-10-07',
              gameState='OFF',
              homeTeam={'abbrev':'NSH','score':4},
              awayTeam={'abbrev':'SJS','score':1},
              venue={'default':'O2 Arena'},
              tvBroadcasts=[{'network':'A'}])
    game.update(overrides)
    return game

def test_schedule_duplicates_ignore_noncanonical_metadata_and_final_alias():
    games={}
    first=schedule_game()
    second=schedule_game(gameState='FINAL',venue={'default':'Different label'},
                         tvBroadcasts=[{'network':'B'}],gameCenterLink='/different')
    m.merge_schedule_game(games,first,20222023)
    m.merge_schedule_game(games,second,20222023)
    assert list(games)==[2022020001]
    assert games[2022020001]=={
        'id':2022020001,'season':20222023,'gameType':2,'gameDate':'2022-10-07',
        'gameState':'FINAL',
        'homeTeam':{'abbrev':'NSH','score':4},
        'awayTeam':{'abbrev':'SJS','score':1},
    }

def test_schedule_duplicates_reject_real_game_contract_conflicts():
    for conflicting in [
        schedule_game(gameDate='2022-10-08'),
        schedule_game(homeTeam={'abbrev':'NSH','score':5}),
        schedule_game(awayTeam={'abbrev':'SEA','score':1}),
    ]:
        games={}
        m.merge_schedule_game(games,schedule_game(),20222023)
        with pytest.raises(ValueError,match='Conflicting canonical schedule game'):
            m.merge_schedule_game(games,conflicting,20222023)

def test_schedule_contract_rejects_nonfinal_or_missing_score():
    with pytest.raises(ValueError,match='Non-final'):
        m.canonical_schedule_game(schedule_game(gameState='FUT'),20222023)
    with pytest.raises(ValueError,match='final score'):
        m.canonical_schedule_game(schedule_game(homeTeam={'abbrev':'NSH'}),20222023)
