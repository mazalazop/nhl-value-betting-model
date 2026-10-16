import copy,json
from pathlib import Path
import pandas as pd
import pytest
from henachel.unibet_ingestion import normalize_event
from henachel.goal_odds import match_goal_odds


@pytest.fixture
def contract():
    source=json.loads((Path(__file__).parent/'data/unibet_real_2026_10_03.json').read_text())
    # Same public schema observed for group 4266; explicit synthetic market fixture.
    event=copy.deepcopy(source['event'])
    for group in event['groupedMarkets']:
        if group['id']!=4267:continue
        group['id']=4266;group['description']=group['description'].replace('Points','Buts')
        for market in group['markets']:
            market['description']=market['description'].replace('Points','Buts')
            for outcome in market['outcomes']:outcome['groupId']=4266
    rows,rejected=normalize_event(event,source['url'],source['captured_at'],pd.DataFrame(source['matches']),pd.DataFrame(source['roster']),source['captured_at'],market_kind='goals')
    assert len(rows)==1 and not rejected
    r=rows[0]
    model=pd.DataFrame([dict(game_id=r['nhl_game_id'],player_id=r['nhl_player_id'],player_name=r['player_name'],team=r['team'],opponent='BUF',
        event_start_utc=r['event_start_utc'],game_date=r['date_match'],probability_goal=.6,goal_probability_rank=1,model_status='research_only_operational')])
    return model,rows,source['captured_at']


def test_goal_identity_value_and_probability_unchanged(contract):
    model,rows,now=contract
    got,rejected=match_goal_odds(model,rows,now)
    assert not rejected and len(got)==1
    assert got.probability_goal.iloc[0]==.6
    assert got.implied_probability.iloc[0]==.5
    assert got.edge.iloc[0]==pytest.approx(.1)
    assert got.expected_value.iloc[0]==pytest.approx(.2)
    assert got.experimental_value_eligible.all()


@pytest.mark.parametrize('key,value',[('threshold',2),('market','player_points'),('stat','points'),('odds_decimal',float('nan')),('odds_decimal',1),('event_id',None),('team',''),('nhl_player_id',float('nan')),('nhl_player_id',999),('player_name','Someone Else'),('collected_at_utc',None),('collected_at_utc','2026-09-01T00:00:00Z'),('suspended',True),('event_start_utc','2026-10-04T00:00:00Z')])
def test_goal_rejections(contract,key,value):
    model,rows,now=contract;rows[0][key]=value
    got,rejected=match_goal_odds(model,rows,now)
    assert got.empty and len(rejected)==1


def test_goal_duplicate_or_ambiguous_event_rejected(contract):
    model,rows,now=contract
    assert match_goal_odds(model,rows*2,now)[0].empty
    other=dict(rows[0],nhl_game_id=999)
    assert match_goal_odds(model,rows+[other],now)[0].empty


def test_no_odds_keeps_probability_ranking(contract):
    model,_,now=contract
    got,rejected=match_goal_odds(model,[],now)
    assert got.empty and not rejected and model.probability_goal.iloc[0]==.6


def test_annotated_ranking_preserves_all_players_with_or_without_odds(contract):
    from henachel.goal_odds import annotate_probability_ranking
    model,rows,now=contract
    extra=model.copy();extra['player_id']=999;extra['probability_goal']=.2;extra['goal_probability_rank']=2
    predictions=pd.concat([model,extra],ignore_index=True)
    for odds in [rows, []]:
        matched,_=match_goal_odds(predictions,odds,now)
        ranking=annotate_probability_ranking(predictions,matched)
        pd.testing.assert_frame_equal(ranking[predictions.columns],predictions)
        assert pd.isna(ranking.iloc[1].odds_decimal)
        assert ranking.odds_decimal.notna().sum()==len(odds)


def test_collector_goal_flag_does_not_change_point_payload(tmp_path,monkeypatch):
    import importlib.util
    import sys
    from types import SimpleNamespace
    from unittest.mock import Mock
    source=json.loads((Path(__file__).parent/'data/unibet_real_2026_10_03.json').read_text())
    spec=importlib.util.spec_from_file_location('collector',Path(__file__).parents[1]/'scripts/collect_unibet_structured.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    raw=tmp_path/'raw';raw.mkdir()
    pd.DataFrame(source['matches']).to_csv(raw/'matchs.csv',index=False)
    pd.DataFrame(source['roster']).to_csv(raw/'roster_current.csv',index=False)
    response=Mock();response.status_code=200;response.headers={}
    response.text=source['url']
    monkeypatch.setattr(module.requests,'get',lambda *a,**kw:response)
    monkeypatch.setattr(module,'public_events',lambda text:[source['event']])
    module.pd=SimpleNamespace(read_csv=pd.read_csv,Timestamp=SimpleNamespace(now=lambda **kw:pd.Timestamp(source['captured_at'])))
    for enabled in [False,True]:
        output=tmp_path/str(enabled)
        argv=['collect','--raw-dir',str(raw),'--output-dir',str(output)]+(['--include-goal'] if enabled else [])
        monkeypatch.setattr(sys,'argv',argv);module.main()
        assert (output/'normalized_goals_odds.json').exists()==enabled
    assert (tmp_path/'False/normalized_points_odds.json').read_bytes()==(tmp_path/'True/normalized_points_odds.json').read_bytes()
