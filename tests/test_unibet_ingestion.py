"""Contract tests derived from the real public Unibet document captured 2026-10-03."""
import copy
import json
from pathlib import Path
import pandas as pd
import pytest
from conftest import load_script
from henachel.unibet_ingestion import normalize_event,normalized_payload,public_events
from henachel.bookmaker_contract import validate_rows


@pytest.fixture
def source():
    return json.loads((Path(__file__).parent/'data/unibet_real_2026_10_03.json').read_text())


def parse(source):
    return normalize_event(source['event'],source['url'],source['captured_at'],
        pd.DataFrame(source['matches']),pd.DataFrame(source['roster']),source['captured_at'])


def test_real_document_obeys_existing_06_schema(source,tmp_path):
    rows,rejected=parse(source)
    assert len(rows)==1 and not rejected
    assert rows[0]['player_name']=='Tyler Bertuzzi'
    assert rows[0]['team']=='CHI' and rows[0]['home_team']=='BUF'
    assert rows[0]['captured_at']==rows[0]['collected_at_utc']==source['captured_at']
    report=validate_rows(rows,source['captured_at']);assert report['status']=='ok'
    path=tmp_path/'odds.json';path.write_text(json.dumps(normalized_payload(rows,[],report)))
    _,loaded=load_script('06_match_model_to_unibet_odds').load_odds_json(path)
    assert len(loaded)==1 and loaded.odds_decimal.iloc[0]==2.0


@pytest.mark.parametrize('case',['no_game','ambiguous_game','wrong_time','unknown_team','missing_start','missing_capture','stale'])
def test_invalid_event_rejected(source,case):
    if case=='no_game':source['matches'][0]['id_equipe_domicile']='TOR'
    if case=='ambiguous_game':source['matches']*=2
    if case=='wrong_time':source['event']['parsedStart']='2026-10-04T23:00:00Z'
    if case=='unknown_team':source['event']['opponentA']['label']='Unknown'
    if case=='missing_start':source['event']['parsedStart']=None
    if case=='missing_capture':source['captured_at']=None
    if case=='stale':
        with pytest.raises(ValueError):
            normalize_event(source['event'],source['url'],pd.Timestamp(source['captured_at'])-pd.Timedelta(hours=7),pd.DataFrame(source['matches']),pd.DataFrame(source['roster']),source['captured_at'])
        return
    with pytest.raises(ValueError):parse(source)


@pytest.mark.parametrize('case',['absent','homonym','old_roster','abbreviated'])
def test_no_unproven_player_identity(source,case):
    player=next(p for p in source['roster'] if p['nom']=='Tyler Bertuzzi')
    if case=='absent':source['roster'].remove(player)
    if case=='homonym':source['roster'].append(dict(player,id_joueur=999,id_equipe='BUF'))
    if case=='old_roster':player['observed_at']='2026-09-01T00:00:00Z'
    if case=='abbreviated':player['nom']='T. Bertuzzi'
    rows,rejected=parse(source);assert not rows and rejected[0]['reason']=='no_unique_recent_roster_identity'


def test_accent_and_proven_transfer(source):
    player=next(p for p in source['roster'] if p['nom']=='Tyler Bertuzzi')
    player['nom']='Tylér Bertuzzi'
    assert parse(source)[0][0]['team']=='CHI'
    # Synthetic transfer boundary: old observation stale, new roster is authoritative.
    source['roster'].append(dict(player,id_equipe='BUF',observed_at='2026-09-01T00:00:00Z'))
    assert parse(source)[0][0]['team']=='CHI'


@pytest.mark.parametrize('field,value',[('team',''),('team',float('nan')),('bookmaker',''),('date_match',None),('event_start_utc',None),('captured_at',None),('odds_decimal',float('nan')),('odds_decimal',1),('threshold',2),('stat','goals')])
def test_invalid_contract_remains_rejected(source,field,value):
    rows,_=parse(source);rows[0][field]=value
    assert validate_rows(rows,source['captured_at'])['status']=='invalid_contract'


def test_event_conflict_duplicate_and_stale_contract(source):
    rows,_=parse(source)
    conflict=copy.deepcopy(rows[0]);conflict['event_start_utc']='2026-10-04T23:00:00Z'
    report=validate_rows(rows+[conflict],source['captured_at'])
    assert all('ambiguous_event' in r['reasons'] for r in report['rejected'])
    assert validate_rows(rows*2,source['captured_at'])['valid_rows']==0
    rows[0]['captured_at']=(pd.Timestamp(source['captured_at'])-pd.Timedelta(hours=7)).isoformat()
    assert validate_rows(rows,source['captured_at'])['valid_rows']==0


def test_public_state_excludes_session_fields(source):
    html='<script id="serverApp-state" type="application/json">'+json.dumps({'private_session_marker':'discard','EventsDetail':{'events':[source['event']]}})+'</script>'
    assert public_events(html)==[source['event']]
    with pytest.raises(ValueError):public_events(html+html)
