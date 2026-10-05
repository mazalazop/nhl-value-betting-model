"""Offline contracts for the bounded historical-context probe; no network."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('research_context_probe', Path(__file__).parents[1] / 'scripts/probe_research_context.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_roster_contract_ids_positions_and_duplicates():
    row = {'id': 1, 'positionCode': 'C', 'firstName': {'default': 'A'}, 'lastName': {'default': 'B'}}
    payload = {'forwards': [row], 'defensemen': [], 'goalies': []}
    assert probe.roster_contract(payload)['contract_ok']
    assert not probe.roster_contract(dict(payload, forwards=[row, row]))['contract_ok']
    assert not probe.roster_contract(dict(payload, forwards=[dict(row, positionCode='')]))['contract_ok']
    assert not probe.roster_contract(dict(payload, forwards=[dict(row, id=None)]))['contract_ok']


def test_standings_contract_exact_date_season_and_32_teams():
    payload = {'standings': [{'date': '2023-02-01', 'seasonId': 20222023, 'teamAbbrev': {'default': t}} for t in probe.TEAMS]}
    assert probe.standings_contract(payload, 20222023, '2023-02-01')['contract_ok']
    assert not probe.standings_contract(payload, 20222023, '2023-02-02')['contract_ok']
    assert not probe.standings_contract(payload, 20232024, '2023-02-01')['contract_ok']
    assert not probe.standings_contract({'standings': payload['standings'][:-1]}, 20222023, '2023-02-01')['contract_ok']
