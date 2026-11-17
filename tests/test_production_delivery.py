import datetime as dt
import json
import os
import csv
import pytest
from test_runtime import module

m = module('validate_production')
NOW = dt.datetime(2026, 11, 17, 12, tzinfo=dt.timezone.utc)


def fixture(root):
    common = dict(id_match='1', id_joueur='2', date_match='2026-11-17',
                  start_time_utc='2026-11-17T23:00:00Z')
    files = {
        'predictions_upcoming_point_enrichi_calibre_v2.csv': [dict(common, proba_point_1p_calibree=.6)],
        'predictions_upcoming_goal_research.csv': [dict(common, probability_goal=.2,
            model_status='research_only_operational', generated_at_utc='2026-11-17T11:45:00Z')],
        '07_daily_bets.csv': [dict(common, run_date='2026-11-17', bookmaker='Unibet',
            odds_decimal=1.35, captured_at='2026-11-17T11:50:00Z')],
    }
    for name, rows in files.items():
        with (root/name).open('w') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (root/'05_predict_upcoming_games_summary.json').write_text(json.dumps(
        dict(status='ok', target_date='2026-11-17', future_players_rows=1)))
    for p in root.iterdir(): os.utime(p, (NOW.timestamp(), NOW.timestamp()))


def check(root):
    return m.validate(root, '2026-11-17', '2026-11-17T11:30:00Z', now=NOW)


def test_verified_manifest_contains_hashes_and_research_status(tmp_path):
    fixture(tmp_path)
    result = check(tmp_path)
    assert result['status'] == 'verified'
    assert result['goal_production_approved'] is False
    assert result['markets']['GOAL']['rows'] == 1
    assert len(result['files']) == 4


@pytest.mark.parametrize('replacement', ['nan', '1.01', '-0.01'])
def test_invalid_probability_rejected(tmp_path, replacement):
    fixture(tmp_path)
    p = tmp_path/'predictions_upcoming_goal_research.csv'
    p.write_text(p.read_text().replace('0.2', replacement))
    with pytest.raises(ValueError, match='Invalid probability'): check(tmp_path)


def test_missing_goal_and_stale_file_rejected(tmp_path):
    fixture(tmp_path)
    p = tmp_path/'predictions_upcoming_goal_research.csv'
    os.utime(p, (0, 0))
    with pytest.raises(ValueError, match='Stale'): check(tmp_path)
    p.unlink()
    with pytest.raises(FileNotFoundError): check(tmp_path)


@pytest.mark.parametrize('old,new', [
    ('2026-11-17,', '2026-11-16,'),
    ('2026-11-17T23:00:00Z', '2026-11-17T10:00:00Z'),
    ('research_only_operational', 'approved'),
])
def test_wrong_date_started_game_and_goal_activation_rejected(tmp_path, old, new):
    fixture(tmp_path)
    p = tmp_path/'predictions_upcoming_goal_research.csv'
    p.write_text(p.read_text().replace(old, new))
    with pytest.raises(ValueError): check(tmp_path)


def test_single_paris_schedule_and_terminal_guard():
    import yaml
    from pathlib import Path
    base = Path(__file__).parents[1]/'.github/workflows'
    w = yaml.safe_load((base/'Workflow global Henachel — POINTS.yml').read_text())
    assert {'cron': '30 12 * * *', 'timezone': 'Europe/Paris'} in w[True]['schedule']
    assert 'push' not in w[True]
    dispatcher = yaml.safe_load((base/'henachel-dispatch.yml').read_text())
    assert 'schedule' not in dispatcher[True] and 'push' not in dispatcher[True]
    assert 'always()' in w['jobs']['production_status']['if']
    steps = w['jobs']['model_points']['steps']
    gate = next(i for i, s in enumerate(steps) if 'validate_production.py' in s.get('run', ''))
    publish = next(i for i, s in enumerate(steps) if s.get('name') == 'Publish to Google Sheet')
    assert gate < publish


@pytest.mark.parametrize('day,offset', [('2026-03-28', 1), ('2026-03-29', 2),
                                      ('2026-10-24', 2), ('2026-10-25', 1)])
def test_paris_noon_survives_dst(day, offset):
    from zoneinfo import ZoneInfo
    local = dt.datetime.fromisoformat(day+'T12:30:00').replace(tzinfo=ZoneInfo('Europe/Paris'))
    assert local.utcoffset() == dt.timedelta(hours=offset)
