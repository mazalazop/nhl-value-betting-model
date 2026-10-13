import copy
import sqlite3
import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parents[1]/'model'))
from henachel.prospective_pp import Archive, collect, overlay, parse_rotowire, snapshots

NOW = '2026-10-13T12:00:00Z'
START = '2026-10-14T00:00:00Z'
# Minimal structural fixture from the public RotoWire response; fictional names.
HTML = '''<div class="lineup is-nhl"><div class="lineup__teams">
<a class="lineup__team is-visit"><div class="lineup__abbr">TOR</div></a>
<a class="lineup__team is-home"><div class="lineup__abbr">MTL</div></a></div>
<ul class="lineup__list is-visit"><li class="lineup__title">POWER PLAY #1</li>
<li class="lineup__player"><a title="Émile Test">E. Test</a></li>
<li class="lineup__title is-middle">POWER PLAY #2</li>
<li class="lineup__player"><a title="Other Player">O. Player</a></li>
<li class="lineup__title">INJURIES</li><li class="lineup__player"><a title="Absent Player">A</a></li></ul></div>'''


def player():
    return dict(game_id=2026020010, player_id=8470001, game_date='2026-10-13', team='TOR', player_name='Emile Test', event_start_utc=START)


def evidence():
    rows = parse_rotowire(HTML)
    return dict(collected_at_utc=NOW, status='source_timestamp_unavailable', observations=rows)


def roster(): return [dict(player(), observed_at=NOW)]


def valid():
    e = evidence(); e['observations'][0]['source_timestamp'] = '2026-10-13T11:00:00Z'
    return snapshots([player()], roster(), e, now=NOW)


def test_explicit_units_only_and_no_invented_timestamp():
    rows = parse_rotowire(HTML)
    assert [r['observed_unit'] for r in rows] == ['PP1', 'PP2']
    assert all(r['source_timestamp'] is None for r in rows)
    assert snapshots([player()], roster(), evidence(), now=NOW)[0]['pp_unit'] == 'UNKNOWN'


def test_valid_timestamp_and_accent_identity(): assert valid()[0]['pp_unit'] == 'PP1'


@pytest.mark.parametrize('stamp', [None, '', '2026-10-13', '2026-10-14T00:00:00Z', '2026-10-01T00:00:00Z'])
def test_missing_naive_future_stale_timestamps_unknown(stamp):
    e = evidence(); e['observations'][0]['source_timestamp'] = stamp
    assert snapshots([player()], roster(), e, now=NOW)[0]['pp_unit'] == 'UNKNOWN'


def test_absence_never_means_pp2():
    p = player(); p['player_name'] = 'Nobody Present'
    assert snapshots([p], [], evidence(), now=NOW)[0]['pp_unit'] == 'UNKNOWN'


def test_ambiguous_player_and_transfer_rejected():
    e = evidence(); e['observations'][0]['source_timestamp'] = NOW
    for r in [roster()*2, [dict(roster()[0], team='MTL')], []]:
        assert snapshots([player()], r, e, now=NOW)[0]['pp_unit'] == 'UNKNOWN'
    e['observations'] *= 2
    assert snapshots([player()], roster(), e, now=NOW)[0]['pp_unit'] == 'UNKNOWN'


def test_puck_drop_guard():
    with pytest.raises(ValueError, match='Already started'):
        snapshots([player()], roster(), evidence(), now=START)


def test_unavailable_source_unknown():
    import requests
    class Session:
        def get(self, *a, **k): raise requests.HTTPError('403')
    e = collect(Session()); assert e['status'] == 'source_unavailable'
    e['collected_at_utc'] = NOW
    assert snapshots([player()], roster(), e, now=NOW)[0]['pp_unit'] == 'UNKNOWN'


def test_overlay_preserves_input_probability_and_optional_filter():
    p = dict(player(), target='POINT', probability=.37); original = copy.deepcopy(p)
    raw, filtered = overlay([p], valid(), True)
    assert p == original and raw[0]['probability'] == .37 and filtered == raw
    raw, filtered = overlay([p], [], True)
    assert raw[0]['pp_unit'] == 'UNKNOWN' and not filtered


def test_archive_atomic_retry_reopen_and_immutable(tmp_path):
    path = tmp_path/'archive.db'; a = Archive(path); rows = valid()
    assert a.append('batch1', 'pregame', rows, now=NOW)
    a.close(); a = Archive(path)
    assert not a.append('batch1', 'pregame', rows, now=START)
    with pytest.raises(ValueError, match='conflict'):
        a.append('batch1', 'pregame', [dict(rows[0], pp_unit='UNKNOWN')], now=NOW)
    with pytest.raises(sqlite3.IntegrityError): a.db.execute('DELETE FROM batches')
    a.db.rollback()
    with pytest.raises(ValueError):
        a.append('bad', 'pregame', rows+[dict(rows[0], player_id=2, event_start_utc=NOW)], now=NOW)
    assert a.db.execute('SELECT count(*) FROM batches').fetchone()[0] == 1
    a.close()


@pytest.mark.parametrize('change', [dict(event_start_utc=NOW), dict(actual_goals=0), dict(probability=float('nan')), dict(player_id=None)])
def test_archive_refuses_late_outcome_nan_identity(tmp_path, change):
    a = Archive(tmp_path/'a.db')
    with pytest.raises((ValueError, TypeError)):
        a.append('bad', 'pregame', [dict(valid()[0], **change)], now=NOW)
    assert a.db.execute('SELECT count(*) FROM batches').fetchone()[0] == 0
    a.close()


def test_final_outcomes_are_separate_append_only_revisions(tmp_path):
    a = Archive(tmp_path/'a.db')
    row = dict(player(), collected_at_utc='2026-10-14T05:00:00Z', game_status='FINAL', actual_points=1, actual_goals=0)
    assert a.append('outcome1', 'outcome', [row], now=row['collected_at_utc'])
    assert a.append('correction1', 'outcome', [dict(row, actual_goals=1)], now=row['collected_at_utc'])
    with pytest.raises(ValueError): a.append('unknown', 'outcome', [dict(row, actual_goals=None)], now=row['collected_at_utc'])
    a.close()


def test_bookmaker_evidence_preserves_probabilities_rejects_stale_and_wrong_market():
    from henachel.prospective_pp import attach_point_odds
    p=dict(player(), target='POINT', probability=.37)
    o=dict(id_match=p['game_id'],id_joueur=p['player_id'],market='player_points',stat='points',threshold=1,
           outcome_key='1_plus',team='TOR',start_time_utc=START,captured_at=NOW,event_id='123',bookmaker='unibet',odds_decimal=2.,implied_probability=.5)
    rows,bad=attach_point_odds([p],[o,dict(o,stat='goals'),dict(o,captured_at='2026-10-11T00:00:00Z'),o],now=NOW)
    assert len(bad)==3 and len(rows[0]['bookmaker_observations'])==1
    assert rows[0]['probability']==p['probability']==.37
    assert 'bookmaker_observations' not in p


def test_second_process_preserves_archive_and_exact_rerun(tmp_path):
    import json
    import subprocess
    a=Archive(tmp_path/'archive.sqlite3'); a.append('first','pregame',valid(),now=NOW); a.close()
    code='''import json,sys
sys.path.insert(0,sys.argv[1])
from henachel.prospective_pp import Archive
archive=Archive(sys.argv[2])
rows=json.loads(archive.db.execute("SELECT payload FROM batches WHERE batch_id='first'").fetchone()[0])
assert not archive.append('first','pregame',rows,now='2026-10-14T06:00:00Z')
assert archive.db.execute('SELECT count(*) FROM batches').fetchone()[0]==1
archive.close()
'''
    subprocess.run([sys.executable,'-c',code,str(Path(__file__).parents[1]/'model'),str(tmp_path/'archive.sqlite3')],check=True)
