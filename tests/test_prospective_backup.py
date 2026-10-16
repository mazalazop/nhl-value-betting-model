import importlib.util,json,hashlib
from pathlib import Path
from henachel.prospective_pp import Archive
from test_prospective_pp import valid,NOW


def test_sqlite_backup_restore_preserves_batches(tmp_path):
    spec=importlib.util.spec_from_file_location('seal',Path(__file__).parents[1]/'scripts/seal_prospective_archive.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    path=tmp_path/'prospective.sqlite3'
    archive=Archive(path);archive.append('test','pregame',valid(),now=NOW);archive.close()
    before=path.read_bytes();report=module.seal(path)
    assert path.read_bytes()==before
    assert report['restore_verified'] and report['batches']==1
    assert report['sha256']==hashlib.sha256(before).hexdigest()
    assert json.loads(path.with_suffix('.manifest.json').read_text())==report


def test_next_day_restore_adds_new_slate_and_outcome_without_rewriting_pregame(tmp_path):
    import io,zipfile
    from unittest.mock import Mock
    from test_runtime import module
    from test_prospective_pp import player
    first=tmp_path/'first/prospective.sqlite3'
    a=Archive(first)
    pre=dict(player(),target='GOAL',probability=.25,frozen_probability=.20,
             player_smoothed_probability=.21,collected_at_utc=NOW)
    assert a.append('day1','pregame',[pre],now=NOW)
    original=a.db.execute("SELECT payload FROM batches WHERE batch_id='day1'").fetchone()[0]
    a.close()
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        z.writestr('master_daily_bets_history.csv','bet_id,bet_status,result\nknown,settled,win\n')
        z.writestr('prospective.sqlite3',first.read_bytes())
        z.writestr('prospective_enabled.json','{}')
        z.writestr('prospective.manifest.json',json.dumps({'sha256':hashlib.sha256(first.read_bytes()).hexdigest()}))
    index=Mock();index.json.return_value={'artifacts':[dict(expired=False,created_at=NOW,archive_download_url='https://example.test/backup')]}
    body=Mock();body.content=stream.getvalue();session=Mock();session.get.side_effect=[index,body]
    dest=tmp_path/'second'
    assert module('restore_history').restore(session,'org/repo','branch',dest)=='restored'
    a=Archive(dest/'prospective.sqlite3')
    next_day='2026-10-14T12:00:00Z'
    second=dict(pre,game_id=2026020011,game_date='2026-10-14',event_start_utc='2026-10-15T00:00:00Z',collected_at_utc=next_day)
    assert a.append('day2','pregame',[second],now=next_day)
    result=dict(player(),game_status='FINAL',actual_goals=1,actual_points=2,collected_at_utc=next_day)
    assert a.append('result1','outcome',[result],now=next_day)
    assert not a.append('result1','outcome',[result],now=next_day)
    assert a.db.execute("SELECT payload FROM batches WHERE batch_id='day1'").fetchone()[0]==original
    assert a.db.execute('SELECT count(*) FROM batches').fetchone()[0]==3
    a.close()
    assert 'known,settled,win' in (dest/'master_daily_bets_history.csv').read_text()
