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
