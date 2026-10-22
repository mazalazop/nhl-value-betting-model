import importlib.util
import json
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('but_backup', Path(__file__).resolve().parents[1] / 'scripts/backup_but_v2_state.py')
b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)


class MemoryStore:
    prefix = 'test-'
    def __init__(self):
        self.blobs = {}; self.names = {}; self.index = None
    def latest(self, allow_incomplete=False):
        return self.index
    def create(self, tag, commit):
        return dict(id=tag, tag_name=tag)
    def upload(self, release, name, data):
        key = (release['id'], name)
        if key in self.names:
            assert self.blobs[self.names[key]] == data
            return self.names[key]
        asset_id = len(self.blobs) + 1
        self.blobs[asset_id] = data; self.names[key] = asset_id
        if name == 'index.json':
            self.index = json.loads(data)
        return asset_id
    def download(self, asset_id):
        return self.blobs[asset_id]


def seal(root):
    actual = b.inventory(root)
    actual.pop('archive_manifest.json', None)
    (root / 'archive_manifest.json').write_text(json.dumps(dict(files=actual)))


def test_incremental_backup_remote_restore_and_old_revision(tmp_path, monkeypatch):
    monkeypatch.setattr(b, 'BUNDLE_LIMIT', 512)
    root = tmp_path / 'state'; root.mkdir()
    (root / 'fixed').write_bytes(b'a' * 400)
    (root / 'changing').write_text('first')
    seal(root); store = MemoryStore()
    first = b.backup(store, root, 'commit', '1'); old = store.index
    assert first['remote_restore_verified']
    (root / 'changing').write_text('second'); seal(root)
    second = b.backup(store, root, 'commit', '2')
    assert second['remote_restore_verified']
    assert old['files']['fixed']['asset_id'] == store.index['files']['fixed']['asset_id']
    b.restore(store, old, tmp_path / 'old')
    b.restore(store, store.index, tmp_path / 'new')
    assert (tmp_path / 'old/changing').read_text() == 'first'
    assert (tmp_path / 'new/changing').read_text() == 'second'
    # A retry cannot overwrite an immutable asset.
    b.backup(store, root, 'commit', '2')
    asset_id = store.index['files']['fixed']['asset_id']
    store.blobs[asset_id] = b'corrupted'
    with pytest.raises(ValueError, match='checksum'):
        b.restore(store, store.index, tmp_path / 'corrupt')


def test_no_unsealed_state_path_traversal_or_oversize(tmp_path, monkeypatch):
    root = tmp_path / 'state'; root.mkdir(); (root / 'file').write_text('source'); seal(root)
    (root / 'file').write_text('tampered')
    with pytest.raises(ValueError, match='integrity'):
        b.backup(MemoryStore(), root, 'commit', '1')
    seal(root); monkeypatch.setattr(b, 'MAX_FILE_SIZE', 1)
    with pytest.raises(ValueError, match='budget'):
        b.backup(MemoryStore(), root, 'commit', '1')
    for name in ['../escape', '/absolute', 'a/../../escape']:
        with pytest.raises(ValueError, match='Unsafe'):
            b.safe_path(name)


def test_compaction_keeps_latest_sources_and_remote_old_views(tmp_path):
    root = tmp_path / 'state'; root.mkdir()
    for name in ['batches/1/cumulative/report.json', 'batches/1/reports/day/settlement.csv',
                 'batches/1/capture_status.json', 'batches/1/inputs/v1.csv',
                 'batches/2/cumulative/report.json', 'captures/day/predictions.csv',
                 'training_sources/old.csv.gz', 'training_sources/current.csv.gz']:
        p = root / name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(name)
    (root / 'latest_batch.json').write_text(json.dumps(dict(path='batches/2')))
    ref=root / 'batches/2/inputs/history_reference.json'; ref.parent.mkdir(exist_ok=True)
    ref.write_text(json.dumps(dict(path='training_sources/current.csv.gz')))
    seal(root); store = MemoryStore(); b.backup(store, root, 'commit', '2'); index = store.index
    removed = b.compact_verified_state(root, index)
    assert len(removed) == 3
    assert not (root / 'training_sources/old.csv.gz').exists()
    assert (root / 'training_sources/current.csv.gz').exists()
    assert (root / 'captures/day/predictions.csv').exists()
    assert (root / 'batches/1/inputs/v1.csv').exists()
    assert (root / 'batches/2/cumulative/report.json').exists()
    b.validate_state(root)
    b.restore(store, index, tmp_path / 'historical')
    assert (tmp_path / 'historical/batches/1/cumulative/report.json').exists()
    assert (tmp_path / 'historical/training_sources/old.csv.gz').exists()
    with pytest.raises(ValueError, match='exact verified'):
        b.compact_verified_state(root, index)


def test_direct_release_discovery_checks_identity_and_receipt_hash():
    class Response:
        def json(self): return dict(id=12,draft=True,tag_name=b.PREFIX+'snapshot')
    class Direct(b.Store):
        def get(self,path,**kwargs): return Response()
        def assets(self,release): return {'index.json':{'id':1},'verified.json':{'id':2}}
        def download(self,asset_id):
            raw=b'{"workflow_run_id":"1","files":{}}'
            return raw if asset_id==1 else json.dumps(dict(index_sha256=b.digest(raw))).encode()
    store=Direct(None,'repo')
    assert store.by_release_id(12)['workflow_run_id']=='1'
    with pytest.raises(ValueError,match='checksum'):
        store.by_release_id(12,'wrong-receipt-hash')
