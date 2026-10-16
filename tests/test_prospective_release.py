import gzip
import importlib.util
from pathlib import Path
import sys
from unittest.mock import Mock
import pytest
from henachel.prospective_pp import Archive
from test_prospective_pp import NOW, valid

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from backup_prospective_release import backup


def response(data=None, content=b'', status=200):
    r=Mock();r.status_code=status;r.json.return_value=data;r.content=content
    return r


def test_draft_backup_remote_roundtrip_then_idempotent_rerun(tmp_path):
    path=tmp_path/'archive.sqlite3'
    archive=Archive(path);archive.append('first','pregame',valid(),now=NOW);archive.close()
    compressed=gzip.compress(path.read_bytes(),mtime=0)
    release={'id':1,'tag_name':'henachel-prospective-backups','draft':True,'upload_url':'https://uploads.github.com/repos/o/r/releases/1/assets{?name}'}
    client=Mock();client.get.side_effect=[response([]),response([]),response(content=compressed)]
    client.post.side_effect=[response(release),response({'id':2})]
    first=backup(client,'o/r',path,'sha')
    assert first['remote_restore_verified'] and first['status']=='created'
    assert client.post.call_args_list[0].kwargs['json']['draft'] is True
    second_client=Mock();second_client.get.side_effect=[response([release]),response([{'id':2,'name':first['name']}]),response(content=compressed)]
    second=backup(second_client,'o/r',path,'sha')
    assert second['status']=='existing'
    second_client.post.assert_not_called()


def test_public_release_or_corrupt_remote_backup_refused(tmp_path):
    path=tmp_path/'archive.sqlite3';archive=Archive(path);archive.close()
    client=Mock();client.get.return_value=response([{'tag_name':'henachel-prospective-backups','draft':False}])
    with pytest.raises(ValueError,match='unpublished'):backup(client,'o/r',path,'sha')
    client.post.assert_not_called()


def test_corrupt_existing_asset_never_overwritten(tmp_path):
    import hashlib
    path=tmp_path/'archive.sqlite3';archive=Archive(path);archive.close()
    name='prospective-'+hashlib.sha256(path.read_bytes()).hexdigest()+'.sqlite3.gz'
    release={'id':1,'tag_name':'henachel-prospective-backups','draft':True}
    client=Mock();client.get.side_effect=[response([release]),response([{'id':2,'name':name}]),response(content=gzip.compress(b'corrupt'))]
    with pytest.raises(ValueError,match='restore mismatch'):backup(client,'o/r',path,'sha')
    client.post.assert_not_called()
