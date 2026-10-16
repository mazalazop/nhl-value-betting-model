import sys
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).parents[1]/'model'))
from henachel.prospective_http import get_json

class Response:
    def __init__(self, status, retry='2'):
        self.status_code=status; self.headers={'Retry-After':retry}
    def raise_for_status(self):
        if self.status_code >= 400: raise RuntimeError('http failure')
    def json(self): return {'data':'public NHL'}

class Session:
    def __init__(self, responses): self.responses=iter(responses); self.calls=0
    def get(self, *a, **k): self.calls+=1; return next(self.responses)


def test_retry_cache_and_original_timestamp_preserved(tmp_path):
    s=Session([Response(429),Response(200)]); sleeps=[]
    a=get_json(s,'https://example.test/nhl',tmp_path,60,sleeps.append)
    b=get_json(s,'https://example.test/nhl',tmp_path,60,sleeps.append)
    assert a==b and s.calls==2 and sleeps==[1.5,2,1.5]
    assert not list(tmp_path.glob('*.tmp'))


def test_long_retry_after_not_ignored(tmp_path):
    with pytest.raises(RuntimeError,match='Retry-After'):
        get_json(Session([Response(429,'120')]),'https://example.test/nhl',tmp_path,60,lambda x:None)
    assert not list(tmp_path.iterdir())


def test_failed_fetch_never_cached(tmp_path):
    with pytest.raises(RuntimeError):
        get_json(Session([Response(403)]),'https://example.test/nhl',tmp_path,60,lambda x:None)
    assert not list(tmp_path.iterdir())
