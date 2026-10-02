from unittest.mock import Mock
import pandas as pd
import pytest
from henachel.nhl_cache import BoxscoreCache


def test_shared_cache_reduces_two_downloads_to_one(tmp_path):
    fetch=Mock(return_value={'id':1,'gameState':'OFF','playerByGameStats':{}})
    now=pd.Timestamp('2026-10-02T10:00:00Z')
    a=BoxscoreCache(tmp_path,now=now);b=BoxscoreCache(tmp_path,now=now)
    assert a.get(1,'2026-10-01',fetch)==b.get(1,'2026-10-01',fetch)
    assert fetch.call_count==1;assert b.hits==1
    c=BoxscoreCache(tmp_path,now=now+pd.Timedelta(hours=2));c.get(1,'2026-10-01',fetch)
    assert fetch.call_count==2
    BoxscoreCache(tmp_path,full_refresh=True,now=now).get(1,'2026-10-01',fetch)
    assert fetch.call_count==3

@pytest.mark.parametrize('payload',[{'id':1,'gameState':'LIVE'},{'id':2,'gameState':'OFF'}])
def test_unverified_payload_never_cached(tmp_path,payload):
    with pytest.raises(ValueError):BoxscoreCache(tmp_path).get(1,'2026-10-01',lambda _:payload)
    assert list(tmp_path.iterdir())==[]
