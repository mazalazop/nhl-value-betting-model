import io
import json
import zipfile
import pytest
from test_runtime import module

m = module('check_production_receipt')


@pytest.mark.parametrize('conclusion,market,day,expected', [
    ('success','GOAL','2026-11-17',123), ('failure','GOAL','2026-11-17',None),
    ('success','OTHER','2026-11-17',None), ('success','GOAL','2026-11-16',None),
])
def test_only_verified_matching_success_deduplicates(conclusion, market, day, expected):
    receipt=dict(status='verified',run_date=day,run_id='123',markets={'POINT':{},market:{}})
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:z.writestr('production_verified.json',json.dumps(receipt))
    def get(path,binary=False):
        if binary:return stream.getvalue()
        if path.startswith('actions/runs/'):
            return dict(id=123,head_branch='main',conclusion=conclusion,
                        path='.github/workflows/Workflow global Henachel — POINTS.yml')
        return {'artifacts':[dict(id=9,name='henachel-production-verified-2026-11-17',
                                 expired=False,workflow_run={'id':123})]}
    assert m.already_verified('2026-11-17','main',True,get)==expected


def test_api_failure_is_not_permission_to_recompute():
    def get(*args,**kwargs):raise RuntimeError('API unavailable')
    with pytest.raises(RuntimeError):m.already_verified('2026-11-17','main',True,get)
