import importlib.util
from pathlib import Path
import pytest
spec=importlib.util.spec_from_file_location('source_selector',Path(__file__).resolve().parents[1]/'scripts/select_but_v2_source.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_discovery_skips_context_only_and_untrusted_runs(monkeypatch):
    def run(n,**kw):
        return dict(id=n,head_branch='main',status='completed',conclusion='success',path=m.PATH,
                    event='workflow_dispatch',created_at=f'2026-10-22T{n:02d}:00Z',**kw)
    runs=[run(1),run(2),dict(run(3),head_branch='untrusted')]
    def api(path):
        if path.startswith('actions/workflows?'):return {'workflows':[dict(id=5,path=m.PATH,state='active')]}
        if '/runs?' in path:return {'workflow_runs':runs}
        if '/2/artifacts' in path:return {'artifacts':[dict(name='context-only',expired=False)]}
        assert '/1/artifacts' in path
        return {'artifacts':[dict(name='henachel-model-outputs-date',expired=False)]}
    monkeypatch.setattr(m,'api',api)
    assert m.select()=='1'


def test_explicit_untrusted_source_is_rejected(monkeypatch):
    monkeypatch.setattr(m,'api',lambda path:dict(head_branch='PR',status='completed',conclusion='success',path=m.PATH,event='pull_request'))
    with pytest.raises(ValueError,match='Untrusted'):m.select('1')
    with pytest.raises(ValueError,match='Numeric'):m.select('../bad')


def test_ambiguous_artifacts_never_selected(monkeypatch):
    run=dict(id=1,head_branch='main',status='completed',conclusion='success',path=m.PATH,event='schedule')
    artifact=dict(name='henachel-model-outputs-date',expired=False)
    monkeypatch.setattr(m,'api',lambda path: {'artifacts':[artifact,artifact]} if 'artifacts' in path else run)
    with pytest.raises(ValueError,match='Ambiguous'):m.select('1')
