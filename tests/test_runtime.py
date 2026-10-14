import importlib.util
from pathlib import Path
from unittest.mock import Mock
import pytest

def module(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

@pytest.mark.parametrize('key,value',[('RUN_DATE','2026-10-02\nEVIL=x'),('HUB_URL','https://evil.test'),('HEADLESS','$(touch x)'),('DISCOVERY_MAX_MATCHES','12;echo x')])
def test_runtime_inputs_not_shell_code(key,value):
    with pytest.raises(ValueError):module('resolve_runtime').resolve({'INPUT_'+key:value})

def test_history_missing_requires_explicit_bootstrap(tmp_path):
    session=Mock();session.get.return_value.json.return_value={'artifacts':[]};m=module('restore_history')
    with pytest.raises(ValueError):m.restore(session,'org/repo','refs/heads/test',tmp_path)
    assert m.restore(session,'org/repo','refs/heads/test',tmp_path,True)=='bootstrap'
    assert m.artifact_name('refs/heads/test')!=m.artifact_name('refs/heads/main')

def test_expired_history_never_bootstraps(tmp_path):
    session=Mock();session.get.return_value.json.return_value={'artifacts':[{'expired':True,'created_at':'2026-01-01'}]}
    with pytest.raises(ValueError):module('restore_history').restore(session,'org/repo','main',tmp_path,True)

def test_workflow_contract_and_shell_syntax():
    import subprocess
    import yaml
    path=Path(__file__).resolve().parents[1]/'.github/workflows/Workflow global Henachel — POINTS.yml'
    workflow=yaml.safe_load(path.read_text())
    assert workflow['concurrency']['cancel-in-progress'] is False
    steps=workflow['jobs']['model_points']['steps']
    joined='\n'.join(step.get('run','') for step in steps)
    for script in ['00c_refresh_team_standings','09_settle_previous_bets','restore_history']:
        assert script in joined
    for job in workflow['jobs'].values():
        for step in job['steps']:
            if 'run' not in step:continue
            assert '${{' not in step['run']
            proc=subprocess.run(['bash','-n'],input=step['run'],text=True,capture_output=True)
            assert proc.returncode==0,proc.stderr


def test_prospective_archive_restored_and_loss_refused(tmp_path):
    import io
    import zipfile
    def session_for(files):
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w') as z:
            for name,content in files.items(): z.writestr(name,content)
        index=Mock();index.json.return_value={'artifacts':[dict(expired=False,created_at='2026-10-13',archive_download_url='https://example.test/archive')]}
        body=Mock();body.content=data.getvalue()
        session=Mock();session.get.side_effect=[index,body];return session
    files={'master_daily_bets_history.csv':'bet_id\n', 'prospective_enabled.json':'{}', 'prospective.sqlite3':'opaque bytes'}
    m=module('restore_history')
    assert m.restore(session_for(files),'org/repo','branch',tmp_path/'ok')=='restored'
    assert (tmp_path/'ok/prospective.sqlite3').read_text()=='opaque bytes'
    del files['prospective.sqlite3']
    with pytest.raises(ValueError,match='Prospective archive missing'):
        m.restore(session_for(files),'org/repo','branch',tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_goal_operational_workflow_is_opt_in_and_archived():
    import yaml
    workflow=yaml.safe_load((Path(__file__).parents[1]/'.github/workflows/Workflow global Henachel — POINTS.yml').read_text())
    steps=workflow['jobs']['model_points']['steps']
    goal=next(s for s in steps if 'predict_goal_daily.py' in s.get('run',''))
    assert 'capture_prospective_pp' in goal['if']
    assert '--target-date "$RUN_DATE"' in goal['run']
    assert any('seal_prospective_archive.py' in s.get('run','') for s in steps)
    assert any('prospective.manifest.json' in s.get('with',{}).get('path','') for s in steps)


def test_archive_manifest_checksum_checked_before_restore(tmp_path):
    import io,zipfile,json,hashlib
    def session_for(digest):
        data=io.BytesIO()
        with zipfile.ZipFile(data,'w') as z:
            z.writestr('master_daily_bets_history.csv','bet_id\n')
            z.writestr('prospective.sqlite3',b'archive')
            z.writestr('prospective.manifest.json',json.dumps({'sha256':digest}))
        index=Mock();index.json.return_value={'artifacts':[dict(expired=False,created_at='2026-10-14',archive_download_url='https://example.test/archive')]}
        body=Mock();body.content=data.getvalue()
        session=Mock();session.get.side_effect=[index,body];return session
    m=module('restore_history')
    with pytest.raises(ValueError,match='checksum'):
        m.restore(session_for('wrong'),'org/repo','branch',tmp_path/'bad')
    assert not (tmp_path/'bad').exists()
    assert m.restore(session_for(hashlib.sha256(b'archive').hexdigest()),'org/repo','branch',tmp_path/'ok')=='restored'
