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
