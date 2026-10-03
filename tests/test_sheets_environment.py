import importlib.util
from pathlib import Path
from unittest.mock import Mock
import pytest
from henachel import sheets_auth


def test_authentication_memory_only(monkeypatch):
    # Synthetic configuration, never a real credential.
    monkeypatch.setenv('GOOGLE_CREDENTIALS','{"test_marker":"synthetic"}')
    constructor=Mock();authorize=Mock()
    monkeypatch.setattr(sheets_auth.Credentials,'from_service_account_info',constructor)
    monkeypatch.setattr(sheets_auth.gspread,'authorize',authorize)
    sheets_auth.authorize_environment(readonly=True)
    constructor.assert_called_once_with({'test_marker':'synthetic'},scopes=['https://www.googleapis.com/auth/spreadsheets.readonly'])
    assert 'GOOGLE_CREDENTIALS' not in sheets_auth.os.environ


def test_invalid_secret_not_echoed(monkeypatch):
    monkeypatch.setenv('GOOGLE_CREDENTIALS','SYNTHETIC_DO_NOT_ECHO')
    with pytest.raises(ValueError) as err:sheets_auth.authorize_environment()
    assert 'SYNTHETIC_DO_NOT_ECHO' not in str(err.value)


def test_sheet_probe_is_readonly():
    path=Path(__file__).resolve().parents[1]/'scripts/validate_sheets_readonly.py'
    spec=importlib.util.spec_from_file_location('readonly_probe',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    client=Mock();sheet=client.open_by_key.return_value;sheet.title='Henachel'
    ws=Mock();ws.title='history_raw';ws.row_values.return_value=['bet_id','id_match','id_joueur','result','bet_status']
    sheet.worksheets.return_value=[ws]
    report=module.inspect(client,'synthetic-id')
    assert report['status']=='read_access_ok'
    assert report['worksheets']['history_raw']['missing_canonical_columns']==[]
    assert [c[0] for c in sheet.method_calls]==['worksheets']
    ws.row_values.assert_called_once_with(1)


def test_operational_workflow_never_writes_production_or_credentials():
    import yaml
    root=Path(__file__).resolve().parents[1]
    workflow=yaml.safe_load((root/'.github/workflows/Workflow global Henachel — POINTS.yml').read_text())
    jobs=workflow['jobs']
    assert jobs['sheets_readonly']['if']=="github.ref == 'refs/heads/astra/audit-remediation'"
    steps=jobs['model_points']['steps']
    publication=next(s for s in steps if s.get('name')=='Publish to Google Sheet')
    assert "github.ref != 'refs/heads/astra/audit-remediation'" in publication['if']
    assert '--credentials-env' in publication['run'] and 'mktemp' not in publication['run']
    assert any('collect_unibet_structured.py' in s.get('run','') for s in steps)
