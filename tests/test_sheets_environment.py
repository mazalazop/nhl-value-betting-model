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
    ws=Mock();ws.title='history_raw';ws.row_values.return_value=module.EXPECTED['history_raw']
    sheet.worksheets.return_value=[ws]
    report=module.inspect(client,'synthetic-id')
    assert report['status']=='read_access_ok'
    assert report['worksheets']['history_raw']['missing_canonical_columns']==[]
    assert report['schema_status']=='missing_or_incompatible'  # daily tab absent
    assert [c[0] for c in sheet.method_calls]==['worksheets']
    ws.row_values.assert_called_once_with(1)


def test_probe_columns_match_actual_published_views():
    from conftest import load_script
    from test_history import candidate
    path=Path(__file__).resolve().parents[1]/'scripts/validate_sheets_readonly.py'
    spec=importlib.util.spec_from_file_location('readonly_schema',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    publisher=load_script('08_publish_to_google_sheet')
    assert set(module.EXPECTED['daily_picks'])==set(publisher.build_daily_display_df(candidate()).columns)
    assert set(module.EXPECTED['history_raw'])==set(publisher.build_history_display_df(candidate()).columns)


def test_operational_workflow_never_writes_production_or_credentials():
    import yaml
    root=Path(__file__).resolve().parents[1]
    workflow=yaml.safe_load((root/'.github/workflows/Workflow global Henachel — POINTS.yml').read_text())
    jobs=workflow['jobs']
    assert "github.ref == 'refs/heads/astra/audit-remediation'" in jobs['sheets_readonly']['if']
    steps=jobs['model_points']['steps']
    publication=next(s for s in steps if s.get('name')=='Publish to Google Sheet')
    assert "github.ref != 'refs/heads/astra/audit-remediation'" in publication['if']
    assert '--credentials-env' in publication['run'] and 'mktemp' not in publication['run']
    assert any('collect_unibet_structured.py' in s.get('run','') for s in steps)


def test_schema_validation_never_runs_migration_and_manual_apply_is_gated():
    import yaml
    root=Path(__file__).resolve().parents[1]
    workflow=yaml.safe_load((root/'.github/workflows/Workflow global Henachel — POINTS.yml').read_text())
    jobs=workflow['jobs']
    assert not any('migrate_history_sheet.py' in s.get('run','') for s in jobs['sheets_readonly']['steps'])
    migration=jobs['history_schema_migration']
    assert "github.event_name == 'workflow_dispatch'" in migration['if']
    assert 'inputs.migrate_history_schema' in migration['if']
    assert jobs['sheets_readonly']['needs']==['history_schema_migration']
    triggers=workflow.get('on',workflow.get(True))
    assert triggers['workflow_dispatch']['inputs']['migrate_history_schema']['default'] is False
    apply=next(s for s in migration['steps'] if '--apply' in s.get('run',''))
    assert apply['env']['MIGRATION_EXPECTED_SNAPSHOT']=='${{ inputs.migration_expected_snapshot }}'
    assert '--expected-snapshot "$MIGRATION_EXPECTED_SNAPSHOT"' in apply['run']
    assert '${{ inputs.' not in apply['run']


@pytest.mark.parametrize('legacy',[False,True])
def test_readonly_schema_accepts_reruns_and_refuses_missing_columns(legacy):
    from test_sheets_migration import LEGACY,CANONICAL
    path=Path(__file__).resolve().parents[1]/'scripts/validate_sheets_readonly.py'
    spec=importlib.util.spec_from_file_location('schema_rerun',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    client=Mock();sheet=client.open_by_key.return_value;sheet.title='Henachel'
    daily=Mock();daily.title='daily_picks';daily.row_values.return_value=module.EXPECTED['daily_picks']
    history=Mock();history.title='history_raw';history.row_values.return_value=LEGACY if legacy else CANONICAL+['analyst_note']
    sheet.worksheets.return_value=[daily,history]
    first=module.inspect(client,'synthetic-id')
    # Ordinary external changes to historical rows must not trigger a migration.
    history.get_all_values.side_effect=AssertionError('Schema validation must not depend on data rows')
    second=module.inspect(client,'synthetic-id')
    assert first==second
    assert first['schema_status']==('missing_or_incompatible' if legacy else 'ok')
    assert set(c[0] for c in history.method_calls)=={'row_values'}
    assert set(c[0] for c in sheet.method_calls)=={'worksheets'}
