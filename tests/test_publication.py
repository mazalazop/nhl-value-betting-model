from types import SimpleNamespace
from unittest.mock import Mock
import pandas as pd
import pytest
from conftest import load_script
from test_history import candidate

publisher=load_script('08_publish_to_google_sheet')

def test_republication_cannot_overwrite_results():
    old=candidate();old['result']='win';old['bet_status']='settled';old['actual_stat_value']=2;old['settled_at']='2026-10-03'
    merged=publisher.merge_history(publisher.build_history_display_df(old),publisher.build_history_display_df(candidate()))
    assert merged.result.tolist()==['win'];assert merged.bet_status.tolist()==['settled']

def test_write_single_atomic_batch_no_clear_and_literal_values():
    sh=Mock();ws=SimpleNamespace(id=12,row_count=2000,col_count=50,spreadsheet=sh)
    publisher.write_replace(ws,pd.DataFrame({'player':['=IMPORTXML("x")']}))
    sh.batch_update.assert_called_once()
    body=sh.batch_update.call_args.args[0]
    assert len(body['requests'])==1
    cells=body['requests'][0]['updateCells']
    assert cells['range']['endRowIndex']==2000
    assert cells['rows'][1]['values'][0]['userEnteredValue']=={'stringValue':'=IMPORTXML("x")'}

def test_write_failure_has_no_clear_side_effect():
    sh=Mock();sh.batch_update.side_effect=RuntimeError('offline')
    ws=SimpleNamespace(id=12,row_count=2000,col_count=50,spreadsheet=sh)
    with pytest.raises(RuntimeError):publisher.write_replace(ws,candidate())
    assert sh.method_calls[0][0]=='batch_update'

def test_scope_minimal_and_history_ids_retained():
    assert publisher.SCOPES==['https://www.googleapis.com/auth/spreadsheets']
    frame=publisher.build_history_display_df(candidate())
    assert {'id_match','id_joueur','outcome_key'}.issubset(frame.columns)
