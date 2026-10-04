from copy import deepcopy
import pytest
from conftest import load_script
from henachel.sheets_migration import migrate,snapshot,plan_migration,fingerprint

CANONICAL=load_script('08_publish_to_google_sheet').HISTORY_OUTPUT_COLUMNS
# Exact production header sequence observed by read-only run 37130070589.
LEGACY=['bet_id','run_date','date_match','player_name','team','opponent','bookmaker','market',
        'stat','threshold','odds_decimal','implied_probability_pct','model_probability_pct',
        'edge_probability_pct','bet_status','result','actual_stat_value','settled_at']


def cell(value):
    if value is None:return {}
    return {'numberValue':value} if isinstance(value,(float,int)) else {'stringValue':value}


class Sheet:
    def __init__(self,columns=LEGACY,rows=None):
        self.rows=[[cell(c) for c in columns]] if columns else []
        if rows is None:
            rows=[]
            for n,result in enumerate(['win','loss','void']):
                row={c:'' for c in columns}
                row.update(bet_id=f'legacy-00{n}',run_date='2026-10-02',date_match='2026-10-02',player_name='Émile Test',team='TOR',opponent='MTL',bookmaker='Unibet',market='player_points',stat='points',threshold=1,odds_decimal=2.15,model_probability_pct='55.00',bet_status='settled',result=result,actual_stat_value=n,settled_at='2026-10-03T02:10:00Z')
                rows.append([row.get(c,'legacy extra') for c in columns])
        self.rows.extend([[cell(v) for v in row] for row in rows]);self.columns=max(len(columns),26);self.calls=[]

    def fetch_sheet_metadata(self,params):
        assert params['ranges']=="'history_raw'"
        return {'sheets':[{'properties':{'sheetId':9,'title':'history_raw','gridProperties':{'columnCount':self.columns}},
            'data':[{'rowData':[{'values':[{'userEnteredValue':c} for c in row]} for row in deepcopy(self.rows)]}]}]}

    def batch_update(self,body):
        self.calls.append(deepcopy(body))
        for request in body['requests']:
            if 'insertDimension' in request:
                pos=request['insertDimension']['range']['startIndex'];self.columns+=1
                for row in self.rows:row.insert(pos,{})
            elif 'appendDimension' in request:
                self.columns+=1
                for row in self.rows:row.append({})
            elif 'moveDimension' in request:
                move=request['moveDimension'];src=move['source']['startIndex'];dest=move['destinationIndex']
                for row in self.rows:row.insert(dest,row.pop(src))
            else:
                update=request['updateCells'];assert update['range']['sheetId']==9
                assert update['range']['startRowIndex']==0 and update['range']['endRowIndex']==1
                pos=update['range']['startColumnIndex']
                if not self.rows:self.rows=[[]]
                while len(self.rows[0])<=pos:self.rows[0].append({})
                self.rows[0][pos]=deepcopy(update['rows'][0]['values'][0]['userEnteredValue'])


def test_exact_production_legacy_schema_lossless_atomic_and_idempotent():
    sheet=Sheet();before=deepcopy(sheet.rows)
    dry=migrate(sheet,CANONICAL)
    assert not sheet.calls and sheet.rows==before
    assert dry['columns_added']==['id_match','id_joueur','outcome_key']
    assert dry['rows_before']==dry['rows_after']==3
    assert dry['original_values_sha256']==dry['projected_values_sha256']
    done=migrate(sheet,CANONICAL,apply=True,expected_before=dry['snapshot_sha256'])
    assert done['status']=='verified' and done['idempotent']
    assert len(sheet.calls)==1
    assert sheet.rows[0]==[cell(c) for c in CANONICAL]
    assert sheet.rows[1:]==[[{}, {}, {}]+r for r in before[1:]]
    assert done['result_preserved'] and done['bet_status_preserved'] and done['settled_at_preserved']
    assert done['duplicate_bet_ids_after']==0
    migrate(sheet,CANONICAL,apply=True,expected_before=dry['snapshot_sha256'])
    assert len(sheet.calls)==1


def test_current_schema_and_known_ids_are_never_overwritten():
    sheet=Sheet(CANONICAL)
    sheet.rows[1][0]=cell(2026020010);sheet.rows[1][1]=cell(8471234);sheet.rows[1][2]=cell('1_plus')
    before=deepcopy(sheet.rows)
    assert migrate(sheet,CANONICAL,apply=True)['request_count']==0
    assert sheet.rows==before and not sheet.calls


def test_partial_migration_preserves_known_id():
    sheet=Sheet(['id_match']+LEGACY);sheet.rows[1][0]=cell(2026020010)
    migrate(sheet,CANONICAL,apply=True)
    assert sheet.rows[1][:3]==[cell(2026020010),{},{}]


def test_empty_sheet_initializes_headers_only():
    sheet=Sheet([],[])
    report=migrate(sheet,CANONICAL,apply=True)
    assert report['rows_before']==report['rows_after']==0
    assert sheet.rows==[[cell(c) for c in CANONICAL]]
    migrate(sheet,CANONICAL,apply=True);assert len(sheet.calls)==1


def test_extra_legacy_columns_and_reordered_headers_preserved():
    old=['analyst_note']+list(reversed(LEGACY))+['custom_flag']
    sheet=Sheet(old);before=deepcopy(sheet.rows)
    migrate(sheet,CANONICAL,apply=True)
    names=[v['stringValue'] for v in sheet.rows[0]]
    assert names==CANONICAL+['analyst_note','custom_flag']
    assert [[row[names.index(c)] for c in old] for row in sheet.rows[1:]]==before[1:]


@pytest.mark.parametrize('case',['duplicate_header','unnamed_data','formula'])
def test_unsafe_legacy_structures_fail_without_write(case):
    sheet=Sheet()
    if case=='duplicate_header':sheet.rows[0][1]=sheet.rows[0][0]
    if case=='unnamed_data':sheet.rows[0][0]={}
    if case=='formula':sheet.rows[1][0]={'formulaValue':'=ROW()'}
    with pytest.raises(ValueError):migrate(sheet,CANONICAL,apply=True)
    assert not sheet.calls


def test_changed_since_reviewed_simulation_aborts():
    sheet=Sheet();dry=migrate(sheet,CANONICAL)
    sheet.rows[1][LEGACY.index('result')]=cell('void')
    with pytest.raises(ValueError,match='changed since'):
        migrate(sheet,CANONICAL,apply=True,expected_before=dry['snapshot_sha256'])
    assert not sheet.calls


def test_concurrent_change_aborts_before_batch():
    sheet=Sheet();read=sheet.fetch_sheet_metadata;count=0
    def mutate(params):
        nonlocal count
        count+=1
        if count==2:sheet.rows[1][0]=cell('changed')
        return read(params)
    sheet.fetch_sheet_metadata=mutate
    with pytest.raises(ValueError,match='Concurrent'):
        migrate(sheet,CANONICAL,apply=True)
    assert not sheet.calls


def test_atomic_request_failure_does_not_clear_or_rewrite_history():
    sheet=Sheet();before=deepcopy(sheet.rows)
    def fail(body):raise RuntimeError('API unavailable')
    sheet.batch_update=fail
    with pytest.raises(RuntimeError):migrate(sheet,CANONICAL,apply=True)
    assert sheet.rows==before


def test_changed_canonical_rows_ignore_legacy_fingerprint_without_writing():
    sheet=Sheet();review=migrate(sheet,CANONICAL)
    migrate(sheet,CANONICAL,apply=True,expected_before=review['snapshot_sha256'])
    sheet.calls.clear()
    sheet.rows[1][CANONICAL.index('result')]=cell('void')
    before=deepcopy(sheet.rows)
    done=migrate(sheet,CANONICAL,apply=True,expected_before=review['snapshot_sha256'])
    assert done['request_count']==0 and done['idempotent']
    assert sheet.rows==before and not sheet.calls


def test_observed_legacy_schema_with_56_rows_rejects_stale_snapshot_losslessly():
    sheet=Sheet()
    row=deepcopy(sheet.rows[1])
    sheet.rows=sheet.rows[:1]+[deepcopy(row) for _ in range(56)]
    for i,r in enumerate(sheet.rows[1:]):r[LEGACY.index('bet_id')]=cell(f'bet-{i}')
    before=deepcopy(sheet.rows)
    review=migrate(sheet,CANONICAL)
    assert review['rows_before']==review['rows_after']==56
    assert review['columns_added']==['id_match','id_joueur','outcome_key']
    with pytest.raises(ValueError,match='changed since approved simulation'):
        migrate(sheet,CANONICAL,apply=True,expected_before='obsolete-snapshot')
    assert sheet.rows==before and not sheet.calls
    done=migrate(sheet,CANONICAL,apply=True,expected_before=review['snapshot_sha256'])
    assert done['all_original_cells_preserved'] and done['result_preserved']
    assert done['bet_status_preserved'] and done['settled_at_preserved']
    assert done['duplicate_bet_ids_after']==0
