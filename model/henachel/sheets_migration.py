"""Lossless schema migration: insert/move columns and write only new headers."""
from copy import deepcopy
import hashlib
import json


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def snapshot(metadata):
    sheets=[s for s in metadata.get('sheets',[]) if s['properties']['title']=='history_raw']
    if len(sheets)!=1:raise ValueError('Expected exactly one history_raw worksheet')
    sheet=sheets[0];props=sheet['properties'];cells={}
    for block in sheet.get('data',[]):
        for r,row in enumerate(block.get('rowData',[]),block.get('startRow',0)):
            for c,cell in enumerate(row.get('values',[]),block.get('startColumn',0)):
                value=cell.get('userEnteredValue',{})
                if value:cells[r,c]=value
    height=max((r+1 for r,c in cells),default=0)
    width=max((c+1 for r,c in cells),default=0)
    rows=[[deepcopy(cells.get((r,c),{})) for c in range(width)] for r in range(height)]
    return {'sheet_id':props['sheetId'],'grid_columns':props['gridProperties']['columnCount'],
            'rows':rows}


def headers(rows):
    if not rows:return []
    result=[]
    for cell in rows[0]:
        value=cell.get('stringValue')
        if not isinstance(value,str) or not value.strip():
            raise ValueError('Blank/non-text header above populated columns: explicit review required')
        result.append(value)
    if len(result)!=len(set(result)):raise ValueError('Duplicate headers: explicit review required')
    return result


def projection(rows,columns):
    names=headers(rows)
    return [[row[names.index(c)] for c in columns] for row in rows[1:]]


def plan_migration(before,canonical):
    original=deepcopy(before['rows']);old=headers(original)
    if len(canonical)!=len(set(canonical)):raise ValueError('Duplicate canonical columns')
    target=list(canonical)+[c for c in old if c not in canonical]
    if old!=target and any('formulaValue' in cell for row in original for cell in row):
        raise ValueError('Formula cells require explicit migration review')
    current=list(old);rows=deepcopy(original) or [[]]
    requests=[];sheet_id=before['sheet_id'];grid_columns=before['grid_columns']
    for destination,column in enumerate(target):
        if column not in current:
            if destination>=grid_columns:
                requests.append({'appendDimension':{'sheetId':sheet_id,'dimension':'COLUMNS','length':1}})
            else:
                requests.append({'insertDimension':{'range':{'sheetId':sheet_id,'dimension':'COLUMNS','startIndex':destination,'endIndex':destination+1},'inheritFromBefore':False}})
            grid_columns+=1
            requests.append({'updateCells':{'range':{'sheetId':sheet_id,'startRowIndex':0,'endRowIndex':1,
                'startColumnIndex':destination,'endColumnIndex':destination+1},
                'rows':[{'values':[{'userEnteredValue':{'stringValue':column}}]}],'fields':'userEnteredValue'}})
            current.insert(destination,column)
            for row in rows:row.insert(destination,{})
            rows[0][destination]={'stringValue':column}
        else:
            source=current.index(column)
            if source!=destination:
                assert source>destination  # Earlier target columns are already fixed.
                requests.append({'moveDimension':{'source':{'sheetId':sheet_id,'dimension':'COLUMNS',
                    'startIndex':source,'endIndex':source+1},'destinationIndex':destination}})
                current.insert(destination,current.pop(source))
                for row in rows:row.insert(destination,row.pop(source))
    after={'sheet_id':sheet_id,'grid_columns':grid_columns,'rows':rows}
    # Compare every original typed cell, including unknown legacy columns and blank rows.
    old_values=original[1:] if original else []
    restored=projection(rows,old) if old else []
    if old_values!=restored:raise ValueError('Simulation lost historical data')
    report={'headers_before':old,'headers_after':target,'columns_added':[c for c in canonical if c not in old],
        'legacy_extra_columns':[c for c in old if c not in canonical],
        'rows_before':max(len(original)-1,0),'rows_after':len(rows)-1,
        'all_original_cells_preserved':True,'original_values_sha256':fingerprint(old_values),
        'projected_values_sha256':fingerprint(restored),'request_count':len(requests)}
    for column in ['bet_id','result','bet_status','actual_stat_value','settled_at']:
        if column in old:
            values=projection(original,[column])
            report[column+'_preserved']=values==projection(rows,[column])
    if 'bet_id' in old:
        values=[json.dumps(r[0],sort_keys=True) for r in projection(original,['bet_id']) if r[0]]
        report['duplicate_bet_ids_before']=len(values)-len(set(values))
        report['duplicate_bet_ids_after']=report['duplicate_bet_ids_before']
    return requests,after,report


def read_snapshot(sheet):
    return snapshot(sheet.fetch_sheet_metadata(params={'includeGridData':True,'ranges':"'history_raw'",
        'fields':'sheets(properties(sheetId,title,gridProperties),data(startRow,startColumn,rowData(values(userEnteredValue))))'}))


def migrate(sheet,canonical,*,apply=False,expected_before=None):
    before=read_snapshot(sheet)
    requests,after,report=plan_migration(before,canonical)
    report.update(mode='apply' if apply else 'simulation',snapshot_sha256=fingerprint(before),status='simulated')
    if requests and expected_before is not None and report['snapshot_sha256']!=expected_before:
        raise ValueError('Sheet changed since approved simulation; regenerate report')
    if not apply:return report
    if read_snapshot(sheet)!=before:raise ValueError('Concurrent change detected before migration')
    if requests:sheet.batch_update({'requests':requests})
    actual=read_snapshot(sheet)
    if actual['sheet_id']!=after['sheet_id'] or actual['rows']!=after['rows']:
        raise ValueError('Post-write verification failed; do not retry a blind overwrite')
    report.update(status='verified',rows_after=max(len(actual['rows'])-1,0),
                  after_sha256=fingerprint(actual),idempotent=not plan_migration(actual,canonical)[0])
    return report
