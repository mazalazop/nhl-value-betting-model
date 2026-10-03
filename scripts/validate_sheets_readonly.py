"""Read only spreadsheet metadata and column headers. No creation, update or deletion."""
import json
import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model'))
from henachel.sheets_auth import authorize_environment


def inspect(client,sheet_id):
    sheet=client.open_by_key(sheet_id)
    if sheet.title!='Henachel':raise ValueError('Unexpected spreadsheet title')
    worksheets={ws.title:ws for ws in sheet.worksheets()}
    report={'status':'read_access_ok','mode':'readonly','write_permissions':'not_tested','worksheets':{}}
    for title in ['daily_picks','history_raw']:
        if title not in worksheets:
            report['worksheets'][title]={'status':'missing'}
            continue
        headers=worksheets[title].row_values(1)
        required=['bet_id','id_match','id_joueur','result','bet_status']
        report['worksheets'][title]={'status':'inspected','column_count':len(headers),'missing_canonical_columns':[c for c in required if c not in headers]}
    return report


def main():
    try:
        sheet_id=os.environ.get('SHEET_ID','').strip()
        if not sheet_id:raise ValueError('Missing sheet identifier')
        report=inspect(authorize_environment(readonly=True),sheet_id)
        code=0
    except Exception as exc:
        report={'status':'unavailable','mode':'readonly','error_type':type(exc).__name__}
        code=1
    output=Path('outputs/sheets_readonly_validation.json');output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report,indent=2));print(json.dumps(report))
    return code


if __name__=='__main__':raise SystemExit(main())
