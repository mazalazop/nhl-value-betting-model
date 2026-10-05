"""Simulate or atomically migrate only history_raw. Credentials remain in memory."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'model'))
from henachel.sheets_auth import authorize_environment
from henachel.sheets_migration import migrate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--expected-snapshot')
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('publisher_schema',ROOT/'model/08_publish_to_google_sheet.py')
    publisher=importlib.util.module_from_spec(spec);spec.loader.exec_module(publisher)
    output=ROOT/'outputs/history_schema_migration.json';output.parent.mkdir(exist_ok=True)
    try:
        if args.apply and not args.expected_snapshot:raise ValueError('Apply requires a reviewed snapshot fingerprint')
        client=authorize_environment(readonly=not args.apply)
        sheet=client.open_by_key(os.environ['SHEET_ID'])
        if sheet.title!='Henachel':raise ValueError('Unexpected spreadsheet')
        report=migrate(sheet,publisher.HISTORY_OUTPUT_COLUMNS,apply=args.apply,expected_before=args.expected_snapshot)
        code=0
    except Exception as exc:
        report={'status':'failed','error_type':type(exc).__name__}
        code=1
    output.write_text(json.dumps(report,indent=2));print(json.dumps(report))
    return code


if __name__=='__main__':raise SystemExit(main())
