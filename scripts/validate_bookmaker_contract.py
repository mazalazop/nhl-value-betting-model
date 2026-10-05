"""Inspect a real normalized artifact, without changing it or manufacturing metadata."""
import argparse
import json
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model'))
from henachel.bookmaker_contract import validate_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',type=Path)
    parser.add_argument('--output',type=Path,default=Path('outputs/bookmaker_contract_validation.json'))
    args=parser.parse_args()
    payload=json.loads(args.input.read_text())
    if not isinstance(payload,dict) or not isinstance(payload.get('rows'),list):raise ValueError('Expected rows list')
    report=validate_rows(payload['rows'],pd.Timestamp.now(tz='UTC'))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='rejected'}))
    return 1 if report['status']=='invalid_contract' else 0


if __name__=='__main__':raise SystemExit(main())
