"""Collect public Unibet SSR markets; save only event data, never page/session tokens."""
import argparse
import json
import re
import sys
from pathlib import Path
import pandas as pd
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model'))
from henachel.unibet_ingestion import public_events, normalize_event, normalized_payload
from henachel.bookmaker_contract import validate_rows

HUB='https://www.unibet.fr/paris-hockey-sur-glace/etats-unis/nhl'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir',type=Path,default=Path('data/raw'))
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/structured_bookmaker'))
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    matches=pd.read_csv(args.raw_dir/'matchs.csv');roster=pd.read_csv(args.raw_dir/'roster_current.csv')
    response=requests.get(HUB,timeout=30);response.raise_for_status()
    paths=sorted(set(re.findall(r'/paris-hockey-sur-glace/etats-unis/nhl/\d+/[a-z0-9-]+',response.text)))
    rows=[];reports=[]
    for path in paths:
        if '-vs-' not in path:continue
        url='https://www.unibet.fr'+path
        response=requests.get(url,timeout=30)
        captured=pd.Timestamp.now(tz='UTC')
        if response.status_code!=200:
            reports.append(dict(url=url,http_status=response.status_code));continue
        if float(response.headers.get('Age','0'))>21600:raise ValueError('Stale HTTP cache')
        events=public_events(response.text)
        if len(events)!=1:raise ValueError('Ambiguous event page')
        event=events[0]
        (args.output_dir/f'event_{event["id"]}.json').write_text(json.dumps(dict(url=url,captured_at=captured.isoformat(),event=event),indent=2))
        try:
            accepted,rejected=normalize_event(event,url,captured,matches,roster,captured)
            rows.extend(accepted);reports.append(dict(url=url,rows=len(accepted),rejected=rejected))
        except ValueError as exc:
            reports.append(dict(url=url,reason=str(exc)))
    contract=validate_rows(rows,pd.Timestamp.now(tz='UTC'))
    payload=normalized_payload(rows,reports,contract)
    (args.output_dir/'normalized_points_odds.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(dict(rows=len(rows),contract=contract['status'],events=len(reports))))
    if contract['rejected']:raise SystemExit(1)
    if not rows and any(r.get('reason') or r.get('rejected') or r.get('http_status') for r in reports):
        raise SystemExit('No usable markets: source or identity failures; not a verified empty market day')


if __name__=='__main__':main()
