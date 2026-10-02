#!/usr/bin/env python3
"""Settle the canonical ledger locally; publish its view separately with 08."""
import argparse
import json
from pathlib import Path
import pandas as pd
from henachel.history import read_ledger,ledger_lock,atomic_csv
from henachel.settlement import settle

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-csv',type=Path,default=ROOT/'outputs/history/master_daily_bets_history.csv')
    parser.add_argument('--stats-csv',type=Path,default=ROOT/'data/raw/stats.csv')
    parser.add_argument('--matches-csv',type=Path,default=ROOT/'data/raw/matchs.csv')
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs')
    parser.add_argument('--run-date',default=None,help='Run metadata only; NHL final status determines settlement.')
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    with ledger_lock(args.history_csv):
        history=read_ledger(args.history_csv)
        stats=pd.read_csv(args.stats_csv);matches=pd.read_csv(args.matches_csv)
        updated,changes,unresolved=settle(history,stats,matches,pd.Timestamp.now(tz='UTC').isoformat())
        # Record revisions before updating ledger. A crash can repeat an audit record, never erase a result.
        audit=args.history_csv.with_name('settlement_revisions.csv')
        if len(changes):
            previous=pd.read_csv(audit) if audit.exists() else changes.iloc[:0]
            atomic_csv(pd.concat([previous,changes],ignore_index=True),audit)
        atomic_csv(updated,args.history_csv)
    atomic_csv(changes,args.output_dir/'09_settled_rows.csv')
    atomic_csv(unresolved,args.output_dir/'09_unresolved_pending_rows.csv')
    (args.output_dir/'09_settlement_summary.json').write_text(json.dumps(dict(status='ok',rows=len(updated),changed=len(changes),unresolved=len(unresolved),run_date=args.run_date),indent=2))

if __name__=='__main__': main()
