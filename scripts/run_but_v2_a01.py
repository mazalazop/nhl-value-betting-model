"""Isolated A01 lifecycle. Production mode requires a preregistered activation record."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
import pandas as pd
import requests
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model'))
from henachel.but_v2_a01_registry import cycle, validate_start, write
from backup_but_v2_state import authenticated_store, restore, backup


def configuration(mode):
    start=json.loads(Path('configs/but_v2_a01_start.json').read_text())
    release=json.loads(Path('configs/but_v2_a01_release.json').read_text())
    if mode=='PROSPECTIVE':validate_start(start,release,datetime.now(timezone.utc).isoformat())
    return start,release


def prefix(mode,start):
    if mode=='EXPLORATORY':return 'henachel-but-v2-a01-exploratory-'
    study=start['study_id']
    if not study.replace('-','').isalnum():raise ValueError('Unsafe study ID')
    return 'henachel-but-v2-a01-'+study+'-'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['stage','evaluate','backup'])
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--mode',choices=['EXPLORATORY','PROSPECTIVE'],required=True)
    p.add_argument('--decisions',type=Path)
    p.add_argument('--initialize',action='store_true',help='First registry only; never recovery')
    args=p.parse_args();start,release=configuration(args.mode)
    root=args.root;root.mkdir(parents=True,exist_ok=True)
    state=root/'a01-state';original=root/'original-state'
    if args.action=='stage':
        store=authenticated_store()
        if args.mode=='PROSPECTIVE':
            commits=store.get('commits',params={'path':'configs/but_v2_a01_start.json','sha':'main','per_page':1}).json()
            if not commits:raise ValueError('No committed preregistration')
            from henachel.but_v2_participation_a01 import utc
            committed=commits[0]['commit']['committer']['date']
            if utc(committed)>=utc(start['t0_utc']):raise ValueError('Registration commit is not before T0')
            write(root/'preregistration-receipt.json',dict(commit=commits[0]['sha'],committed_at=committed,start=start))
        idx=store.latest()
        if idx is None:raise ValueError('No original durable registry')
        restore(store,idx,original)
        write(root/'original-receipt.json',dict(snapshot=idx['state_sha256'],workflow_run_id=idx['workflow_run_id']))
        a01=authenticated_store(prefix(args.mode,start));prior=a01.latest()
        if prior:
            restore(a01,prior,state)
        else:
            if not args.initialize:raise ValueError('Missing durable A01 state; never reset without explicit first initialization')
            workflows=store.get('actions/workflows',params={'per_page':100}).json()['workflows']
            matches=[w for w in workflows if w['path']=='.github/workflows/but-v2-a01.yml']
            if matches:
                runs=store.get(f"actions/workflows/{matches[0]['id']}/runs",params={'branch':'main','per_page':100}).json()['workflow_runs']
                for run in runs:
                    if str(run['id'])==os.environ.get('GITHUB_RUN_ID'):continue
                    artifacts=store.get(f"actions/runs/{run['id']}/artifacts").json()['artifacts']
                    if any(a['name']=='a01-state-'+args.mode for a in artifacts):
                        raise ValueError('Prior A01 artifact exists: restore its durable receipt, never reinitialize')
    elif args.action=='evaluate':
        with (root/'a01.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            evidence={};errors=[]
            gids=set()
            for cap in (original/'captures').glob('*/predictions.csv'):
                gids.update(int(g) for g in pd.read_csv(cap).id_match.unique())
            for gid in sorted(gids):
                evidence[str(gid)]={}
                for endpoint in ['boxscore','play-by-play']:
                    url=f'https://api-web.nhle.com/v1/gamecenter/{gid}/{endpoint}'
                    try:
                        response=requests.get(url,timeout=35);response.raise_for_status()
                        evidence[str(gid)][endpoint]=dict(source=url,document=response.json(),observed_at_utc=datetime.now(timezone.utc).isoformat())
                    except (requests.RequestException,ValueError) as exc:
                        errors.append(dict(source=url,error=str(exc),observed_at_utc=datetime.now(timezone.utc).isoformat()))
            as_of=datetime.now(timezone.utc).isoformat()
            run_id=os.environ.get('GITHUB_RUN_ID',datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S'))+'-'+os.environ.get('GITHUB_RUN_ATTEMPT','1')
            review_path=args.decisions or Path('configs/but_v2_a01_reviews.json')
            decisions=json.loads(review_path.read_text())
            pending=[]
            for decision in decisions:
                stored=state/'decisions'/(decision['case_id']+'.json')
                if stored.exists() and json.loads(stored.read_text())==decision:continue
                pending.append(decision)
            decisions=pending
            # Errors remain evidence, outside NHL document maps, and are backed up after sealing.
            result=cycle(original,state,evidence,as_of,run_id,args.mode,start,release,decisions)
            write(state/'collection-errors'/(run_id+'.json'),errors)
            from henachel.but_v2_a01_registry import seal
            seal(state)
            print(json.dumps(result,indent=2))
    else:
        print(json.dumps(backup(authenticated_store(prefix(args.mode,start)),state,
                               os.environ['GITHUB_SHA'],os.environ['GITHUB_RUN_ID'])))


if __name__=='__main__':main()
