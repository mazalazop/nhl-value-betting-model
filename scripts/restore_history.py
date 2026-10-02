"""Restore branch-scoped ledger artifacts. Missing/expired history fails closed."""
import argparse
import hashlib
import io
import os
from pathlib import Path
import zipfile
import requests


def artifact_name(ref):
    return 'henachel-canonical-history-'+hashlib.sha256(ref.encode()).hexdigest()[:16]


def restore(session,repository,ref,destination,bootstrap=False):
    name=artifact_name(ref)
    response=session.get(f'https://api.github.com/repos/{repository}/actions/artifacts',params={'name':name,'per_page':100},timeout=60)
    response.raise_for_status(); artifacts=response.json()['artifacts']
    if not artifacts:
        if not bootstrap: raise ValueError('No canonical history artifact: import existing history or explicitly bootstrap a new ledger')
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model'))
        from henachel.history import read_ledger,atomic_csv
        path=destination/'master_daily_bets_history.csv'
        if path.exists():raise ValueError('Existing local history cannot be bootstrapped')
        atomic_csv(read_ledger(path),path)
        return 'bootstrap' 
    newest=max(artifacts,key=lambda x:x['created_at'])
    if newest['expired']: raise ValueError('Canonical history expired; restore a durable backup')
    data=session.get(newest['archive_download_url'],timeout=60);data.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(data.content)) as archive:
        files={Path(p).name:p for p in archive.namelist() if not p.endswith('/')}
        canonical='master_daily_bets_history.csv'
        if canonical not in files: raise ValueError('Artifact lacks canonical ledger')
        destination.mkdir(parents=True,exist_ok=True)
        for filename in [canonical,'settlement_revisions.csv']:
            if filename in files:
                target=destination/filename
                if target.exists():raise ValueError('Refuse to overwrite existing local history during restore')
                target.write_bytes(archive.read(files[filename]))
    return 'restored'

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--bootstrap',action='store_true');args=parser.parse_args()
    session=requests.Session();session.headers.update({'Authorization':'Bearer '+os.environ['GH_TOKEN'],'Accept':'application/vnd.github+json'})
    result=restore(session,os.environ['GITHUB_REPOSITORY'],os.environ['GITHUB_REF'],Path('outputs/history'),args.bootstrap)
    with Path(os.environ['GITHUB_OUTPUT']).open('a') as out:
        out.write(f'artifact_name={artifact_name(os.environ["GITHUB_REF"])}\nstatus={result}\n')
