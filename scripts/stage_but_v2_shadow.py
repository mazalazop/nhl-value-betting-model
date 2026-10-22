#!/usr/bin/env python3
"""Read-only GitHub staging. Called only by the explicitly enabled shadow job."""
import argparse,json,subprocess
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from backup_but_v2_state import authenticated_store, restore, compact_verified_state
from pathlib import Path

REPO='mazalazop/nhl-value-betting-model'

def api(path):
    return json.loads(subprocess.check_output(['gh','api',f'repos/{REPO}/{path}'],text=True))

def validate_run(run):
    if (run['head_branch']!='main' or run['status']!='completed' or run['conclusion']!='success'
        or run['path']!='.github/workflows/Workflow global Henachel — POINTS.yml'
        or run['event'] not in {'schedule','workflow_dispatch','push'}):
        raise ValueError('Upstream must be a successful main production workflow, never a PR')


def discover_durable(store, prior, root):
    durable=store.latest()
    if durable is not None or not prior:return durable
    # A receipt is a discovery pointer only. All data still comes from the verified
    # durable release, never from an unchecked/expired Actions state snapshot.
    latest=max(prior,key=lambda a:a['created_at'])
    run_id=latest['workflow_run']['id'];previous=api(f'actions/runs/{run_id}')
    if previous['head_branch']!='main' or previous['path']!='.github/workflows/but-v2-shadow.yml':
        raise ValueError('Untrusted backup receipt workflow')
    artifacts=api(f'actions/runs/{run_id}/artifacts?per_page=100')['artifacts']
    receipts=[a for a in artifacts if a['name']=='but-v2-backup-receipt' and not a['expired']]
    if len(receipts)!=1:raise ValueError('No unique verified receipt to discover missing draft listing')
    destination=Path(root)/'recovery-receipt'
    subprocess.run(['gh','run','download',str(run_id),'--repo',REPO,'--name','but-v2-backup-receipt',
                    '--dir',str(destination)],check=True)
    receipt=json.loads((destination/'backup-receipt.json').read_text())
    if receipt.get('remote_restore_verified') is not True:raise ValueError('Unverified backup receipt')
    durable=store.by_release_id(receipt['release_id'],receipt['index_sha256'])
    if str(durable['workflow_run_id'])!=str(run_id):raise ValueError('Backup receipt/run mismatch')
    return durable


def stage(run_id,root,initialize=False):
    if not str(run_id).isdigit():raise ValueError('Numeric run ID required')
    run=api(f'actions/runs/{run_id}');validate_run(run)
    artifacts=api(f'actions/runs/{run_id}/artifacts?per_page=100')['artifacts']
    candidates=[a for a in artifacts if a['name'].startswith('henachel-model-outputs-') and not a['expired']]
    if len(candidates)!=1:raise ValueError('Exactly one unexpired model output artifact required')
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    subprocess.run(['gh','run','download',str(run_id),'--repo',REPO,'--name',candidates[0]['name'],'--dir',str(root/'upstream')],check=True)
    # Fixed name across runs; API returns newest first. Never fall back to a clean empty archive.
    prior=api('actions/artifacts?name=but-v2-shadow-state&per_page=100')['artifacts']
    store=authenticated_store()
    durable=discover_durable(store,prior,root)
    if durable is not None:
        if prior and max(int(a['workflow_run']['id']) for a in prior)>int(durable['workflow_run_id']):
            raise ValueError('Newer Actions state has no durable backup; repair it before capture')
        restore(store,durable,root/'state')
        compact_verified_state(root/'state',durable)
    elif prior:
        raise ValueError('Actions state exists without a verified durable backup; repair before capture')
    elif not initialize:raise ValueError('No archive: first start needs explicit initialize approval')
    else:(root/'state').mkdir()
    (root/'upstream_run.json').write_text(json.dumps(run,indent=2))
    return root

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--initialize',action='store_true')
    a=p.parse_args();stage(a.run_id,a.root,a.initialize)
