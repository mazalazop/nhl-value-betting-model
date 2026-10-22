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
    durable=store.latest()
    if durable is not None:
        if prior and max(int(a['workflow_run']['id']) for a in prior)>int(durable['workflow_run_id']):
            raise ValueError('Newer Actions state has no durable backup; repair it before capture')
        restore(store,durable,root/'state')
        compact_verified_state(root/'state',durable)
    elif prior:
        latest=max(prior,key=lambda a:a['created_at'])
        if latest['expired']:raise ValueError('Latest shadow archive expired: restore durable backup; do not reset')
        previous=api(f"actions/runs/{latest['workflow_run']['id']}")
        if previous['head_branch']!='main' or previous['path']!='.github/workflows/but-v2-shadow.yml':raise ValueError('Untrusted shadow state artifact')
        subprocess.run(['gh','run','download',str(latest['workflow_run']['id']),'--repo',REPO,'--name','but-v2-shadow-state','--dir',str(root/'state')],check=True)
        if not (root/'state/archive_manifest.json').exists():raise ValueError('Restored archive has no integrity manifest')
        manifest=json.loads((root/'state/archive_manifest.json').read_text())
        import hashlib
        actual={str(p.relative_to(root/'state')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'state').rglob('*') if p.is_file() and p.name!='archive_manifest.json'}
        if actual!=manifest['files']:raise ValueError('Restored archive hash/coverage mismatch')
    elif not initialize:raise ValueError('No archive: first start needs explicit initialize approval')
    else:(root/'state').mkdir()
    (root/'upstream_run.json').write_text(json.dumps(run,indent=2))
    return root

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run-id',required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--initialize',action='store_true')
    a=p.parse_args();stage(a.run_id,a.root,a.initialize)
