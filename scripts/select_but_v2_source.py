"""Discover a genuine completed V1 artifact; never reconstruct predictions."""
import json
import os
import subprocess

REPO = os.environ.get('GITHUB_REPOSITORY', 'mazalazop/nhl-value-betting-model')
PATH = '.github/workflows/Workflow global Henachel — POINTS.yml'


def api(path):
    return json.loads(subprocess.check_output(['gh', 'api', f'repos/{REPO}/{path}'], text=True))


def trusted(run):
    return (run['head_branch'] == 'main' and run['status'] == 'completed'
            and run['conclusion'] == 'success' and run['path'] == PATH
            and run['event'] in {'schedule', 'workflow_dispatch', 'push'})


def select(run_id=None):
    if run_id:
        if not str(run_id).isdigit():
            raise ValueError('Numeric run ID required')
        candidates = [api(f'actions/runs/{run_id}')]
        if not trusted(candidates[0]):
            raise ValueError('Untrusted upstream workflow')
    else:
        workflows = api('actions/workflows?per_page=100')['workflows']
        matches = [w for w in workflows if w['path'] == PATH and w['state'] == 'active']
        if len(matches) != 1:
            raise ValueError('Expected one active V1 workflow')
        runs = api(f"actions/workflows/{matches[0]['id']}/runs?branch=main&status=success&per_page=50")['workflow_runs']
        candidates = sorted((r for r in runs if trusted(r)), key=lambda r: r['created_at'], reverse=True)
    for run in candidates:
        artifacts = api(f"actions/runs/{run['id']}/artifacts?per_page=100")['artifacts']
        models = [a for a in artifacts if a['name'].startswith('henachel-model-outputs-') and not a['expired']]
        if len(models) > 1:
            raise ValueError('Ambiguous model artifact')
        if models:
            return str(run['id'])
    return None


if __name__ == '__main__':
    requested = os.environ.get('INPUT_RUN_ID', '')
    chosen = select(requested)
    if not chosen and requested and os.environ.get('EVENT_NAME') == 'workflow_dispatch':
        raise ValueError('Explicit run contains no usable model artifact')
    result = dict(eligible=bool(chosen), input_run_id=chosen,
                  note='Existing dates are report-only; fresh captures still require pregame V1 age <=2h.')
    print(json.dumps(result))
    with open(os.environ['GITHUB_OUTPUT'], 'a') as out:
        out.write(f"eligible={'true' if chosen else 'false'}\ninput_run_id={chosen or ''}\n")
