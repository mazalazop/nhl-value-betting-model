"""Idempotency gate: only a successful run with a verified receipt counts."""
import io
import json
import os
import subprocess
import zipfile


def api(path, binary=False):
    payload = subprocess.check_output(['gh', 'api', 'repos/'+os.environ['GITHUB_REPOSITORY']+'/'+path])
    return payload if binary else json.loads(payload)


def already_verified(day, ref, goal_required, get=api):
    name = 'henachel-production-verified-' + day
    page = 1
    while True:
        artifacts = get(f'actions/artifacts?name={name}&per_page=100&page={page}')['artifacts']
        for artifact in artifacts:
            if artifact['expired'] or artifact['name'] != name: continue
            run = get(f"actions/runs/{artifact['workflow_run']['id']}")
            if (run['head_branch'] != ref or run['conclusion'] != 'success'
                    or run['path'] != '.github/workflows/Workflow global Henachel — POINTS.yml'):
                continue
            payload = get(f"actions/artifacts/{artifact['id']}/zip", binary=True)
            with zipfile.ZipFile(io.BytesIO(payload)) as z:
                receipt = json.loads(z.read('production_verified.json'))
            if (receipt.get('status') == 'verified' and receipt.get('run_date') == day
                    and str(receipt.get('run_id')) == str(run['id'])
                    and 'POINT' in receipt.get('markets', {})
                    and (not goal_required or 'GOAL' in receipt.get('markets', {}))):
                return run['id']
        if len(artifacts) < 100: return None
        page += 1


if __name__ == '__main__':
    previous = already_verified(os.environ['RUN_DATE'], os.environ['GITHUB_REF_NAME'],
                                os.environ['HENACHEL_GOAL_RESEARCH_ENABLED'] == 'true')
    print(json.dumps(dict(already_verified=bool(previous), previous_run_id=previous)))
    with open(os.environ['GITHUB_OUTPUT'], 'a') as out:
        out.write(f"already_verified={'true' if previous else 'false'}\n")
    with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as out:
        out.write(f'Production already verified in run {previous}; no duplicate computation.\n' if previous
                  else 'No verified production receipt for this NHL date; computation required.\n')
