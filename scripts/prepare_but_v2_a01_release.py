"""Seal the reviewed A01 release, without assigning T0 or enabling execution."""
import hashlib
import json
from pathlib import Path


def main():
    original=json.loads(Path('docs/but_v2/PRE_T0_REGISTRATION.json').read_text())['file_sha256']
    files=list(original)+[
        'model/henachel/but_v2_participation_a01.py','model/henachel/but_v2_a01_registry.py',
        'scripts/run_but_v2_a01.py','scripts/backup_but_v2_state.py',
        'configs/but_v2_a01_proposed.json','docs/but_v2/A01_STARTUP.md','docs/but_v2/A01_PROTOCOL_PREPARATION_1.md',
        '.github/workflows/but-v2-a01.yml']
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
    if any(hashes[p]!=v for p,v in original.items()):raise ValueError('Original frozen study drift')
    value=dict(version='A01_RELEASE_1',files=hashes,promotion_allowed=False)
    Path('configs/but_v2_a01_release.json').write_text(json.dumps(value,indent=2)+'\n')


if __name__=='__main__':main()
