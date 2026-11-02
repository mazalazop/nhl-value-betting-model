"""Verify delivered research evidence and the untouched operational baseline, offline."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def verify():
 manifest=json.loads((ROOT/'research/v3/delivery_manifest.json').read_text())
 for name,digest in manifest['files'].items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Research evidence changed: '+name)
 protected=json.loads((ROOT/'research/v3/protected_files.json').read_text())
 for name,digest in protected.items():
  if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:raise ValueError('Protected baseline changed: '+name)
 print(f'Verified {len(manifest["files"])} research files and {len(protected)} protected operational/archive files')
if __name__=='__main__':verify()
