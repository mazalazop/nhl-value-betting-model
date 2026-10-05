"""Select only outputs created during this run, never stale self-hosted runner files."""
import os
from pathlib import Path
import sys

root=Path(sys.argv[1]);started=float(os.environ['HENACHEL_RUN_STARTED'])
paths=[p for p in root.iterdir() if p.is_dir() and p.stat().st_mtime>=started]
if not paths: raise SystemExit(f'No fresh scraper output under {root}')
print(max(paths,key=lambda p:p.stat().st_mtime))
