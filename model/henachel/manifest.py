"""Run provenance from explicitly supplied data files; never inspect environment secrets."""
import hashlib
from importlib.metadata import version,PackageNotFoundError
from pathlib import Path
import platform
import subprocess
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]


def manifest(inputs=(),parameters=None):
    packages={}
    for package in ['numpy','pandas','scikit-learn','scipy','requests','gspread','google-auth','joblib']:
        try:packages[package]=version(package)
        except PackageNotFoundError:packages[package]='unavailable'
    try:
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        dirty=bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True))
    except (OSError,subprocess.CalledProcessError):commit=None;dirty=None
    sources=[]
    for source in inputs:
        path=Path(source)
        if any('secret' in part.lower() or 'credential' in part.lower() or part.startswith('.env') for part in path.parts):
            raise ValueError('Sensitive paths excluded from provenance')
        item=dict(path=str(path),available=path.is_file())
        if path.is_file():
            digest=hashlib.sha256()
            with path.open('rb') as data:
                for chunk in iter(lambda:data.read(1024*1024),b''):digest.update(chunk)
            item.update(sha256=digest.hexdigest(),bytes=path.stat().st_size)
        sources.append(item)
    return dict(commit=commit,tracked_worktree_dirty=dirty,created_at_utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),platform=platform.platform(),dependencies=packages,parameters=parameters or {},sources=sources)
