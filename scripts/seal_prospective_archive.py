"""Record integrity/hash and verify a SQLite backup restore before artifact upload."""
import argparse,hashlib,json,sqlite3
from pathlib import Path
from datetime import datetime,timezone


def seal(path):
    path=Path(path)
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as source:
        if source.execute('PRAGMA integrity_check').fetchone()[0]!='ok':raise ValueError('Archive corrupt')
        restored=sqlite3.connect(':memory:')
        try:
            source.backup(restored)
            original=source.execute('SELECT batch_id,kind,payload,archived_at_utc FROM batches ORDER BY batch_id').fetchall()
            copy=restored.execute('SELECT batch_id,kind,payload,archived_at_utc FROM batches ORDER BY batch_id').fetchall()
            if original!=copy:raise ValueError('Restore mismatch')
        finally:restored.close()
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(filename=path.name,sha256=digest,batches=len(original),restore_verified=True,created_at_utc=datetime.now(timezone.utc).isoformat())
    manifest=path.with_suffix('.manifest.json');temporary=manifest.with_suffix('.tmp')
    temporary.write_text(json.dumps(report,indent=2));temporary.replace(manifest)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);a=p.parse_args()
    print(json.dumps(seal(a.archive)))
