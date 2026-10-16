"""Small public NHL cache with genuine retrieval times and bounded 429 handling."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import gzip
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


def get_json(session, url, cache, ttl_seconds, sleep=time.sleep):
    key = hashlib.sha256(url.encode()).hexdigest(); path = Path(cache)/(key+'.json.gz')
    clock = datetime.now(timezone.utc)
    if path.exists():
        with gzip.open(path, 'rt') as f: saved = json.load(f)
        observed = datetime.fromisoformat(saved['observed_at'])
        if observed.tzinfo is None: raise ValueError('Naive cached timestamp')
        age = (clock-observed).total_seconds()
        if saved['url'] != url: raise ValueError('Cache provenance mismatch')
        if 0 <= age <= ttl_seconds: return saved
    for attempt in range(3):
        sleep(1.5)
        response = session.get(url, timeout=30)
        observed = datetime.now(timezone.utc)
        if response.status_code not in {429, 503} or attempt == 2:
            response.raise_for_status(); break
        retry = response.headers.get('Retry-After', '30')
        try: delay = float(retry)
        except ValueError: delay = (parsedate_to_datetime(retry)-observed).total_seconds()
        if delay > 60: raise RuntimeError('NHL Retry-After exceeds bounded wait; rerun later with cache')
        sleep(max(1, delay))
    saved = dict(url=url, observed_at=observed.isoformat(), document=response.json())
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(dir=path.parent, suffix='.tmp')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(gzip.compress(json.dumps(saved).encode())); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)
    return saved
