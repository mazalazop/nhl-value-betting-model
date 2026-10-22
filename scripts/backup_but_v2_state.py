"""Append-only incremental shadow backups in unpublished GitHub draft releases.

Each index is a complete manifest; immutable tar bundles hold only changed files.
No Actions expiry, deletion, publication, external service or production write.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
from urllib.parse import urlparse
import requests

PREFIX = 'henachel-but-v2-state-'
BUNDLE_LIMIT = 128 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def inventory(root):
    root = Path(root)
    paths = list(root.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('Symlink in state')
    return {p.relative_to(root).as_posix(): digest(p.read_bytes())
            for p in sorted(paths) if p.is_file()}


def validate_state(root):
    actual = inventory(root)
    manifest = json.loads((Path(root) / 'archive_manifest.json').read_text())
    expected = {k: v for k, v in actual.items() if k != 'archive_manifest.json'}
    if expected != manifest['files']:
        raise ValueError('State integrity mismatch')
    return actual


def safe_path(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or str(p) != name or not p.parts:
        raise ValueError('Unsafe backup path')
    return p


class Store:
    def __init__(self, session, repository, prefix=PREFIX):
        self.session = session
        self.api = 'https://api.github.com/repos/' + repository
        self.prefix = prefix

    def get(self, path, **kwargs):
        r = self.session.get(self.api + '/' + path, timeout=180, **kwargs)
        r.raise_for_status()
        return r

    def pages(self, path):
        result = []
        page = 1
        while True:
            batch = self.get(path, params={'per_page': 100, 'page': page}).json()
            result.extend(batch)
            if len(batch) < 100:
                return result
            page += 1

    def releases(self):
        values = [r for r in self.pages('releases') if r['tag_name'].startswith(self.prefix)]
        if any(not r['draft'] for r in values):
            raise ValueError('Backup release was published; manual review required')
        return sorted(values, key=lambda r: r['id'], reverse=True)

    def assets(self, release):
        return {a['name']: a for a in self.pages(f"releases/{release['id']}/assets")}

    def download(self, asset_id):
        return self.get(f'releases/assets/{asset_id}',
                        headers={'Accept': 'application/octet-stream'}).content

    def upload(self, release, name, data):
        existing = self.assets(release).get(name)
        if existing:
            if self.download(existing['id']) != data:
                raise ValueError('Immutable asset collision')
            return existing['id']
        url = release['upload_url'].split('{')[0]
        if urlparse(url).scheme != 'https' or urlparse(url).hostname != 'uploads.github.com':
            raise ValueError('Unexpected upload host')
        r = self.session.post(url, params={'name': name}, data=data,
                              headers={'Content-Type': 'application/octet-stream'}, timeout=180)
        r.raise_for_status()
        asset_id = r.json()['id']
        if self.download(asset_id) != data:
            raise ValueError('Remote asset checksum mismatch')
        return asset_id

    def latest(self, allow_incomplete=False):
        for release in self.releases():
            assets = self.assets(release)
            if 'verified.json' not in assets or 'index.json' not in assets:
                if allow_incomplete:
                    continue
                raise ValueError('Newest durable backup incomplete; repair before next capture')
            raw = self.download(assets['index.json']['id'])
            receipt = json.loads(self.download(assets['verified.json']['id']))
            if digest(raw) != receipt['index_sha256']:
                raise ValueError('Backup index checksum mismatch')
            return json.loads(raw)
        return None

    def create(self, tag, commit):
        matching = [r for r in self.releases() if r['tag_name'] == tag]
        if matching:
            return matching[0]
        r = self.session.post(self.api + '/releases', json={
            'tag_name': tag, 'target_commitish': commit, 'draft': True, 'prerelease': True,
            'name': tag, 'body': 'BUT V2 scientific registry. Append-only incremental backup. '
            'Keep every bundle and index for at least 365 days after study closure. Never publish or delete.'}, timeout=60)
        r.raise_for_status()
        return r.json()


def restore(store, index, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    bundles = {}
    for name, meta in index['files'].items():
        safe_path(name)
        bundles.setdefault(meta['asset_id'], []).append((name, meta))
    for asset_id, entries in bundles.items():
        raw = store.download(asset_id)
        if any(digest(raw) != meta['bundle_sha256'] for _, meta in entries):
            raise ValueError('Bundle checksum mismatch')
        with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
            for name, meta in entries:
                member = archive.getmember(name)
                if not member.isfile():
                    raise ValueError('Non-file backup member')
                payload = archive.extractfile(member).read()
                if digest(payload) != meta['sha256']:
                    raise ValueError('Restored file checksum mismatch')
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
    actual = validate_state(destination)
    if actual != {name: m['sha256'] for name, m in index['files'].items()}:
        raise ValueError('Restored inventory mismatch')
    return actual



def compact_verified_state(root, index):
    """Drop only reproducible old views from working copy, after durable restoration.

    Every removed byte remains in the immutable snapshot named in the receipt.
    Captures, source data, status logs and latest reports are never removed.
    """
    root = Path(root)
    actual = validate_state(root)
    if actual != {name: meta['sha256'] for name, meta in index['files'].items()}:
        raise ValueError('Compaction requires exact verified durable snapshot')
    pointer = root / 'latest_batch.json'
    if not pointer.exists():
        return []
    latest = json.loads(pointer.read_text())['path']
    removed = []
    for name in actual:
        parts = PurePosixPath(name).parts
        if (len(parts) >= 3 and parts[0] == 'batches'
            and '/'.join(parts[:2]) != latest
            and parts[2] in {'reports', 'cumulative', 'outcomes.csv'}):
            (root / name).unlink(); removed.append(name)
    if removed:
        receipt = root / 'compactions' / (index['state_sha256'] + '.json')
        receipt.parent.mkdir(exist_ok=True)
        receipt.write_text(json.dumps(dict(durable_snapshot=index['state_sha256'],
            workflow_run_id=index['workflow_run_id'], removed_working_views=sorted(removed)), sort_keys=True))
        files = inventory(root); files.pop('archive_manifest.json', None)
        (root / 'archive_manifest.json').write_text(json.dumps(dict(files=files), sort_keys=True))
    return removed


def backup(store, root, commit, run_id):
    root = Path(root)
    files = validate_state(root)
    state_hash = digest(json.dumps(files, sort_keys=True).encode())
    previous = store.latest(allow_incomplete=True)
    old = previous['files'] if previous else {}
    index = dict(version=1, workflow_run_id=str(run_id), commit=commit,
                 state_sha256=state_hash, files={k: old[k] for k in files
                 if k in old and old[k]['sha256'] == files[k]})
    changed = [k for k in files if k not in index['files']]
    groups = []; group = []; size = 0
    for name in changed:
        safe_path(name)
        n = (root / name).stat().st_size
        if n > BUNDLE_LIMIT:
            raise ValueError('Individual file exceeds 128 MiB budget; archive rotation required')
        if size + n > BUNDLE_LIMIT and group:
            groups.append(group); group = []; size = 0
        group.append(name); size += n
    if group:
        groups.append(group)
    if len(groups) > 990:
        raise ValueError('Release capacity budget exceeded')
    release = store.create(store.prefix + state_hash, commit)
    for group in groups:
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode='w') as archive:
            for name in group:
                payload = (root / name).read_bytes()
                info = tarfile.TarInfo(name); info.size = len(payload); info.mtime = 0
                archive.addfile(info, io.BytesIO(payload))
        raw = gzip.compress(buffer.getvalue(), mtime=0); bundle_hash = digest(raw)
        asset_id = store.upload(release, bundle_hash + '.tar.gz', raw)
        for name in group:
            index['files'][name] = dict(asset_id=asset_id, bundle_sha256=bundle_hash, sha256=files[name])
    # Verify full restoration, including unchanged files in earlier releases, before commit marker.
    with tempfile.TemporaryDirectory() as temp:
        restored = restore(store, index, Path(temp) / 'restored')
        if restored != files:
            raise ValueError('Full remote restore failed')
    raw = json.dumps(index, sort_keys=True).encode()
    store.upload(release, 'index.json', raw)
    receipt = dict(index_sha256=digest(raw), remote_restore_verified=True,
                   files=len(files), bytes=sum((root / n).stat().st_size for n in files))
    store.upload(release, 'verified.json', json.dumps(receipt, sort_keys=True).encode())
    return dict(release_id=release['id'], tag=release['tag_name'], **receipt)


def authenticated_store(prefix=PREFIX):
    session = requests.Session()
    session.headers.update({'Authorization': 'Bearer ' + os.environ['GH_TOKEN'],
                            'Accept': 'application/vnd.github+json'})
    return Store(session, os.environ.get('GITHUB_REPOSITORY', 'mazalazop/nhl-value-betting-model'), prefix)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['backup', 'restore'])
    p.add_argument('--state', type=Path, required=True)
    p.add_argument('--prefix', default=PREFIX)
    args = p.parse_args(); store = authenticated_store(args.prefix)
    if args.action == 'backup':
        print(json.dumps(backup(store, args.state, os.environ['GITHUB_SHA'], os.environ['GITHUB_RUN_ID'])))
    else:
        index = store.latest()
        if index is None:
            raise ValueError('No durable backup')
        restored = restore(store, index, args.state)
        print(json.dumps(dict(files=len(restored), state_sha256=index['state_sha256'])))
