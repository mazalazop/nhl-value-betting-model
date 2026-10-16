"""Content-addressed, non-overwriting SQLite backups in a private draft release.

Authentication stays in memory. No release is published; no asset is overwritten.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse
import requests
from seal_prospective_archive import seal

TAG = 'henachel-prospective-backups'


def backup(session, repository, archive, commit):
    report = seal(archive)
    raw = Path(archive).read_bytes()
    compressed = gzip.compress(raw, mtime=0)
    if gzip.decompress(compressed) != raw:
        raise ValueError('Backup roundtrip mismatch')
    name = 'prospective-' + report['sha256'] + '.sqlite3.gz'
    api = f'https://api.github.com/repos/{repository}'
    releases = []
    page = 1
    while True:
        response = session.get(f'{api}/releases', params={'per_page': 100, 'page': page}, timeout=60)
        response.raise_for_status()
        batch = response.json(); releases.extend(r for r in batch if r['tag_name'] == TAG)
        if len(batch) < 100: break
        page += 1
    if len(releases) > 1: raise ValueError('Ambiguous backup release')
    if releases:
        release = releases[0]
    else:
        response = session.post(f'{api}/releases', json={
            'tag_name': TAG, 'target_commitish': commit,
            'name': 'Henachel prospective backups — internal research',
            'body': 'Append-only content-addressed archives. Do not publish or replace assets.',
            'draft': True, 'prerelease': True}, timeout=60)
        response.raise_for_status(); release = response.json()
    if release.get('draft') is not True:
        raise ValueError('Backup release must remain an unpublished draft')
    # Paginate: daily history must not accidentally skip assets beyond page one.
    assets = []
    page = 1
    while True:
        response = session.get(f'{api}/releases/{release["id"]}/assets',
                               params={'per_page': 100, 'page': page}, timeout=60)
        response.raise_for_status()
        batch = response.json(); assets.extend(batch)
        if len(batch) < 100: break
        page += 1
    existing = [a for a in assets if a['name'] == name]
    if len(existing) > 1: raise ValueError('Duplicate backup asset')
    if existing:
        asset = existing[0]
    else:
        url = release['upload_url'].split('{')[0]
        if urlparse(url).scheme != 'https' or urlparse(url).hostname != 'uploads.github.com':
            raise ValueError('Unexpected upload host')
        response = session.post(url, params={'name': name}, data=compressed,
                                headers={'Content-Type': 'application/gzip'}, timeout=120)
        response.raise_for_status(); asset = response.json()
    downloaded = session.get(f'{api}/releases/assets/{asset["id"]}',
                             headers={'Accept': 'application/octet-stream'}, timeout=120)
    downloaded.raise_for_status()
    restored = gzip.decompress(downloaded.content)
    if hashlib.sha256(restored).hexdigest() != report['sha256'] or restored != raw:
        raise ValueError('Remote backup restore mismatch; no overwrite attempted')
    return dict(asset_id=asset['id'], name=name, sha256=report['sha256'],
                remote_restore_verified=True, status='existing' if existing else 'created')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    args = parser.parse_args()
    session = requests.Session()
    session.headers.update({'Authorization': 'Bearer ' + os.environ['GH_TOKEN'],
                            'Accept': 'application/vnd.github+json'})
    print(json.dumps(backup(session, os.environ['GITHUB_REPOSITORY'], args.archive,
                            os.environ['GITHUB_SHA'])))
