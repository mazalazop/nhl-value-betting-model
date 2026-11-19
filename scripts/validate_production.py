"""Fail closed on incomplete daily outputs. Does not alter predictions or history."""
import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path


def utc(value):
    stamp = dt.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if stamp.tzinfo is None:
        raise ValueError('Timestamp must include timezone')
    return stamp.astimezone(dt.timezone.utc)


def validate(root, day, started, goal_required=True, now=None):
    root = Path(root)
    dt.date.fromisoformat(day)
    clock = now or dt.datetime.now(dt.timezone.utc)
    began = utc(started)
    manifest = dict(schema_version=1, run_date=day, run_id=os.getenv('GITHUB_RUN_ID'),
                    commit=os.getenv('GITHUB_SHA'), validated_at_utc=clock.isoformat(),
                    goal_production_approved=False, files={}, markets={})

    def read(name, as_json=False):
        path = root / name
        if path.stat().st_mtime < began.timestamp():
            raise ValueError('Stale output: ' + name)
        payload = path.read_bytes()
        manifest['files'][name] = dict(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))
        return json.loads(payload) if as_json else list(csv.DictReader(payload.decode('utf-8-sig').splitlines()))

    summary = read('05_predict_upcoming_games_summary.json', True)
    if summary.get('target_date') != day or summary.get('status') != 'ok':
        raise ValueError('No complete POINT slate: explicit no-games review required')
    pairs = {}
    for market, filename, probability in [
        ('POINT', 'predictions_upcoming_point_enrichi_calibre_v2.csv', 'proba_point_1p_calibree'),
        *([('GOAL', 'predictions_upcoming_goal_research.csv', 'probability_goal')] if goal_required else []),
    ]:
        rows = read(filename)
        if not rows:
            raise ValueError('Empty predictions: ' + market)
        identities = set()
        for row in rows:
            identity = (row['id_match'], row['id_joueur'])
            if not all(identity) or identity in identities:
                raise ValueError('Missing/duplicate prediction identity')
            identities.add(identity)
            if row['date_match'][:10] != day:
                raise ValueError('Wrong prediction date')
            p = float(row[probability])
            if not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError('Invalid probability')
            if utc(row['start_time_utc']) <= clock:
                raise ValueError('Started game in production delivery')
            if market == 'GOAL':
                if row['model_status'] != 'research_only_operational':
                    raise ValueError('Unexpected GOAL approval status')
                if not began <= utc(row['generated_at_utc']) <= clock:
                    raise ValueError('Stale GOAL generation')
        pairs[market] = identities
        manifest['markets'][market] = dict(rows=len(rows), games=len({x[0] for x in identities}))
    if goal_required and pairs['POINT'] != pairs['GOAL']:
        raise ValueError('POINT/GOAL universes differ')
    if summary['future_players_rows'] != len(pairs['POINT']):
        raise ValueError('POINT summary count mismatch')
    bets = read('07_daily_bets.csv')
    for row in bets:
        if row['run_date'] != day or (row['id_match'], row['id_joueur']) not in pairs['POINT']:
            raise ValueError('Bet outside verified slate')
        price = float(row['odds_decimal'])
        if not math.isfinite(price) or price < 1.35 or row['bookmaker'].lower() != 'unibet':
            raise ValueError('Invalid Unibet price')
        if utc(row['captured_at']) >= utc(row['start_time_utc']):
            raise ValueError('Quote not pregame')
    manifest['recommended_point_bets'] = len(bets)
    manifest['status'] = 'verified'
    return manifest


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('outputs'))
    p.add_argument('--date', required=True)
    p.add_argument('--started', required=True)
    p.add_argument('--goal-required', choices=['true', 'false'], default='true')
    a = p.parse_args()
    result = validate(a.root, a.date, a.started, a.goal_required == 'true')
    (a.root / 'production_verified.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
