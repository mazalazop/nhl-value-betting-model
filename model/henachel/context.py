"""Versioned pregame context, separate from models, betting policy and settlement."""
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
import pandas as pd
from henachel.matching import utc
from henachel.prospective_pp import Archive, TEAMS

SCHEMA = 1
KINDS = {'availability', 'line', 'pp_unit', 'goalie', 'calendar', 'team_stats',
         'special_teams', 'player_usage', 'goalie_season', 'travel', 'dangerous_shots'}


def clock(): return datetime.now(timezone.utc).isoformat()
def encoded(value): return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
def digest(value): return hashlib.sha256(encoded(value).encode()).hexdigest()


def validate_observation(row, now):
    r = dict(row)
    r.pop('observation_id', None)
    for field in ['game_id', 'game_date', 'team', 'event_start_utc', 'collected_at_utc', 'source', 'kind']:
        if not r.get(field): raise ValueError('Missing context '+field)
    if r['team'] not in TEAMS or r['kind'] not in KINDS: raise ValueError('Unknown identity/category')
    for field in ['game_id', 'player_id']:
        value = r.get(field)
        if field == 'player_id' and value is None: continue
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0: raise ValueError('Invalid identity')
    collected, start = utc(r['collected_at_utc']), utc(r['event_start_utc'])
    if collected > utc(now) or collected >= start or utc(now) >= start:
        raise ValueError('Not pregame')
    r['schema_version'] = SCHEMA
    level = r.get('evidence_status', 'UNKNOWN')
    if level not in {'CONFIRMED', 'PROBABLE', 'UNKNOWN'}: raise ValueError('Invalid evidence')
    ttl = float(r.get('max_age_seconds', 7200))
    if not math.isfinite(ttl) or not 0 < ttl <= 86400: raise ValueError('Invalid freshness limit')
    r['max_age_seconds'] = ttl
    r['source_timestamp'] = r.get('source_timestamp')
    r['freshness_status'] = 'UNVERIFIED'
    r['source_age_seconds'] = None
    try:
        stamp = utc(r['source_timestamp'])
        age = (collected-stamp).total_seconds()
        r['source_age_seconds'] = age
        r['freshness_status'] = 'FRESH' if 0 <= age <= ttl and stamp < start else 'STALE_OR_INVALID'
    except (ValueError, TypeError): pass
    if r.get('value') is None or r['freshness_status'] != 'FRESH': level = 'UNKNOWN'
    # Statistical observations may use retrieval time as proof of observed FINAL,
    # never as a fabricated publication time. Every contributing game is explicit.
    if r.get('stat_window') is not None:
        members = r.get('games_used', [])
        if not members: level = 'UNKNOWN'
        for game in members:
            if game['game_id'] == r['game_id'] or game['status'] not in {'FINAL', 'OFF'}:
                raise ValueError('Target/non-final game in aggregate')
            if not utc(game['event_start_utc']) < utc(game['final_observed_at_utc']) <= collected:
                raise ValueError('Final status not observed before collection')
        through = r.get('data_through_utc')
        if members and (not through or utc(through) > collected or utc(through) != max(utc(g['final_observed_at_utc']) for g in members)):
            raise ValueError('Invalid aggregate cutoff')
    r['evidence_status'] = level
    r['observation_id'] = digest(r)
    return r


class ContextStore:
    """Extra immutable tables in the existing prospective SQLite file."""
    def __init__(self, path):
        self.archive = Archive(path)
        self.db = self.archive.db
        for table in ['context_observations', 'context_views', 'context_baselines']:
            game_path='$.model_baseline.game_id' if table=='context_baselines' else '$.game_id'
            self.db.executescript(f'''
                CREATE TABLE IF NOT EXISTS {table} (id TEXT PRIMARY KEY, payload TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS {table}_game ON {table}(json_extract(payload,'{game_path}'));
                CREATE TRIGGER IF NOT EXISTS {table}_no_update BEFORE UPDATE ON {table} BEGIN SELECT RAISE(ABORT,'append only'); END;
                CREATE TRIGGER IF NOT EXISTS {table}_no_delete BEFORE DELETE ON {table} BEGIN SELECT RAISE(ABORT,'append only'); END;
            ''')

    def close(self): self.archive.close()

    def rows(self, table, game_ids=None):
        if table not in {'context_observations', 'context_views', 'context_baselines'}: raise ValueError('Invalid table')
        query=f'SELECT payload FROM {table}'
        params=[]
        if game_ids is not None:
            params=sorted(game_ids)
            if not params:return []
            path='$.model_baseline.game_id' if table=='context_baselines' else '$.game_id'
            query+=f" WHERE json_extract(payload,'{path}') IN ({','.join('?' for _ in params)})"
        return [json.loads(r[0]) for r in self.db.execute(query+' ORDER BY rowid',params)]

    def seed_baselines(self, now):
        # Earliest archived prediction wins, never a later daily refit. Original
        # payload (including odds evidence) stays intact inside model_baseline.
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            batches = self.db.execute("SELECT batch_id,payload,archived_at_utc FROM batches WHERE kind='pregame' ORDER BY archived_at_utc,batch_id").fetchall()
            for batch, payload, archived in batches:
                groups = defaultdict(list)
                for row in json.loads(payload):
                    if row.get('target') in {'POINT', 'GOAL'}:
                        groups[(row['game_date'], row['target'])].append(row)
                for group in groups.values():
                    ordered = sorted(group, key=lambda r: (-r['probability'], r['game_id'], r['player_id']))
                    for rank, row in enumerate(ordered, 1):
                        if not utc(row['collected_at_utc']) <= utc(archived) < utc(row['event_start_utc']): continue
                        if utc(now) >= utc(row['event_start_utc']): continue
                        key = f"{row['game_id']}:{row['player_id']}:{row['target']}"
                        value = dict(baseline_id=key, prediction_batch_id=batch, model_baseline=row,
                                     model_rank=row.get('model_rank', rank),
                                     rank_definition=('archived model_rank unchanged' if 'model_rank' in row else
                                                      'legacy fallback: initial probability descending, ties game_id/player_id'), schema_version=SCHEMA)
                        self.db.execute('INSERT OR IGNORE INTO context_baselines VALUES (?,?)', (key, encoded(value)))

    def append_observations(self, rows, now):
        validated = [validate_observation(r, now) for r in rows]
        inserted = 0
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            for row in validated:
                inserted += self.db.execute('INSERT OR IGNORE INTO context_observations VALUES (?,?)',
                                            (row['observation_id'], encoded(row))).rowcount
        return inserted

    def update_views(self, now, game_ids=None):
        current = utc(now); changed = []
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            observations = self.rows('context_observations',game_ids)
            old = {r['baseline_id']: r for r in self.rows('context_views',game_ids)}
            for base in self.rows('context_baselines',game_ids):
                p = base['model_baseline']
                if current >= utc(p['event_start_utc']) or (game_ids is not None and p['game_id'] not in game_ids): continue
                applicable = [r for r in observations if r['game_id']==p['game_id'] and r['team']==p['team']
                              and r.get('player_id') in {None, p['player_id']} and utc(r['collected_at_utc']) <= current
                              and utc(r['event_start_utc']) == utc(p['event_start_utc'])]
                groups = defaultdict(list)
                for row in sorted(applicable, key=lambda r:(utc(r['collected_at_utc']),r['observation_id'])):
                    groups[(row['kind'], row.get('player_id'), row['source'])].append(row)
                latest = {}
                for key, group in groups.items():
                    valid = [r for r in group if r['evidence_status']!='UNKNOWN' and r['source_timestamp']
                             and 0 <= (current-utc(r['source_timestamp'])).total_seconds() <= r['max_age_seconds']]
                    chosen = valid or group
                    stamp = chosen[-1]['collected_at_utc']
                    selected = [r for r in chosen if r['collected_at_utc']==stamp]
                    # A failed/unknown refresh cannot silently retract still-fresh evidence.
                    if group[-1]['collected_at_utc'] != stamp:
                        selected += [r for r in group if r['collected_at_utc']==group[-1]['collected_at_utc']]
                    latest[key] = selected
                review = []
                for group in latest.values():
                    for row in group:
                        status = row['evidence_status']
                        age = (current-utc(row['source_timestamp'])).total_seconds() if row['freshness_status']=='FRESH' else None
                        if age is None or not 0 <= age <= row['max_age_seconds']: status='UNKNOWN'
                        review.append(dict(observation_id=row['observation_id'],kind=row['kind'],value=row.get('value'),
                                           evidence_status=status,source=row['source'],source_timestamp=row['source_timestamp']))
                review.sort(key=lambda r:r['observation_id'])
                avail = [r for r in review if r['kind']=='availability' and r['evidence_status']=='CONFIRMED']
                values = {encoded(r['value']) for r in avail}
                conflict = len(values)>1
                absent = not conflict and any(r['value'] in ['OUT', 'SCRATCH', 'NOT_PLAYING'] for r in avail)
                status = 'EXCLUDED_CONFIRMED_ABSENCE' if absent else 'ACTIVE_REVIEW_CONFLICT' if conflict else 'ACTIVE'
                reasons = ['confirmed absence; baseline retained'] if absent else ['contradictory confirmed evidence; no certain exclusion'] if conflict else ['baseline ordering retained; annotations only']
                previous = old.get(base['baseline_id'])
                signature = digest(dict(review=review,status=status,baseline=base['baseline_id']))
                if previous and previous['signature']==signature: continue
                view = dict(schema_version=SCHEMA,baseline_id=base['baseline_id'],model_baseline=base['model_baseline'],
                            prediction_batch_id=base['prediction_batch_id'],model_rank=base['model_rank'],
                            generated_at_utc=current.isoformat(),game_id=p['game_id'],player_id=p['player_id'],team=p['team'],
                            market=p['target'],event_start_utc=p['event_start_utc'],context_review=review,
                            operational_status=status,active=not absent,reasons=reasons,signature=signature,
                            previous_view_id=previous['view_id'] if previous else None)
                view['view_id']=digest(view)
                self.db.execute('INSERT INTO context_views VALUES (?,?)',(view['view_id'],encoded(view)))
                changed.append(view)
        return changed


def upcoming_baselines(store, now, window_hours=18):
    current=utc(now)
    query="""SELECT payload FROM context_baselines
             WHERE julianday(json_extract(payload,'$.model_baseline.event_start_utc')) > julianday(?)
             AND julianday(json_extract(payload,'$.model_baseline.event_start_utc')) <= julianday(?)"""
    return [json.loads(r[0]) for r in store.db.execute(query,(current.isoformat(),(current+pd.Timedelta(hours=window_hours)).isoformat()))]
