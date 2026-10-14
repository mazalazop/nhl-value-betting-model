"""Prospective evidence only. An observation without a source time is not a role."""
from datetime import datetime, timezone
from html.parser import HTMLParser
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import pandas as pd

from henachel.matching import name, utc

SOURCE_URL = 'https://www.rotowire.com/hockey/nhl-lineups.php'
TEAMS = frozenset('ANA BOS BUF CAR CBJ CGY CHI COL DAL DET EDM FLA LAK MIN MTL NJD NSH NYI NYR OTT PHI PIT SEA SJS STL TBL TOR UTA VAN VGK WPG WSH'.split())


class LineupParser(HTMLParser):
    """Read only explicit public RotoWire PP headings and full-name anchors.

    No HTTP Date / fetch time / goalie confirmation substitutes for a PP source time.
    The current public page has no demonstrated unit-update timestamp.
    """
    def __init__(self):
        super().__init__(); self.stack = []; self.rows = []; self.teams = {}; self.side = None
        self.unit = None; self.item = None; self.text = ''; self.abbr = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs); classes = set(a.get('class', '').split())
        if 'lineup' in classes:
            self.teams = {}; self.unit = None
        if 'lineup__team' in classes:
            self.abbr = 'home' if 'is-home' in classes else 'visit' if 'is-visit' in classes else None
        if 'lineup__abbr' in classes: self.text = ''
        if 'lineup__list' in classes:
            self.side = 'home' if 'is-home' in classes else 'visit' if 'is-visit' in classes else None
            self.unit = None
        if tag == 'li':
            self.item = classes; self.text = ''
            if 'lineup__title' in classes: self.unit = None
        if tag == 'a' and self.item and 'lineup__player' in self.item and self.unit and a.get('title'):
            team = self.teams.get(self.side)
            if team in TEAMS:
                self.rows.append(dict(team=team, player_name=a['title'], observed_unit=self.unit,
                                      source_timestamp=None, source='RotoWire', source_url_or_identifier=SOURCE_URL))
        if tag not in {'img', 'br', 'hr', 'input', 'meta', 'link', 'source', 'wbr'}:
            self.stack.append((tag, classes))

    def handle_data(self, data): self.text += data

    def handle_endtag(self, tag):
        if tag == 'div' and self.stack and 'lineup__abbr' in self.stack[-1][1]:
            self.teams[self.abbr] = self.text.strip()
        if tag == 'li':
            if self.item and 'lineup__title' in self.item:
                self.unit = {'POWER PLAY #1': 'PP1', 'POWER PLAY #2': 'PP2'}.get(self.text.strip())
            self.item = None
        if tag == 'ul': self.side = None; self.unit = None
        for i in range(len(self.stack)-1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]; break


def parse_rotowire(html):
    parser = LineupParser(); parser.feed(html)
    if not parser.rows: raise ValueError('No explicit public PP units parsed')
    return parser.rows


def collect(session):
    """One public request; never bypass 403/auth or fabricate a source timestamp."""
    try:
        response = session.get(SOURCE_URL, timeout=30)
        captured = datetime.now(timezone.utc).isoformat()
        response.raise_for_status()
        rows = parse_rotowire(response.text)
        return dict(collected_at_utc=captured, observations=rows,
                    document_sha256=hashlib.sha256(response.content).hexdigest(),
                    status='source_timestamp_unavailable', source_url=SOURCE_URL)
    except (ValueError, OSError) as exc:
        return dict(collected_at_utc=datetime.now(timezone.utc).isoformat(), observations=[],
                    status='source_unavailable', error_type=type(exc).__name__, source_url=SOURCE_URL)
    except Exception as exc:
        # Requests errors have their own hierarchy; do not mask programming defects.
        import requests
        if not isinstance(exc, requests.RequestException): raise
        return dict(collected_at_utc=datetime.now(timezone.utc).isoformat(), observations=[],
                    status='source_unavailable', error_type=type(exc).__name__, source_url=SOURCE_URL)


def snapshots(players, roster, evidence, now=None, max_age_hours=24):
    """Players carry verified schedule identity; roster is an actual current snapshot.

    Full normalized names only. No guesses from surnames or historical membership.
    Evidence times must precede both observation and puck drop.
    """
    clock = utc(now or datetime.now(timezone.utc)); captured = utc(evidence['collected_at_utc'])
    if captured > clock: raise ValueError('Future collection time')
    if clock - captured > pd.Timedelta(hours=max_age_hours):
        raise ValueError('Stale collection')
    output = []; seen = set()
    for p in players:
        key = (p['game_id'], p['player_id'])
        if key in seen: raise ValueError('Duplicate player/game')
        seen.add(key)
        start = utc(p['event_start_utc'])
        if clock >= start or captured >= start: raise ValueError('Already started; no pregame snapshot')
        if p['team'] not in TEAMS: raise ValueError('Unknown NHL team')
        row = {k: p[k] for k in ['game_id', 'game_date', 'event_start_utc', 'team', 'player_id', 'player_name']}
        row.update(collected_at_utc=captured.isoformat(), source_timestamp=None, pp_unit='UNKNOWN',
                   confidence='unknown', source='RotoWire', source_url_or_identifier=SOURCE_URL,
                   evidence_status='player_not_observed', document_sha256=evidence.get('document_sha256'))
        candidates = [x for x in evidence['observations'] if x['team'] == p['team'] and name(x['player_name']) == name(p['player_name'])]
        identities = [r for r in roster if r['team'] == p['team'] and name(r['player_name']) == name(p['player_name'])]
        valid_roster = (len(identities) == 1 and identities[0]['player_id'] == p['player_id']
                        and sum(r['player_id'] == p['player_id'] for r in roster) == 1)
        if valid_roster:
            try:
                observed = utc(identities[0]['observed_at'])
                valid_roster = observed <= captured and (captured-observed).total_seconds() <= 172800
            except (ValueError, TypeError): valid_roster = False
        if not evidence['observations']: row['evidence_status'] = evidence['status']
        elif len(candidates) > 1: row['evidence_status'] = 'ambiguous_source'
        elif not valid_roster: row['evidence_status'] = 'roster_identity_unverified'
        elif candidates:
            c = candidates[0]; row['observed_unit'] = c['observed_unit']
            row['source_timestamp'] = c.get('source_timestamp')
            row['source'] = c['source']; row['source_url_or_identifier'] = c['source_url_or_identifier']
            try:
                stamp = utc(c.get('source_timestamp'))
                if stamp > captured or (captured-stamp).total_seconds() > max_age_hours*3600:
                    raise ValueError('Stale/future source')
                if c['observed_unit'] not in {'PP1', 'PP2'}: raise ValueError('Unknown unit')
                row.update(pp_unit=c['observed_unit'], confidence='timestamped_source_observation', evidence_status='VERIFIED_FRESH')
            except (ValueError, TypeError): row['evidence_status'] = 'source_timestamp_unverified'
        output.append(row)
    return output


def overlay(predictions, pp_rows, pp1_only=False):
    """Return raw and operational rankings. Never change probabilities or imply value."""
    lookup = {(x['game_id'], x['player_id']): x for x in pp_rows}
    if len(lookup) != len(pp_rows): raise ValueError('Ambiguous PP snapshots')
    raw = []; keys = set()
    for p in predictions:
        key = (p['game_id'], p['player_id'], p['target'])
        if key in keys: raise ValueError('Duplicate prediction')
        keys.add(key)
        probability = float(p['probability'])
        if not math.isfinite(probability) or not 0 <= probability <= 1: raise ValueError('Invalid probability')
        role = lookup.get(key[:2], {})
        raw.append(dict(p, pp_unit=role.get('pp_unit', 'UNKNOWN'), pp_confidence=role.get('confidence', 'unknown'),
                        pp_collected_at_utc=role.get('collected_at_utc'), pp_source_timestamp=role.get('source_timestamp'),
                        evidence_status=role.get('evidence_status','UNKNOWN')))
    raw.sort(key=lambda x: (x['target'], -x['probability'], x['game_id'], x['player_id']))
    ranks = {}
    for row in raw:
        ranks[row['target']] = ranks.get(row['target'], 0)+1; row['model_rank'] = ranks[row['target']]
    operational = [dict(x) for x in raw if not pp1_only or (x['pp_unit'] == 'PP1' and x['evidence_status']=='VERIFIED_FRESH')]
    return raw, operational


class Archive:
    """Transactional append-only batches; exact retry allowed, revision forbidden.

    SQLite transactions survive partial writes. Export/backup only after close.
    Timestamps are integrity guards, not independently trusted timestamp certification.
    """
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30)
        self.db.executescript('''
          CREATE TABLE IF NOT EXISTS batches (batch_id TEXT PRIMARY KEY, kind TEXT NOT NULL, payload TEXT NOT NULL, archived_at_utc TEXT NOT NULL);
          CREATE TRIGGER IF NOT EXISTS no_update BEFORE UPDATE ON batches BEGIN SELECT RAISE(ABORT, 'append only'); END;
          CREATE TRIGGER IF NOT EXISTS no_delete BEFORE DELETE ON batches BEGIN SELECT RAISE(ABORT, 'append only'); END;
        ''')

    def close(self): self.db.close()

    def append(self, batch_id, kind, rows, now=None):
        if not batch_id or kind not in {'pregame', 'outcome'}: raise ValueError('Invalid batch identity/kind')
        payload = json.dumps(rows, sort_keys=True, separators=(',', ':'), allow_nan=False)
        clock = utc(now or datetime.now(timezone.utc))
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            previous = self.db.execute('SELECT kind,payload FROM batches WHERE batch_id=?', (batch_id,)).fetchone()
            if previous:
                if previous != (kind, payload): raise ValueError('Immutable batch conflict')
                return False
            seen = set()
            for row in rows:
                for key in ['game_id', 'player_id']:
                    if isinstance(row[key], bool) or int(row[key]) != row[key] or int(row[key]) <= 0: raise ValueError('Invalid NHL ID')
                key = (row['game_id'], row['player_id'], row.get('target', 'PP'))
                if key in seen: raise ValueError('Duplicate player/target in batch')
                seen.add(key)
                collected = utc(row['collected_at_utc']); start = utc(row['event_start_utc'])
                if collected > clock: raise ValueError('Future collection')
                if kind == 'pregame':
                    if clock >= start or collected >= start: raise ValueError('Archive must precede actual puck drop')
                    if any(k in row for k in ['actual_points', 'actual_goals', 'result', 'settled_at']): raise ValueError('Post-match data in pregame record')
                    if 'probability' in row:
                        p = float(row['probability'])
                        if not math.isfinite(p) or not 0 <= p <= 1: raise ValueError('Invalid probability')
                    if row.get('pp_unit', 'UNKNOWN') not in {'PP1', 'PP2', 'UNKNOWN'}: raise ValueError('Invalid PP unit')
                    if row.get('pp_unit') in {'PP1', 'PP2'}:
                        stamp = row.get('source_timestamp', row.get('pp_source_timestamp'))
                        if utc(stamp) > collected: raise ValueError('Unproven PP source time')
                else:
                    if row.get('game_status') not in {'OFF', 'FINAL'} or clock < start: raise ValueError('Unfinished outcome')
                    for field in ['actual_points', 'actual_goals']:
                        value = row[field]
                        if isinstance(value, bool) or not isinstance(value, int) or value < 0: raise ValueError('Unknown outcome is not zero')
                    if row['actual_goals'] > row['actual_points']: raise ValueError('Inconsistent outcome')
            self.db.execute('INSERT INTO batches VALUES (?,?,?,?)', (batch_id, kind, payload, clock.isoformat()))
        return True


def attach_point_odds(predictions, odds, now=None):
    """Preserve verified matching output as timed evidence, not as a betting policy."""
    from henachel.matching import value_metrics
    clock = utc(now or datetime.now(timezone.utc))
    output = [dict(p, bookmaker_observations=[]) for p in predictions]
    lookup = {(p['game_id'], p['player_id']): p for p in output if p['target'] == 'POINT'}
    rejected = []; seen = set()
    for o in odds:
        try:
            key = (o['id_match'], o['id_joueur']); p = lookup.get(key)
            if p is None: raise ValueError('No POINT prediction for odds')
            if o.get('market') != 'player_points' or o.get('stat') != 'points' or float(o.get('threshold', 0)) != 1:
                raise ValueError('Unsupported market')
            if o.get('outcome_key') not in {'1_plus', 'points_1_plus', '1+', 'over_0.5'}: raise ValueError('Unsupported outcome')
            if o.get('team') != p['team']: raise ValueError('Team mismatch')
            if utc(o['start_time_utc']) != utc(p['event_start_utc']): raise ValueError('Event time mismatch')
            captured = utc(o['captured_at'])
            if not 0 <= (clock-captured).total_seconds() <= 21600 or clock >= utc(p['event_start_utc']):
                raise ValueError('Stale/future/postgame odds')
            if any(not isinstance(o.get(k), str) or not o[k].strip() for k in ['event_id', 'bookmaker']):
                raise ValueError('Missing bookmaker identity')
            identity = key+(o['bookmaker'], o['event_id'])
            if identity in seen: raise ValueError('Duplicate odds')
            metrics = value_metrics(p['probability'], o['odds_decimal'], o.get('implied_probability'))
            observation = {k: o[k] for k in ['event_id', 'bookmaker', 'market', 'stat', 'threshold', 'outcome_key', 'captured_at']}
            observation.update(metrics)
            p['bookmaker_observations'].append(observation); seen.add(identity)
        except (ValueError, TypeError, KeyError) as exc:
            rejected.append(dict(reason=str(exc)))
    return output, rejected
