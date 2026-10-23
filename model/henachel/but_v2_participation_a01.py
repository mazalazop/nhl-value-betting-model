"""Read-only, exploratory A01 audit. Never imported by the frozen study/workflows."""
import hashlib
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

KNOWN = 'PARTICIPANT_RESULT_KNOWN'
ABSENT = 'NONPARTICIPATION_CORROBORATED'
MISSING = 'PARTICIPANT_RESULT_MISSING'
UNKNOWN = 'PARTICIPATION_UNKNOWN'
PENDING = 'GAME_PENDING_OR_POSTPONED'
FINAL = {'OFF', 'FINAL'}
MATURITY_HOURS = 12
MAX_SOURCE_SKEW_MINUTES = 15
MIN_INFORMATION_COVERAGE = .98
MIN_PARTICIPANT_RESULT_AVAILABILITY = .98


def utc(value):
    result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    if result.tzinfo is None:
        raise ValueError('Timezone required')
    return result.astimezone(timezone.utc)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def archive_evidence(root, envelope):
    """Content addressed, append only; source and observation time are part of hash."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    key = digest(envelope)
    path = root / (key + '.json')
    payload = json.dumps(envelope, sort_keys=True, allow_nan=False).encode()
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError('Evidence collision')
    else:
        with path.open('xb') as stream:
            stream.write(payload)
    return key


def read_evidence(path):
    path = Path(path)
    value = json.loads(path.read_text())
    if digest(value) != path.stem:
        raise ValueError('Evidence checksum mismatch')
    return value


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def toi(value):
    if value is None:
        return None
    try:
        minutes, seconds = str(value).split(':')
        if int(minutes) < 0 or not 0 <= int(seconds) < 60:
            return None
        return int(minutes) * 60 + int(seconds)
    except (ValueError, TypeError):
        return None


def goals(value):
    return number(value) and value >= 0 and int(value) == value


def source_document(envelope, gid, endpoint, start, as_of):
    expected = f'https://api-web.nhle.com/v1/gamecenter/{gid}/{endpoint}'
    if envelope.get('source') != expected:
        raise ValueError('Untrusted official source')
    observed = utc(envelope['observed_at_utc'])
    if observed > as_of:
        raise ValueError('Future evidence')
    document = envelope['document']
    if document.get('id') != gid:
        raise ValueError('Game identity mismatch')
    if document.get('gameState') in FINAL and observed < start:
        raise ValueError('Final observed before game')
    return document, observed


def classify(row, evidence, as_of, closure=False):
    """All ambiguity is explicit. Incomplete responses never establish absence."""
    as_of = utc(as_of)
    start = utc(row['start_time_utc'])
    gid, pid = int(row['id_match']), int(row['id_joueur'])
    result = dict(id_match=gid, id_joueur=pid, category=UNKNOWN, reason='NO_OFFICIAL_EVIDENCE',
                  goals=None, evaluable=bool(closure or as_of >= start + timedelta(hours=MATURITY_HOURS)),
                  evidence_sha256=digest(evidence), as_of_utc=as_of.isoformat())
    if as_of < start and not closure:
        result.update(category=PENDING, reason='BEFORE_SCHEDULED_START')
        return result
    docs = {}
    times = []
    try:
        for endpoint in ['boxscore', 'play-by-play']:
            if evidence.get(endpoint):
                docs[endpoint], observed = source_document(evidence[endpoint], gid, endpoint, start, as_of)
                times.append(observed)
    except (ValueError, KeyError, TypeError) as exc:
        result['reason'] = 'INVALID_EVIDENCE: ' + str(exc)
        return result
    box, play = docs.get('boxscore', {}), docs.get('play-by-play', {})
    if any(d.get('gameState') in FINAL for d in docs.values()):
        result['evaluable'] = True
    if not result['evaluable']:
        result.update(category=PENDING, reason='AWAITING_FINAL_OR_MATURITY')
        return result
    if not box or not play:
        result['reason'] = 'MISSING_OFFICIAL_ENDPOINT'
        return result
    if max(times) - min(times) > timedelta(minutes=MAX_SOURCE_SKEW_MINUTES):
        result['reason'] = 'UNSYNCHRONIZED_OFFICIAL_SOURCES'
        return result
    final = [d.get('gameState') in FINAL for d in [box, play]]
    if any(final) and not all(final):
        result['reason'] = 'CONTRADICTORY_GAME_STATES'
        return result
    if not all(final):
        result.update(category=PENDING, reason='POSTPONED' if any(
            d.get('gameScheduleState') in {'PPD', 'SUSP'} for d in [box, play]) else 'NOT_OFFICIAL_FINAL')
        return result
    try:
        teams = {box[s]['id']: box[s]['abbrev'] for s in ['homeTeam', 'awayTeam']}
        if len(teams) != 2 or row['team_player_match'] not in teams.values():
            raise ValueError('Prediction team mismatch')
        if any(play[s]['id'] != box[s]['id'] or play[s]['abbrev'] != box[s]['abbrev']
               for s in ['homeTeam', 'awayTeam']):
            raise ValueError('Official team mismatch')
        roster = play.get('rosterSpots', [])
        roster_ids = [p['playerId'] for p in roster]
        if len(roster_ids) != len(set(roster_ids)):
            raise ValueError('Duplicate official roster identity')
        if any(p['teamId'] not in teams for p in roster):
            raise ValueError('Unknown roster team')
        plays = play.get('plays', [])
        activity_keys = {'scoringPlayerId', 'shootingPlayerId', 'hittingPlayerId',
                         'hitteePlayerId', 'blockingPlayerId', 'winningPlayerId',
                         'losingPlayerId', 'committedByPlayerId', 'drawnByPlayerId',
                         'assist1PlayerId', 'assist2PlayerId'}
        activity = {v for event in plays for k, v in event.get('details', {}).items()
                    if k in activity_keys and isinstance(v, int)}
        by_id, complete = {}, bool(roster) and any(e.get('typeDescKey') == 'game-end' for e in plays)
        for side in ['homeTeam', 'awayTeam']:
            team = box[side]
            stats = box.get('playerByGameStats', {}).get(side, {})
            skaters = stats.get('forwards', []) + stats.get('defense', [])
            keepers = stats.get('goalies', [])
            ids = [p['playerId'] for p in skaters + keepers]
            if len(ids) != len(set(ids)) or set(ids) & set(by_id):
                raise ValueError('Duplicate official box identity')
            for player in skaters + keepers:
                by_id[player['playerId']] = (player, team['abbrev'], player in skaters)
            roster_sk = {p['playerId'] for p in roster if p['teamId'] == team['id'] and p.get('positionCode') in {'C', 'L', 'R', 'D'}}
            roster_g = {p['playerId'] for p in roster if p['teamId'] == team['id'] and p.get('positionCode') == 'G'}
            # Conservative certification: unusual short rosters require review, never infer absence.
            complete &= (18 <= len(skaters) <= 20 and 1 <= len(keepers) <= 3
                         and roster_sk == {p['playerId'] for p in skaters}
                         and roster_g == {p['playerId'] for p in keepers}
                         and all(toi(p.get('toi')) is not None and goals(p.get('goals')) for p in skaters)
                         and all(p.get('positionCode') in {'C', 'L', 'R', 'D', 'G'} for p in roster))
        record = by_id.get(pid)
        roster_player = next((p for p in roster if p['playerId'] == pid), None)
        if record:
            player, team, skater = record
            if team != row['team_player_match'] or not skater:
                raise ValueError('Player team/position mismatch')
            if (roster_player is None or teams[roster_player['teamId']] != team
                    or roster_player.get('positionCode') not in {'C', 'L', 'R', 'D'}):
                raise ValueError('Player absent from conflicting official roster')
            seconds = toi(player.get('toi'))
            if seconds is not None and seconds > 0:
                if goals(player.get('goals')):
                    result.update(category=KNOWN, reason='FINAL_POSITIVE_TOI_AND_GOALS', goals=int(player['goals']))
                else:
                    result.update(category=MISSING, reason='POSITIVE_TOI_GOALS_MISSING')
            else:
                result['reason'] = 'ZERO_TOI_NOT_PROOF_OF_ABSENCE' if seconds == 0 else 'TOI_UNKNOWN'
        elif roster_player:
            result['reason'] = 'ROSTER_PRESENT_BOX_MISSING'
        elif pid in activity:
            result['reason'] = 'PLAY_ACTIVITY_CONTRADICTS_MISSING_ROSTER'
        elif complete:
            result.update(category=ABSENT, reason='ABSENT_FROM_BOTH_COMPLETE_FINAL_SOURCES')
        else:
            result['reason'] = 'INCOMPLETE_OFFICIAL_ROSTERS'
    except (KeyError, ValueError, TypeError) as exc:
        result.update(category=UNKNOWN, reason='CONTRADICTORY_OR_INVALID_DATA: ' + str(exc), goals=None)
    return result


def audit(rows, current, as_of, previous=None, closure=False):
    """Snapshots keyed by game ID. Corrections retained and held for review, not applied."""
    keys = [(int(r['id_match']), int(r['id_joueur'])) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate captured player/game')
    output = []
    for row in rows:
        gid = str(int(row['id_match']))
        evidence = current.get(gid, {})
        item = classify(row, evidence, as_of, closure)
        if previous and gid in previous:
            old = classify(row, previous[gid], as_of, closure)
            older = any(evidence.get(e) and previous[gid].get(e) and
                        utc(evidence[e]['observed_at_utc']) < utc(previous[gid][e]['observed_at_utc'])
                        for e in ['boxscore', 'play-by-play'])
            if older:
                item.update(category=UNKNOWN, reason='EVIDENCE_TIME_ROLLBACK', goals=None,
                            previous_evidence_sha256=old['evidence_sha256'])
                output.append(item)
                continue
            # Only previously evaluable final evidence can be an official correction.
            old_final = all(previous[gid].get(e, {}).get('document', {}).get('gameState') in FINAL
                            for e in ['boxscore', 'play-by-play'])
            changed = (old['category'], old['goals']) != (item['category'], item['goals'])
            if old_final and changed:
                item.update(category=UNKNOWN, reason='OFFICIAL_REVISION_REQUIRES_REVIEW', goals=None,
                            previous_evidence_sha256=old['evidence_sha256'],
                            previous_classification=old, proposed_classification=dict(item))
        output.append(item)
    return output


def summarize(classifications, original_settled, original_n):
    """Original figures supplied unchanged from the frozen report, never recomputed as A01."""
    if original_n != len(classifications) or not 0 <= original_settled <= original_n:
        raise ValueError('Original report/cohort mismatch')
    keys = [(r['id_match'], r['id_joueur']) for r in classifications]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate audit identity')
    eligible = [r for r in classifications if r['evaluable']]
    counts = {k: sum(r['category'] == k for r in eligible) for k in [KNOWN, ABSENT, MISSING, UNKNOWN, PENDING]}
    n, s, a = len(eligible), counts[KNOWN], counts[ABSENT]
    p = s + counts[MISSING]
    def ratio(x, total):
        return x / total if total else None
    info, availability = ratio(s+a, n), ratio(s, p)
    return dict(mode='EXPLORATORY_A01_NOT_ACTIVATED', original_coverage=ratio(original_settled, original_n),
                original_n=original_n, original_settled=original_settled, evaluable_n=n,
                awaiting_maturity=len(classifications)-n, counts=counts, nonparticipant_share=ratio(a,n),
                genuinely_unknown_share=ratio(counts[MISSING]+counts[UNKNOWN],n),
                pending_evaluable_share=ratio(counts[PENDING],n), unresolved_information_share=ratio(n-s-a,n),
                information_coverage=info, identified_participants=p, participant_result_availability=availability,
                information_gate_98=info is not None and info >= MIN_INFORMATION_COVERAGE,
                participant_result_gate_98=availability is not None and availability >= MIN_PARTICIPANT_RESULT_AVAILABILITY,
                promotion_authorized=False, confirmatory_assessment_authorized=False)


def validate_preparation_config(config):
    expected = dict(enabled=False, t0_utc=None, status='DRAFT_NOT_ACTIVATED',
                    min_information_coverage=MIN_INFORMATION_COVERAGE,
                    min_participant_result_availability=MIN_PARTICIPANT_RESULT_AVAILABILITY,
                    maturity_hours=MATURITY_HOURS, max_source_skew_minutes=MAX_SOURCE_SKEW_MINUTES,
                    promotion_allowed=False, final_closure_include_all_captured=True)
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError('Preparation configuration changed or activation attempted')
