"""Read-only bounded historical roster coverage and standings contract probes.

Season rosters are retrospective membership, NOT pregame lineup snapshots.
The immutable report contains counts/quality metadata, not player-level records.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import requests

TEAMS = 'ANA ARI BOS BUF CAR CBJ CGY CHI COL DAL DET EDM FLA LAK MIN MTL NJD NSH NYI NYR OTT PHI PIT SEA SJS STL TBL TOR VAN VGK WPG WSH'.split()
SEASONS = (20222023, 20232024)
DATES = {20222023: ['2022-10-15', '2023-02-01', '2023-04-14'],
         20232024: ['2023-10-15', '2024-02-01', '2024-04-18']}
ROOT = 'https://api-web.nhle.com/v1'


def roster_contract(payload):
    categories = ('forwards', 'defensemen', 'goalies')
    if any(not isinstance(payload.get(k), list) for k in categories):
        raise ValueError('Missing roster position lists')
    rows = [r for k in categories for r in payload[k]]
    ids = [r.get('id') for r in rows]
    invalid = sum(not isinstance(i, int) or isinstance(i, bool) or i <= 0 for i in ids)
    missing_positions = sum(not r.get('positionCode') for r in rows)
    missing_names = sum(not (r.get('firstName', {}).get('default') and r.get('lastName', {}).get('default')) for r in rows)
    duplicates = len(ids) - len(set(ids))
    return dict(contract_ok=bool(rows) and not any([invalid, missing_positions, missing_names, duplicates]),
                players=len(rows), unique_player_ids=len(set(ids)), invalid_player_ids=invalid,
                duplicate_player_ids=duplicates, missing_position=missing_positions, missing_name=missing_names,
                position_counts={k: len(payload[k]) for k in categories}, fields=sorted(set().union(*(r.keys() for r in rows))) if rows else [],
                _ids=ids)


def standings_contract(payload, season, requested_date):
    rows = payload.get('standings')
    if not isinstance(rows, list):
        raise ValueError('Missing standings list')
    dates = sorted({r.get('date', '') for r in rows})
    seasons = sorted({r.get('seasonId', 0) for r in rows})
    teams = [r.get('teamAbbrev', {}).get('default') for r in rows]
    return dict(contract_ok=len(rows) == 32 and len(set(teams)) == 32 and set(teams) == set(TEAMS)
                and dates == [requested_date] and seasons == [season], teams=len(rows),
                unique_teams=len(set(teams)), api_dates=dates, seasons=seasons,
                missing_teams=sorted(set(TEAMS) - set(teams)),
                fields=sorted(rows[0]) if rows else [])


def request_probe(task):
    source, season, key = task
    url = f'{ROOT}/roster/{key}/{season}' if source == 'roster' else f'{ROOT}/standings/{key}'
    record = dict(source=source, season=season, team=key if source == 'roster' else None,
                  requested_date=key if source == 'standings' else None, url=url,
                  collected_at_utc=datetime.now(timezone.utc).isoformat(), http_requests=1)
    try:
        response = requests.get(url, timeout=(10, 20))
        record['http_status'] = response.status_code
        response.raise_for_status()
        payload = response.json()
        record.update(roster_contract(payload) if source == 'roster' else standings_contract(payload, season, key))
    except (requests.RequestException, ValueError, TypeError, KeyError) as error:
        record.update(contract_ok=False, error_type=type(error).__name__, error=str(error))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    # Validate one endpoint for each independent source before expanding that source.
    first_roster = request_probe(('roster', SEASONS[0], TEAMS[0]))
    first_standings = request_probe(('standings', SEASONS[0], DATES[SEASONS[0]][0]))
    records = [first_roster, first_standings]
    tasks = []
    if first_roster['contract_ok']:
        tasks += [('roster', s, t) for s in SEASONS for t in TEAMS if (s, t) != (SEASONS[0], TEAMS[0])]
    if first_standings['contract_ok']:
        tasks += [('standings', s, d) for s in SEASONS for d in DATES[s] if (s, d) != (SEASONS[0], DATES[SEASONS[0]][0])]
    with ThreadPoolExecutor(max_workers=4) as pool:
        records += list(pool.map(request_probe, tasks))
    by_season = {}
    for season in SEASONS:
        roster = [r for r in records if r['source'] == 'roster' and r['season'] == season]
        standings = [r for r in records if r['source'] == 'standings' and r['season'] == season]
        ids = [i for r in roster for i in r.get('_ids', [])]
        by_season[str(season)] = dict(roster_endpoints_checked=len(roster), roster_endpoints_expected=32,
            roster_contracts_ok=sum(r['contract_ok'] for r in roster), player_team_memberships=len(ids),
            unique_player_ids=len(set(ids)), duplicate_memberships_across_teams=len(ids) - len(set(ids)),
            membership_note='Cross-team memberships may reflect trades; no current or pregame team inferred.',
            standings_dates_checked=len(standings), standings_contracts_ok=sum(r['contract_ok'] for r in standings))
    for record in records:
        record.pop('_ids', None)
    report = dict(scope='All 32 season-roster endpoints when initial contract passes; three standings samples per season only.',
        temporal_limitation='Season rosters are retrospective, not dated pregame availability. Do not infer historical role, lineup or injuries.',
        standings_limitation='Sampled dates only; daily standings coverage not established. Pregame consumer must use strictly earlier verified api_date.',
        http_request_count=len(records), max_http_requests=70, retries=0, timeout_connect_seconds=10,
        timeout_read_seconds=20, concurrency=4, seasons=by_season, records=records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps({'http_requests': len(records), 'seasons': by_season}, indent=2))


if __name__ == '__main__':
    main()
