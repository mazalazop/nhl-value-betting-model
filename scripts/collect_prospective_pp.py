"""Public live NHL/PP capture. No training, bets, Sheets or historical backfill.

Optional prediction CSVs are joined to the live NHL slate; probabilities are never
recomputed. Missing GOAL input remains explicitly unavailable, never a proxy POINT.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import requests
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'model'))
from henachel.prospective_pp import Archive, collect, overlay, snapshots, TEAMS, attach_point_odds
from henachel.matching import name, utc
from henachel.prospective_http import get_json


def live_players(session, date, cache):
    response = get_json(session, f'https://api-web.nhle.com/v1/schedule/{date}', cache, 60)
    now = pd.Timestamp.now(tz='UTC'); games = []
    for day in response['document']['gameWeek']:
        if day['date'] != date: continue
        for g in day['games']:
            if g['gameState'] in {'FUT', 'PRE'} and g.get('gameScheduleState') == 'OK' and utc(g['startTimeUTC']) > now:
                games.append(g)
    if len({g['id'] for g in games}) != len(games): raise ValueError('Duplicate NHL schedule game')
    teams = {g[side]['abbrev'] for g in games for side in ['homeTeam', 'awayTeam']}
    if teams-TEAMS: raise ValueError('Unknown live NHL team')
    roster = []
    for team in sorted(teams):
        response = get_json(session, f'https://api-web.nhle.com/v1/roster/{team}/current', cache, 21600)
        observed = response['observed_at']
        document = response['document']
        for group in ['forwards', 'defensemen']:
            for p in document[group]:
                roster.append(dict(team=team, player_id=int(p['id']),
                                   player_name=p['firstName']['default']+' '+p['lastName']['default'], observed_at=observed))
    if len({r['player_id'] for r in roster}) != len(roster): raise ValueError('Ambiguous NHL roster membership')
    players = []
    for g in games:
        for p in roster:
            if p['team'] in {g['homeTeam']['abbrev'], g['awayTeam']['abbrev']}:
                players.append(dict(p, game_id=g['id'], game_date=date, event_start_utc=g['startTimeUTC']))
    return players, roster, games


def read_predictions(path, target, players, captured):
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_csv(path)
    probability = 'proba_point_1p_calibree' if target == 'POINT' else 'proba_goal_1p_calibree'
    required = ['id_match', 'id_joueur', 'team_player_match', 'nom', probability]
    if set(required)-set(frame): raise ValueError(f'{target}: incomplete prediction columns')
    lookup = {(p['game_id'], p['player_id']): p for p in players}; rows = []
    for x in frame.to_dict('records'):
        p = lookup.get((x['id_match'], x['id_joueur']))
        if p is None or p['team'] != x['team_player_match'] or name(p['player_name']) != name(x['nom']):
            raise ValueError(f'{target}: prediction outside verified upcoming roster/slate')
        row = dict(p, probability=x[probability], target=target, collected_at_utc=captured,
                   prediction_file_sha256=source_hash,
                   production_approved=(target == 'POINT'),
                   model_provenance=x.get('model_sha256'),
                   model_status='production_output' if target == 'POINT' else x.get('model_status','research_only'))
        # Preserve optional pregame context without inventing values.
        for key in ['toi_moy_5', 'toi_moy_10', 'pp_moy_5', 'pp_moy_10', 'training_cutoff', 'config_sha', 'model_name', 'calibration_method', 'generated_at_utc', 'algorithm_id', 'algorithm_manifest_sha256', 'frozen_probability', 'player_smoothed_probability']:
            row[key] = None if pd.isna(x.get(key)) else x[key]
        rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--date', default=str(pd.Timestamp.now(tz='America/New_York').date()))
    p.add_argument('--archive', type=Path, default=Path('outputs/prospective/archive.sqlite3'))
    p.add_argument('--output-dir', type=Path, default=Path('outputs/prospective/runs'))
    p.add_argument('--cache', type=Path, default=Path('outputs/prospective/nhl_cache'))
    p.add_argument('--point', type=Path); p.add_argument('--goal', type=Path)
    p.add_argument('--matched-odds', type=Path)
    p.add_argument('--goal-odds', type=Path, help='Strict normalized GOAL source; revalidated at capture')
    p.add_argument('--pp1-only', action='store_true')
    a = p.parse_args()
    if str(pd.Timestamp(a.date).date()) != a.date: raise ValueError('Expected NHL date YYYY-MM-DD')
    if pd.Timestamp(a.date).date() < pd.Timestamp.now(tz='America/New_York').date():
        raise ValueError('No retrospective capture allowed')
    session = requests.Session()
    players, roster, games = live_players(session, a.date, a.cache)
    evidence = collect(session) if players else dict(collected_at_utc=datetime.now(timezone.utc).isoformat(), observations=[], status='no_games')
    roles = snapshots(players, roster, evidence)
    captured = datetime.now(timezone.utc).isoformat()
    predictions = []
    for target, path in [('POINT', a.point), ('GOAL', a.goal)]:
        if path: predictions.extend(read_predictions(path, target, players, captured))
    odds = pd.read_csv(a.matched_odds, dtype={'event_id':str, 'bookmaker':str}).to_dict('records') if a.matched_odds else []
    predictions, rejected_odds = attach_point_odds(predictions, odds)
    if a.goal_odds and a.goal:
        from henachel.goal_odds import match_goal_odds
        matched,goal_rejected=match_goal_odds(pd.read_csv(a.goal),json.loads(a.goal_odds.read_text())['rows'])
        rejected_odds.extend(dict(target='GOAL',**r) for r in goal_rejected)
        lookup={(int(r['game_id']),int(r['player_id'])):r for r in matched.to_dict('records')}
        for row in predictions:
            odds_row=lookup.get((row['game_id'],row['player_id'])) if row['target']=='GOAL' else None
            if odds_row is not None:
                observation={k:odds_row[k] for k in ['bookmaker','event_id','collected_at_utc','odds_decimal','implied_probability','edge','expected_value','source_market_id','source_outcome_id','goal_value_rank']}
                observation.update(market='player_goals',stat='goals',threshold=1,outcome_key='1_plus')
                row['bookmaker_observations'].append(observation)
    raw, operational = overlay(predictions, roles, a.pp1_only)
    run_id = hashlib.sha256(json.dumps(dict(evidence=evidence, roles=roles, predictions=raw), sort_keys=True).encode()).hexdigest()
    archive = Archive(a.archive)
    try:
        archive.append(run_id+':pp', 'pregame', roles)
        archive.append(run_id+':predictions', 'pregame', raw)
    finally: archive.close()
    marker = a.archive.parent/'prospective_enabled.json'
    if not marker.exists():
        with marker.open('x') as f: json.dump(dict(schema_version=1, archive_filename=a.archive.name), f)
    report = dict(run_id=run_id, date=a.date, games=len(games), players=len(players),
                  source_status=evidence['status'], source_observations=len(evidence['observations']),
                  units=dict(Counter(r['pp_unit'] for r in roles)),
                  reasons=dict(Counter(r['evidence_status'] for r in roles)),
                  point_predictions=sum(r['target']=='POINT' for r in raw),
                  goal_predictions=sum(r['target']=='GOAL' for r in raw),
                  status='no_games' if not games else 'captured' if {r['target'] for r in raw} == {'POINT', 'GOAL'} else 'partial_predictions_unavailable',
                  goal_production_approved=False, pp_filter_validated=False,
                  bookmaker_observations=sum(len(r['bookmaker_observations']) for r in raw),
                  rejected_odds=rejected_odds)
    dest = a.output_dir/run_id; dest.mkdir(parents=True, exist_ok=True)
    for file, data in [('report.json', report), ('source_evidence.json', evidence), ('raw_ranking.json', raw), ('operational_ranking.json', operational)]:
        path = dest/file
        payload = json.dumps(data, indent=2, sort_keys=True, allow_nan=False)
        if path.exists():
            if path.read_text() != payload: raise ValueError('Immutable output conflict')
        else:
            with path.open('x') as f: f.write(payload)
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__': main()
