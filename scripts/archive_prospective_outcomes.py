"""Append observed NHL final results for previously archived players; never settle bets."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'model'))
from henachel.data import game_state
from henachel.prospective_pp import Archive


def record(archive, stats, matches, ledger=None, now=None):
    clock = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
    if stats.duplicated(['id_match','id_joueur']).any() or matches.id_match.duplicated().any():
        raise ValueError('Ambiguous final data')
    expected = {}; previous = {}
    for batch_id, kind, payload in archive.db.execute('SELECT batch_id,kind,payload FROM batches ORDER BY rowid'):
        for row in json.loads(payload):
            key = (row['game_id'], row['player_id'])
            if kind == 'pregame': expected[key] = row
            elif kind == 'outcome': previous[key] = (batch_id, row)
    games = matches.set_index('id_match'); results = stats.set_index(['id_match','id_joueur'])
    bets = {}
    if ledger is not None:
        if ledger.bet_id.duplicated().any(): raise ValueError('Ambiguous canonical bet ledger')
        for b in ledger.to_dict('records'):
            key = (b['id_match'], b['id_joueur'])
            observed = {k: None if pd.isna(b.get(k)) else b.get(k) for k in ['bet_id','bet_status','result','actual_stat_value','settled_at']}
            bets.setdefault(key, []).append(observed)
    changes = []; unresolved = []; predecessors = []
    for key, pre in sorted(expected.items()):
        if key[0] not in games.index:
            unresolved.append(dict(game_id=key[0], player_id=key[1], reason='game_unknown')); continue
        game = games.loc[key[0]]
        state = game_state(game.get('status'), game.get('schedule_state'))
        if state != 'final' or key not in results.index:
            unresolved.append(dict(game_id=key[0], player_id=key[1], reason=state if state != 'final' else 'missing_player_not_zero')); continue
        stats_row = results.loc[key]
        try:
            points, goals, assists = [float(stats_row.get(k, float('nan'))) for k in ['points','buts','passes']]
            if not all(math.isfinite(v) and v >= 0 and v.is_integer() for v in [points,goals,assists]) or points != goals+assists:
                raise ValueError('Inconsistent stats')
        except (TypeError, ValueError):
            unresolved.append(dict(game_id=key[0], player_id=key[1], reason='unknown_or_inconsistent_stats')); continue
        state_fields = dict(actual_points=int(points), actual_goals=int(goals), game_status='FINAL',
                            bet_observations=sorted(bets.get(key, []), key=lambda x: str(x['bet_id'])))
        old = previous.get(key)
        if old and all(old[1].get(k) == v for k,v in state_fields.items()): continue
        row = {k: pre[k] for k in ['game_id','player_id','game_date','event_start_utc','team','player_name']}
        row.update(state_fields, collected_at_utc=clock.isoformat(), source='NHL refreshed final data',
                   evidence_status='final_stats_observed', previous_outcome_batch=old[0] if old else None)
        changes.append(row); predecessors.append(old[0] if old else None)
    if changes:
        identity = dict(predecessors=predecessors, states=[{k:v for k,v in r.items() if k!='collected_at_utc'} for r in changes])
        batch_id = 'outcomes:'+hashlib.sha256(json.dumps(identity, sort_keys=True, allow_nan=False).encode()).hexdigest()
        archive.append(batch_id, 'outcome', changes, now=clock)
    return dict(changes=len(changes), unresolved=unresolved)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, default=Path('outputs/history/prospective.sqlite3'))
    p.add_argument('--stats', type=Path, default=Path('data/raw/stats.csv'))
    p.add_argument('--matches', type=Path, default=Path('data/raw/matchs.csv'))
    p.add_argument('--ledger', type=Path, default=Path('outputs/history/master_daily_bets_history.csv'))
    a = p.parse_args()
    if not a.archive.exists(): print(json.dumps(dict(status='no_prospective_history'))); return
    archive = Archive(a.archive)
    try:
        result = record(archive, pd.read_csv(a.stats), pd.read_csv(a.matches), pd.read_csv(a.ledger) if a.ledger.exists() else None)
    finally: archive.close()
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__': main()
