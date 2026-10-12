"""Research-only, resumable J-1 standings; never writes production files."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
import pandas as pd
from collect_research_history import Cache
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'model'))
spec = importlib.util.spec_from_file_location('structural_standings_source', ROOT / 'model/00c_refresh_team_standings.py')
source = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = source
spec.loader.exec_module(source)


def flatten(payload, date, season):
    records = payload.get('standings')
    if not isinstance(records, list):
        raise ValueError('Missing standings list')
    if not records:
        return []
    rows = [source.flatten_team_record(r, date) for r in records]
    if any(pd.Timestamp(r['api_date']) != pd.Timestamp(date) or r['season_id'] != season for r in rows):
        raise ValueError('Wrong standings date/season')
    teams = [r['team_abbrev'] for r in rows]
    if len(teams) != 32 or len(set(teams)) != 32:
        raise ValueError('Incomplete/ambiguous standings')
    return rows


def main():
    p = argparse.ArgumentParser(); p.add_argument('--history', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    dest = a.output / 'standings.csv.gz'
    if dest.exists():
        raise FileExistsError('Existing completed collection')
    history = pd.read_csv(a.history, usecols=['date_match', 'season_source']).drop_duplicates()
    cache = Cache(a.output / 'cache'); rows = []; dates = []
    for n, r in enumerate(history.sort_values('date_match').itertuples(index=False)):
        date = (pd.Timestamp(r.date_match) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        payload = cache.get(f'https://api-web.nhle.com/v1/standings/{date}')
        day = flatten(payload, date, int(r.season_source)); rows.extend(day)
        dates.append(dict(date=date, rows=len(day)))
        if n % 20 == 0:
            print(n, date, len(day), flush=True)
    frame = pd.DataFrame(rows); frame.to_csv(dest, index=False)
    report = dict(dates=dates, rows=len(frame), sha256=hashlib.sha256(dest.read_bytes()).hexdigest(), http_calls=cache.calls, cache_hits=cache.hits)
    (a.output / 'coverage.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
