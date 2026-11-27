"""Recover missing NHL PP observations for the dated GOAL17 analysis."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import pandas as pd
import requests

RAW = Path("data/raw")
URL = "https://api.nhle.com/stats/rest/en/skater/powerplay"


def fetch(pair):
    params = {
        "isAggregate": "false",
        "isGame": "true",
        "limit": 100,
        "cayenneExp": f"gameId>={pair[0]} and gameId<={pair[-1]}",
    }
    for attempt in range(5):
        try:
            response = requests.get(URL, params=params, timeout=40)
            response.raise_for_status()
            result = response.json()
            if len(result["data"]) != result["total"]:
                raise ValueError(f"Incomplete PP response for {pair}")
            return [(r["gameId"], r["playerId"], r["ppTimeOnIce"]) for r in result["data"]]
        except (requests.RequestException, ValueError, KeyError):
            if attempt == 4:
                raise
            time.sleep(2**attempt)


def main():
    source = pd.read_csv(RAW / "base_match_fusionnee.csv", usecols=["id_match", "id_joueur", "date_match"])
    assert source.date_match.max() == "2026-11-25"
    games = sorted(source.id_match.unique())
    batches = []
    for game in games:
        if batches and len(batches[-1]) == 1 and game == batches[-1][0] + 1:
            batches[-1].append(game)
        else:
            batches.append([game])
    rows = []
    with ThreadPoolExecutor(max_workers=12) as executor:
        for batch in executor.map(fetch, batches):
            rows.extend(batch)
    official = pd.DataFrame(rows, columns=["id_match", "id_joueur", "ppTimeOnIce"])
    joined = source.merge(official, on=["id_match", "id_joueur"], how="left", validate="one_to_one")
    print("PP coverage:", joined.ppTimeOnIce.notna().mean(), flush=True)
    official.to_csv(RAW / "pp_stats_game.csv", index=False)


if __name__ == "__main__":
    main()
