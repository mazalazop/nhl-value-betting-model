#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model/00a_refresh_pp_stats.py

Objectif
--------
Tester proprement la récupération des stats PP joueur depuis la stack NHL Stats,
sans toucher au pipeline principal.

Principe
--------
- lit la plage de dates du projet depuis data/final/base_canonique_v2.csv
- fallback propre sur data/raw/matchs.csv si base_canonique_v2.csv n'existe pas
- borne la date de fin à today() pour éviter les requêtes inutiles dans le futur
- découpe la période en mois
- interroge l'endpoint stats/rest/en/skater/powerplay en niveau game
- concatène le brut
- sauvegarde :
    - data/raw/pp_stats_game_raw.json
    - data/raw/pp_stats_game.csv
    - outputs/00a_refresh_pp_stats_summary.json

Important
---------
- script expérimental
- ne modifie pas 00b / 01 / 02 / 03
- ne suppose pas à l'avance toutes les colonnes
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import quote

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_STATS_URL = "https://api.nhle.com/stats/rest/en/skater/powerplay"
REQUEST_TIMEOUT = 45
SLEEP_SECONDS = 0.25


@dataclass
class Config:
    start_date: date
    end_date: date
    game_type_id: int
    sleep_seconds: float
    force: bool


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def build_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry)

    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": "nhl-value-betting-model/00a_refresh_pp_stats",
            "Accept": "application/json",
        }
    )
    return session


def safe_json_get(session: requests.Session, url: str) -> Dict:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def month_chunks(start_date: date, end_date: date) -> List[Tuple[date, date]]:
    chunks: List[Tuple[date, date]] = []

    current = start_date.replace(day=1)
    while current <= end_date:
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)

        chunk_start = max(current, start_date)
        chunk_end = min(next_month - timedelta(days=1), end_date)
        chunks.append((chunk_start, chunk_end))
        current = next_month

    return chunks


def _extract_date_range_from_csv(csv_path: Path, date_candidates: List[str]) -> Tuple[date, date]:
    df = pd.read_csv(csv_path)

    date_col = next((c for c in date_candidates if c in df.columns), None)
    if date_col is None:
        raise ValueError(
            f"Aucune colonne de date exploitable trouvée dans {csv_path}. "
            f"Colonnes disponibles: {list(df.columns)}"
        )

    s = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if s.empty:
        raise ValueError(f"Impossible d'inférer des dates valides depuis {csv_path}")

    return s.min().date(), s.max().date()


def infer_date_range_from_project(repo_root: Path) -> Tuple[date, date]:
    """
    Priorité :
    1) data/final/base_canonique_v2.csv
    2) data/raw/matchs.csv

    Cela permet au script de fonctionner :
    - sur un projet déjà avancé
    - sur une reconstruction from scratch après 00_refresh_sources.py
    """
    base_path = repo_root / "data" / "final" / "base_canonique_v2.csv"
    fallback_path = repo_root / "data" / "raw" / "matchs.csv"

    date_candidates = ["date_match", "game_date", "date", "match_date"]

    if base_path.exists():
        return _extract_date_range_from_csv(base_path, date_candidates)

    if fallback_path.exists():
        return _extract_date_range_from_csv(fallback_path, date_candidates)

    raise FileNotFoundError(
        f"Fichier introuvable : {base_path} ; fallback introuvable aussi : {fallback_path}"
    )


def cap_end_date_to_today(start_date: date, end_date: date) -> Tuple[date, date, bool]:
    today = date.today()
    capped = end_date > today
    end_date_capped = min(end_date, today)

    if start_date > end_date_capped:
        raise ValueError(
            f"Plage de dates invalide après borne haute à today(): start_date={start_date} ; "
            f"end_date_capped={end_date_capped}"
        )

    return start_date, end_date_capped, capped


def build_cayenne_exp(start_date: date, end_date: date, game_type_id: int) -> str:
    return (
        f'gameDate<="{end_date.isoformat()} 23:59:59" '
        f'and gameDate>="{start_date.isoformat()}" '
        f"and gameTypeId={game_type_id}"
    )


def build_url(start_date: date, end_date: date, game_type_id: int) -> str:
    sort = json.dumps(
        [
            {"property": "ppTimeOnIce", "direction": "DESC"},
            {"property": "playerId", "direction": "ASC"},
        ],
        separators=(",", ":"),
    )
    cayenne_exp = build_cayenne_exp(start_date, end_date, game_type_id)

    query = (
        f"isAggregate=false"
        f"&isGame=true"
        f"&sort={quote(sort, safe='')}"
        f"&start=0"
        f"&limit=-1"
        f"&cayenneExp={quote(cayenne_exp, safe='')}"
    )
    return f"{BASE_STATS_URL}?{query}"


def fetch_one_chunk(
    session: requests.Session,
    start_date: date,
    end_date: date,
    game_type_id: int,
) -> List[Dict]:
    url = build_url(start_date, end_date, game_type_id)
    payload = safe_json_get(session, url)

    if not isinstance(payload, dict):
        raise ValueError("Réponse inattendue : payload non dict")

    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError(
            f"Réponse inattendue : clé 'data' absente ou non liste. "
            f"Clés payload: {list(payload.keys())}"
        )

    return data


def ensure_dirs(repo_root: Path) -> Tuple[Path, Path]:
    raw_dir = repo_root / "data" / "raw"
    outputs_dir = repo_root / "outputs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, outputs_dir


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rename_map = {}
    if "playerId" in out.columns:
        rename_map["playerId"] = "id_joueur"
    if "gameId" in out.columns:
        rename_map["gameId"] = "id_match"
    if "gameDate" in out.columns:
        rename_map["gameDate"] = "date_match"

    out = out.rename(columns=rename_map)

    if "date_match" in out.columns:
        out["date_match"] = pd.to_datetime(out["date_match"], errors="coerce").dt.date.astype("string")

    for col in ["ppTimeOnIce", "ppTimeOnIcePerGame", "gamesPlayed"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    front_cols = [
        c
        for c in [
            "id_joueur",
            "id_match",
            "date_match",
            "ppTimeOnIce",
            "ppTimeOnIcePerGame",
            "gamesPlayed",
        ]
        if c in out.columns
    ]
    other_cols = [c for c in out.columns if c not in front_cols]
    return out[front_cols + other_cols]


def build_summary(df: pd.DataFrame, config: Config, chunks: List[Tuple[date, date]], end_date_was_capped: bool) -> Dict:
    summary: Dict = {
        "status": "ok",
        "source": BASE_STATS_URL,
        "date_range": {
            "start_date": config.start_date.isoformat(),
            "end_date": config.end_date.isoformat(),
            "end_date_was_capped_to_today": end_date_was_capped,
            "today": date.today().isoformat(),
        },
        "game_type_id": config.game_type_id,
        "n_chunks": len(chunks),
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "checks": {},
    }

    for col in [
        "ppTimeOnIce",
        "ppTimeOnIcePerGame",
        "gamesPlayed",
        "id_joueur",
        "id_match",
        "date_match",
    ]:
        if col in df.columns:
            summary["checks"][col] = {
                "non_null": int(df[col].notna().sum()),
                "nunique": int(df[col].nunique(dropna=True)),
            }

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--game-type-id", type=int, default=2, help="2 = saison régulière NHL")
    parser.add_argument("--sleep-seconds", type=float, default=SLEEP_SECONDS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    raw_dir, outputs_dir = ensure_dirs(repo_root)

    if args.start_date and args.end_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        start_date, end_date = infer_date_range_from_project(repo_root)

    start_date, end_date, end_date_was_capped = cap_end_date_to_today(start_date, end_date)

    if start_date > end_date:
        raise ValueError("start_date > end_date")

    config = Config(
        start_date=start_date,
        end_date=end_date,
        game_type_id=args.game_type_id,
        sleep_seconds=args.sleep_seconds,
        force=args.force,
    )

    raw_json_path = raw_dir / "pp_stats_game_raw.json"
    csv_path = raw_dir / "pp_stats_game.csv"
    summary_path = outputs_dir / "00a_refresh_pp_stats_summary.json"

    print("=== 00a_refresh_pp_stats.py ===")
    print(f"Repo root     : {repo_root}")
    print(f"Date start    : {config.start_date}")
    print(f"Date end      : {config.end_date}")
    if end_date_was_capped:
        print(f"Date end raw  : {end_date}")
        print(f"Date end cap  : {config.end_date} (bornée à today())")
    print(f"Game type id  : {config.game_type_id}")
    print(f"Raw JSON path : {raw_json_path}")
    print(f"CSV path      : {csv_path}")
    print("")

    chunks = month_chunks(config.start_date, config.end_date)
    print(f"Nombre de chunks mensuels : {len(chunks)}")

    session = build_session()
    all_rows: List[Dict] = []

    for i, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        print(f"[{i}/{len(chunks)}] {chunk_start} -> {chunk_end}")
        rows = fetch_one_chunk(
            session=session,
            start_date=chunk_start,
            end_date=chunk_end,
            game_type_id=config.game_type_id,
        )
        print(f"  lignes récupérées : {len(rows)}")
        all_rows.extend(rows)
        time.sleep(config.sleep_seconds)

    if not all_rows:
        raise ValueError("Aucune ligne récupérée depuis l'endpoint PP stats.")

    raw_json_path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    df = pd.DataFrame(all_rows)
    df = normalize_dataframe(df)
    df.to_csv(csv_path, index=False)

    summary = build_summary(df, config, chunks, end_date_was_capped=end_date_was_capped)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("")
    print("=== CONTROLES ===")
    print(f"Lignes totales : {len(df)}")
    print(f"Colonnes       : {len(df.columns)}")
    print("Colonnes dispo :")
    print(list(df.columns))

    for col in ["ppTimeOnIce", "ppTimeOnIcePerGame", "gamesPlayed", "id_joueur", "id_match", "date_match"]:
        if col in df.columns:
            print("")
            print(f"{col}")
            print(f"  non-null : {int(df[col].notna().sum())}")
            print(f"  nunique  : {int(df[col].nunique(dropna=True))}")

    print("")
    print("=== APERCU ===")
    print(df.head(10).to_string(index=False))

    print("")
    print("Terminé.")
    print(f"[OK] {raw_json_path}")
    print(f"[OK] {csv_path}")
    print(f"[OK] {summary_path}")


if __name__ == "__main__":
    main()
