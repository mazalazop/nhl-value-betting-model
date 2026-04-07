#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model/00a_refresh_pp_stats.py

MODIFIÉ — mode incrémental.
Si data/raw/pp_stats_game.csv existe, on ne refetch que les mois
après la dernière date couverte. Flag --rebuild pour forcer un refresh complet.
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
    retry = Retry(total=5, read=5, connect=5, backoff_factor=1.0,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(["GET"]))
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "nhl-value-betting-model/00a_refresh_pp_stats",
                            "Accept": "application/json"})
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
        raise ValueError(f"Aucune colonne de date trouvée dans {csv_path}.")
    s = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if s.empty:
        raise ValueError(f"Impossible d'inférer des dates depuis {csv_path}")
    return s.min().date(), s.max().date()


def infer_date_range_from_project(repo_root: Path) -> Tuple[date, date]:
    base_path = repo_root / "data" / "final" / "base_canonique_v2.csv"
    fallback_path = repo_root / "data" / "raw" / "matchs.csv"
    date_candidates = ["date_match", "game_date", "date", "match_date"]
    if base_path.exists():
        return _extract_date_range_from_csv(base_path, date_candidates)
    if fallback_path.exists():
        return _extract_date_range_from_csv(fallback_path, date_candidates)
    raise FileNotFoundError(f"Fichier introuvable : {base_path} ni {fallback_path}")


def cap_end_date_to_today(start_date: date, end_date: date) -> Tuple[date, date, bool]:
    today = date.today()
    capped = end_date > today
    end_date_capped = min(end_date, today)
    if start_date > end_date_capped:
        raise ValueError(f"Plage invalide: {start_date} > {end_date_capped}")
    return start_date, end_date_capped, capped


def build_url(start_date: date, end_date: date, game_type_id: int) -> str:
    sort = json.dumps([{"property": "ppTimeOnIce", "direction": "DESC"},
                       {"property": "playerId", "direction": "ASC"}], separators=(",", ":"))
    cayenne_exp = (f'gameDate<="{end_date.isoformat()} 23:59:59" '
                   f'and gameDate>="{start_date.isoformat()}" '
                   f"and gameTypeId={game_type_id}")
    query = (f"isAggregate=false&isGame=true&sort={quote(sort, safe='')}"
             f"&start=0&limit=-1&cayenneExp={quote(cayenne_exp, safe='')}")
    return f"{BASE_STATS_URL}?{query}"


def fetch_one_chunk(session: requests.Session, start_date: date, end_date: date,
                    game_type_id: int) -> List[Dict]:
    url = build_url(start_date, end_date, game_type_id)
    payload = safe_json_get(session, url)
    if not isinstance(payload, dict):
        raise ValueError("Réponse inattendue : payload non dict")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError(f"Clé 'data' absente. Clés: {list(payload.keys())}")
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
    if "playerId" in out.columns: rename_map["playerId"] = "id_joueur"
    if "gameId" in out.columns: rename_map["gameId"] = "id_match"
    if "gameDate" in out.columns: rename_map["gameDate"] = "date_match"
    out = out.rename(columns=rename_map)
    if "date_match" in out.columns:
        out["date_match"] = pd.to_datetime(out["date_match"], errors="coerce").dt.date.astype("string")
    for col in ["ppTimeOnIce", "ppTimeOnIcePerGame", "gamesPlayed"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    front_cols = [c for c in ["id_joueur", "id_match", "date_match", "ppTimeOnIce",
                               "ppTimeOnIcePerGame", "gamesPlayed"] if c in out.columns]
    other_cols = [c for c in out.columns if c not in front_cols]
    return out[front_cols + other_cols]


def load_existing(csv_path: Path) -> Tuple[pd.DataFrame, date | None]:
    if not csv_path.exists():
        return pd.DataFrame(), None
    df = pd.read_csv(csv_path, low_memory=False)
    if "date_match" not in df.columns:
        return pd.DataFrame(), None
    dates = pd.to_datetime(df["date_match"], errors="coerce").dropna()
    if dates.empty:
        return df, None
    return df, dates.max().date()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--game-type-id", type=int, default=2)
    parser.add_argument("--sleep-seconds", type=float, default=SLEEP_SECONDS)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = get_repo_root()
    raw_dir, outputs_dir = ensure_dirs(repo_root)

    csv_path = raw_dir / "pp_stats_game.csv"
    raw_json_path = raw_dir / "pp_stats_game_raw.json"
    summary_path = outputs_dir / "00a_refresh_pp_stats_summary.json"
    rebuild = args.force or args.rebuild

    if args.start_date and args.end_date:
        inferred_start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
        inferred_end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    else:
        inferred_start, inferred_end = infer_date_range_from_project(repo_root)

    start_date, end_date, end_date_was_capped = cap_end_date_to_today(inferred_start, inferred_end)

    # --- Mode incrémental ---
    existing_df, existing_max_date = (pd.DataFrame(), None) if rebuild else load_existing(csv_path)

    if existing_max_date and not rebuild:
        incremental_start = existing_max_date - timedelta(days=3)
        if incremental_start > end_date:
            print(f"[pp_stats] déjà à jour (max: {existing_max_date}, fin: {end_date})")
            summary = {"status": "ok_already_up_to_date", "existing_max_date": str(existing_max_date),
                        "n_rows": int(len(existing_df))}
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            return
        start_date = max(start_date, incremental_start)
        print(f"[pp_stats] incrémental : existant → {existing_max_date}, fetch depuis {start_date}")
    else:
        print(f"[pp_stats] rebuild complet : {start_date} → {end_date}")

    print(f"=== 00a_refresh_pp_stats.py ===")

    chunks = month_chunks(start_date, end_date)
    print(f"Chunks mensuels : {len(chunks)}")

    session = build_session()
    all_rows: List[Dict] = []

    for i, (cs, ce) in enumerate(chunks, start=1):
        print(f"[{i}/{len(chunks)}] {cs} -> {ce}")
        rows = fetch_one_chunk(session, cs, ce, args.game_type_id)
        print(f"  lignes : {len(rows)}")
        all_rows.extend(rows)
        time.sleep(args.sleep_seconds)

    if not all_rows and existing_df.empty:
        raise ValueError("Aucune ligne récupérée depuis l'endpoint PP stats.")

    new_df = pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    if not new_df.empty:
        new_df = normalize_dataframe(new_df)

    if not existing_df.empty and not new_df.empty:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        for col in ["id_joueur", "id_match"]:
            if col in combined.columns:
                combined[col] = pd.to_numeric(combined[col], errors="coerce")
        combined = combined.drop_duplicates(subset=["id_joueur", "id_match"], keep="last")
    elif not new_df.empty:
        combined = new_df
    else:
        combined = existing_df

    combined.to_csv(csv_path, index=False)
    if all_rows:
        raw_json_path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    summary = {"status": "ok", "mode": "rebuild" if rebuild else "incremental",
               "existing_rows": int(len(existing_df)), "new_rows": len(all_rows),
               "combined_rows": int(len(combined))}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nLignes totales : {len(combined)}")
    print("Terminé.")


if __name__ == "__main__":
    main()
