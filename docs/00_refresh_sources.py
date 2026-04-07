#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model/00_refresh_sources.py

Objectif
--------
Reconstruire proprement les sources amont sans dépendre de Supabase :

- data/raw/matchs.csv
- data/raw/joueurs.csv

Sorties attendues
-----------------
matchs.csv :
    - id_match
    - date_match
    - saison
    - id_equipe_domicile
    - id_equipe_exterieur
    - buts_domicile
    - buts_exterieur
    - status

joueurs.csv :
    - id_joueur
    - nom
    - position
    - id_equipe

Logique
-------
1. matchs.csv :
   - on boucle par saison
   - on appelle le schedule par équipe
   - on déduplique par id_match

2. joueurs.csv :
   - base principale = rosters par équipe / saison
   - supplément de robustesse = boxscores des matchs
   - on garde au final 1 ligne par joueur, avec la saison la plus récente vue

Important
---------
- Ce script est volontairement robuste et explicite.
- Il ne fait PAS encore stats.csv ni base_match_fusionnee.csv.
- Il prépare seulement les 2 sources amont propres : joueurs + matchs.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://api-web.nhle.com/v1"
REQUEST_TIMEOUT = 30
DEFAULT_MIN_SEASON = 20212022
DEFAULT_SLEEP_SECONDS = 0.10

# Unions de codes équipes utiles sur la plage récente du projet.
# ARI existe avant UTA. UTA remplace ARI à partir de 20242025.
TEAM_CODES = [
    "ANA", "ARI", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL",
    "DET", "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR",
    "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN",
    "VGK", "WPG", "WSH",
]


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def get_raw_dir() -> Path:
    raw_dir = get_repo_root() / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir


def build_session() -> requests.Session:
    session = requests.Session()

    retry = Retry(
        total=5,
        read=5,
        connect=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers.update(
        {
            "User-Agent": "nhl-value-betting-model/00_refresh_sources",
            "Accept": "application/json",
        }
    )
    return session


def safe_json_get(session: requests.Session, url: str) -> Any:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()


def first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def normalize_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    return value


def normalize_position(value: Any) -> Optional[str]:
    value = normalize_str(value)
    if value is None:
        return None

    value = value.upper()

    mapping = {
        "LW": "L",
        "RW": "R",
        "LD": "D",
        "RD": "D",
    }
    return mapping.get(value, value)


def normalize_team_code(value: Any) -> Optional[str]:
    value = normalize_str(value)
    if value is None:
        return None
    return value.upper()


def to_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def valid_team_for_season(team: str, season: int) -> bool:
    # Arizona -> Utah
    if team == "UTA" and season < 20242025:
        return False
    if team == "ARI" and season >= 20242025:
        return False

    # Seattle entre dans la ligue à partir de 20212022.
    if team == "SEA" and season < 20212022:
        return False

    return True


def fetch_available_seasons(session: requests.Session) -> List[int]:
    url = f"{BASE_URL}/season"
    data = safe_json_get(session, url)

    seasons: List[int] = []
    if isinstance(data, list):
        for item in data:
            try:
                seasons.append(int(item))
            except Exception:
                pass

    seasons = sorted(set(seasons))
    if not seasons:
        raise ValueError("Impossible de récupérer la liste des saisons NHL.")
    return seasons


def get_seasons_in_scope(
    session: requests.Session,
    min_season: int,
    max_season: Optional[int],
) -> List[int]:
    all_seasons = fetch_available_seasons(session)

    if max_season is None:
        max_season = max(all_seasons)

    seasons = [s for s in all_seasons if min_season <= s <= max_season]
    if not seasons:
        raise ValueError(
            f"Aucune saison trouvée dans la plage demandée : {min_season} -> {max_season}"
        )
    return seasons


def extract_games_from_schedule_payload(payload: Any) -> List[Dict[str, Any]]:
    games: List[Dict[str, Any]] = []

    if isinstance(payload, dict):
        if isinstance(payload.get("games"), list):
            for game in payload["games"]:
                if isinstance(game, dict):
                    games.append(game)

        if isinstance(payload.get("gameWeek"), list):
            for block in payload["gameWeek"]:
                if isinstance(block, dict) and isinstance(block.get("games"), list):
                    for game in block["games"]:
                        if isinstance(game, dict):
                            games.append(game)

    if not games:
        raise ValueError("Aucun match trouvé dans la réponse schedule.")
    return games


def extract_team_abbrev(team_obj: Any) -> Optional[str]:
    if isinstance(team_obj, dict):
        for key in ["abbrev", "triCode", "teamAbbrev"]:
            value = team_obj.get(key)
            value = normalize_team_code(value)
            if value is not None:
                return value

    if isinstance(team_obj, str):
        value = normalize_team_code(team_obj)
        if value is not None and len(value) == 3:
            return value

    return None


def extract_score(team_obj: Any) -> Optional[int]:
    if isinstance(team_obj, dict):
        return first_not_none(
            to_int_or_none(team_obj.get("score")),
            to_int_or_none(team_obj.get("goals")),
        )
    return None


def build_match_row(game: Dict[str, Any], season: int) -> Optional[Dict[str, Any]]:
    game_id = first_not_none(game.get("id"), game.get("gameId"))
    game_id = to_int_or_none(game_id)
    if game_id is None:
        return None

    date_match = first_not_none(
        normalize_str(game.get("gameDate")),
        normalize_str(game.get("gameDateTime"))[:10] if game.get("gameDateTime") else None,
        normalize_str(game.get("startTimeUTC"))[:10] if game.get("startTimeUTC") else None,
        normalize_str(game.get("gameCenterLink"))[-10:] if game.get("gameCenterLink") else None,
    )

    home_team = extract_team_abbrev(game.get("homeTeam"))
    away_team = extract_team_abbrev(game.get("awayTeam"))

    status = normalize_str(
        first_not_none(
            game.get("gameState"),
            game.get("gameScheduleState"),
            game.get("gameStatus"),
            game.get("status"),
        )
    )

    row = {
        "id_match": game_id,
        "date_match": date_match,
        "saison": str(season),
        "id_equipe_domicile": home_team,
        "id_equipe_exterieur": away_team,
        "buts_domicile": extract_score(game.get("homeTeam")),
        "buts_exterieur": extract_score(game.get("awayTeam")),
        "status": status,
    }

    if row["id_equipe_domicile"] is None or row["id_equipe_exterieur"] is None:
        return None

    return row


def collect_matches(
    session: requests.Session,
    seasons: List[int],
    sleep_seconds: float,
) -> pd.DataFrame:
    all_rows: List[Dict[str, Any]] = []

    for season in seasons:
        print(f"[matchs] saison {season}")

        for team in TEAM_CODES:
            if not valid_team_for_season(team, season):
                continue

            url = f"{BASE_URL}/club-schedule-season/{team}/{season}"

            try:
                payload = safe_json_get(session, url)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    continue
                raise

            games = extract_games_from_schedule_payload(payload)

            for game in games:
                row = build_match_row(game, season)
                if row is not None:
                    all_rows.append(row)

            time.sleep(sleep_seconds)

    if not all_rows:
        raise ValueError("Aucun match collecté. matchs.csv ne peut pas être construit.")

    df = pd.DataFrame(all_rows)

    df = df.drop_duplicates(subset=["id_match"], keep="last").copy()

    df["date_match"] = pd.to_datetime(df["date_match"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["id_match"] = pd.to_numeric(df["id_match"], errors="coerce").astype("Int64")
    df["buts_domicile"] = pd.to_numeric(df["buts_domicile"], errors="coerce").astype("Int64")
    df["buts_exterieur"] = pd.to_numeric(df["buts_exterieur"], errors="coerce").astype("Int64")

    required_cols = [
        "id_match",
        "date_match",
        "saison",
        "id_equipe_domicile",
        "id_equipe_exterieur",
        "buts_domicile",
        "buts_exterieur",
        "status",
    ]
    df = df[required_cols].sort_values(["date_match", "id_match"]).reset_index(drop=True)

    missing_game_ids = int(df["id_match"].isna().sum())
    missing_dates = int(df["date_match"].isna().sum())
    missing_teams = int(df["id_equipe_domicile"].isna().sum() + df["id_equipe_exterieur"].isna().sum())

    if missing_game_ids > 0:
        raise ValueError(f"matchs.csv invalide : {missing_game_ids} id_match manquants.")
    if missing_dates > 0:
        raise ValueError(f"matchs.csv invalide : {missing_dates} date_match manquantes.")
    if missing_teams > 0:
        raise ValueError(f"matchs.csv invalide : {missing_teams} codes équipes manquants.")

    return df


def extract_person_name(player_obj: Dict[str, Any]) -> Optional[str]:
    # Cas 1 : name = {"default": "..."}
    name_obj = player_obj.get("name")
    if isinstance(name_obj, dict):
        default_name = normalize_str(name_obj.get("default"))
        if default_name is not None:
            return default_name

    # Cas 2 : fullName = "...".
    full_name = normalize_str(player_obj.get("fullName"))
    if full_name is not None:
        return full_name

    # Cas 3 : firstName / lastName au format dict {"default": "..."}
    first_name = player_obj.get("firstName")
    last_name = player_obj.get("lastName")

    first_value: Optional[str] = None
    last_value: Optional[str] = None

    if isinstance(first_name, dict):
        first_value = normalize_str(first_name.get("default"))
    else:
        first_value = normalize_str(first_name)

    if isinstance(last_name, dict):
        last_value = normalize_str(last_name.get("default"))
    else:
        last_value = normalize_str(last_name)

    if first_value and last_value:
        return f"{first_value} {last_value}"

    # Cas 4 : nom simple
    simple_name = normalize_str(player_obj.get("playerName"))
    if simple_name is not None:
        return simple_name

    return None


def extract_player_id(player_obj: Dict[str, Any]) -> Optional[int]:
    for key in ["id", "playerId", "id_joueur"]:
        value = to_int_or_none(player_obj.get(key))
        if value is not None:
            return value
    return None


def extract_player_position(player_obj: Dict[str, Any], bucket_name: Optional[str] = None) -> Optional[str]:
    for key in ["positionCode", "position", "positionAbbrev"]:
        value = normalize_position(player_obj.get(key))
        if value is not None:
            return value

    if bucket_name is not None:
        bucket_name = bucket_name.lower()
        if bucket_name in {"defensemen", "defense"}:
            return "D"
        if bucket_name in {"goalies", "goalie"}:
            return "G"

    return None


def parse_roster_payload(payload: Any, team_code: str, season: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if not isinstance(payload, dict):
        return rows

    for bucket_name, bucket_value in payload.items():
        if not isinstance(bucket_value, list):
            continue

        for player in bucket_value:
            if not isinstance(player, dict):
                continue

            player_id = extract_player_id(player)
            name = extract_person_name(player)
            position = extract_player_position(player, bucket_name=bucket_name)

            if player_id is None or name is None or position is None:
                continue

            rows.append(
                {
                    "id_joueur": player_id,
                    "nom": name,
                    "position": position,
                    "id_equipe": team_code,
                    "saison": int(season),
                    "source_priority": 1,  # roster
                }
            )

    return rows


def collect_players_from_rosters(
    session: requests.Session,
    seasons: List[int],
    sleep_seconds: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    for season in seasons:
        print(f"[joueurs/rosters] saison {season}")

        for team in TEAM_CODES:
            if not valid_team_for_season(team, season):
                continue

            url = f"{BASE_URL}/roster/{team}/{season}"

            try:
                payload = safe_json_get(session, url)
            except requests.HTTPError as exc:
                if exc.response is not None and exc.response.status_code == 404:
                    continue
                raise

            rows.extend(parse_roster_payload(payload, team_code=team, season=season))
            time.sleep(sleep_seconds)

    if not rows:
        raise ValueError("Aucun joueur collecté via rosters.")

    df = pd.DataFrame(rows)
    return df


def parse_boxscore_players(payload: Any, season: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if not isinstance(payload, dict):
        return rows

    home_team = extract_team_abbrev(payload.get("homeTeam"))
    away_team = extract_team_abbrev(payload.get("awayTeam"))

    player_stats = payload.get("playerByGameStats")
    if not isinstance(player_stats, dict):
        return rows

    for side_key, team_code in [("homeTeam", home_team), ("awayTeam", away_team)]:
        if team_code is None:
            continue

        side_block = player_stats.get(side_key)
        if not isinstance(side_block, dict):
            continue

        for bucket_name, bucket_value in side_block.items():
            if not isinstance(bucket_value, list):
                continue

            for player in bucket_value:
                if not isinstance(player, dict):
                    continue

                player_id = extract_player_id(player)
                name = extract_person_name(player)
                position = extract_player_position(player, bucket_name=bucket_name)

                if player_id is None or name is None or position is None:
                    continue

                rows.append(
                    {
                        "id_joueur": player_id,
                        "nom": name,
                        "position": position,
                        "id_equipe": team_code,
                        "saison": int(season),
                        "source_priority": 2,  # boxscore > roster
                    }
                )

    return rows


def fetch_boxscore_with_fallback(
    session: requests.Session,
    game_id: int,
) -> Any:
    # 1) boxscore
    boxscore_url = f"{BASE_URL}/gamecenter/{game_id}/boxscore"
    try:
        return safe_json_get(session, boxscore_url)
    except Exception:
        pass

    # 2) landing fallback
    landing_url = f"{BASE_URL}/gamecenter/{game_id}/landing"
    return safe_json_get(session, landing_url)


def collect_players_from_boxscores(
    session: requests.Session,
    matches_df: pd.DataFrame,
    sleep_seconds: float,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    unique_games = matches_df[["id_match", "saison"]].drop_duplicates().reset_index(drop=True)

    total_games = len(unique_games)
    print(f"[joueurs/boxscores] matchs à parcourir : {total_games}")

    for idx, row in unique_games.iterrows():
        game_id = int(row["id_match"])
        season = int(row["saison"])

        payload = fetch_boxscore_with_fallback(session, game_id)
        rows.extend(parse_boxscore_players(payload, season=season))

        if (idx + 1) % 200 == 0 or (idx + 1) == total_games:
            print(f"[joueurs/boxscores] {idx + 1}/{total_games}")

        time.sleep(sleep_seconds)

    if not rows:
        raise ValueError("Aucun joueur collecté via boxscores.")

    df = pd.DataFrame(rows)
    return df


def build_final_players_dataframe(
    roster_df: pd.DataFrame,
    boxscore_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    frames = [roster_df]
    if boxscore_df is not None and not boxscore_df.empty:
        frames.append(boxscore_df)

    df = pd.concat(frames, ignore_index=True)

    df["id_joueur"] = pd.to_numeric(df["id_joueur"], errors="coerce").astype("Int64")
    df["nom"] = df["nom"].astype(str).str.strip()
    df["position"] = df["position"].astype(str).str.upper().str.strip()
    df["id_equipe"] = df["id_equipe"].astype(str).str.upper().str.strip()
    df["saison"] = pd.to_numeric(df["saison"], errors="coerce").astype("Int64")
    df["source_priority"] = pd.to_numeric(df["source_priority"], errors="coerce").astype("Int64")

    df = df.dropna(subset=["id_joueur", "nom", "position", "id_equipe", "saison", "source_priority"]).copy()

    # On garde la version la plus récente vue du joueur.
    # En cas d'égalité de saison, le boxscore passe devant le roster.
    df = df.sort_values(["id_joueur", "saison", "source_priority"]).drop_duplicates(
        subset=["id_joueur"], keep="last"
    )

    final_df = df[["id_joueur", "nom", "position", "id_equipe"]].sort_values(
        ["nom", "id_joueur"]
    ).reset_index(drop=True)

    if final_df["id_joueur"].isna().any():
        raise ValueError("joueurs.csv invalide : id_joueur manquant.")
    if final_df["nom"].eq("").any():
        raise ValueError("joueurs.csv invalide : nom vide.")
    if final_df["position"].eq("").any():
        raise ValueError("joueurs.csv invalide : position vide.")
    if final_df["id_equipe"].eq("").any():
        raise ValueError("joueurs.csv invalide : id_equipe vide.")

    return final_df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"[OK] fichier créé : {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruit data/raw/matchs.csv et data/raw/joueurs.csv depuis l'API NHL."
    )
    parser.add_argument(
        "--min-season",
        type=int,
        default=DEFAULT_MIN_SEASON,
        help="Première saison à inclure, ex: 20212022",
    )
    parser.add_argument(
        "--max-season",
        type=int,
        default=None,
        help="Dernière saison à inclure. Par défaut = dernière saison disponible via l'API NHL.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Pause entre appels API.",
    )
    parser.add_argument(
        "--skip-boxscore-supplement",
        action="store_true",
        default=True,
        help="Ne pas compléter joueurs.csv avec les boxscores (défaut: activé pour la vitesse).",
    )
    parser.add_argument(
        "--with-boxscore-supplement",
        action="store_true",
        default=False,
        help="Forcer le supplément boxscores (lent, ~6h pour toutes les saisons).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = get_raw_dir()
    session = build_session()

    print("=== 00_refresh_sources.py ===")
    print(f"Repo root : {get_repo_root()}")
    print(f"Raw dir   : {raw_dir}")
    print("")

    seasons = get_seasons_in_scope(
        session=session,
        min_season=args.min_season,
        max_season=args.max_season,
    )

    print(f"Saisons retenues : {seasons[0]} -> {seasons[-1]}")
    print("")

    # 1) MATCHS
    matches_df = collect_matches(
        session=session,
        seasons=seasons,
        sleep_seconds=args.sleep_seconds,
    )

    matches_path = raw_dir / "matchs.csv"
    save_csv(matches_df, matches_path)

    print("")
    print(f"[matchs] nb lignes : {len(matches_df)}")
    print(f"[matchs] min id_match : {matches_df['id_match'].min()}")
    print(f"[matchs] max id_match : {matches_df['id_match'].max()}")
    print("")

    # 2) JOUEURS - base roster
    roster_df = collect_players_from_rosters(
        session=session,
        seasons=seasons,
        sleep_seconds=args.sleep_seconds,
    )

    print(f"[joueurs/rosters] nb lignes brutes : {len(roster_df)}")

    # 3) JOUEURS - supplément boxscores
    boxscore_df: Optional[pd.DataFrame] = None
    if args.with_boxscore_supplement:
        boxscore_df = collect_players_from_boxscores(
            session=session,
            matches_df=matches_df,
            sleep_seconds=args.sleep_seconds,
        )
        print(f"[joueurs/boxscores] nb lignes brutes : {len(boxscore_df)}")
    else:
        print("[joueurs/boxscores] supplément désactivé (défaut). Utilise --with-boxscore-supplement pour forcer.")

    players_df = build_final_players_dataframe(
        roster_df=roster_df,
        boxscore_df=boxscore_df,
    )

    players_path = raw_dir / "joueurs.csv"
    save_csv(players_df, players_path)

    print("")
    print(f"[joueurs] nb lignes finales : {len(players_df)}")
    print(f"[joueurs] positions : {sorted(players_df['position'].dropna().astype(str).unique().tolist())}")
    print("")
    print("Terminé.")


if __name__ == "__main__":
    main()
