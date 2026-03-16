#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model/00b_build_base_match_fusionnee.py

Objectif
--------
Reconstruire :
- data/raw/stats.csv
- data/raw/base_match_fusionnee.csv

Entrées attendues
-----------------
- data/raw/joueurs.csv
- data/raw/matchs.csv

Sorties
-------
- data/raw/stats.csv
- data/raw/base_match_fusionnee.csv

Important
---------
- Ce script ne dépend pas de Supabase.
- Il reconstruit les stats joueurs par match depuis l'API NHL.
- Il ne garde que les matchs déjà joués (scores disponibles).
- Il applique ensuite les mêmes contrôles bloquants que dans l'ancien pipeline :
    - match trouvé
    - cohérence équipe
    - cohérence adversaire
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://api-web.nhle.com/v1"
REQUEST_TIMEOUT = 30
DEFAULT_SLEEP_SECONDS = 0.05

TEAM_NAME_FALLBACK = {
    "ANA": "Ducks",
    "ARI": "Coyotes",
    "BOS": "Bruins",
    "BUF": "Sabres",
    "CAR": "Hurricanes",
    "CBJ": "Blue Jackets",
    "CGY": "Flames",
    "CHI": "Blackhawks",
    "COL": "Avalanche",
    "DAL": "Stars",
    "DET": "Red Wings",
    "EDM": "Oilers",
    "FLA": "Panthers",
    "LAK": "Kings",
    "MIN": "Wild",
    "MTL": "Canadiens",
    "NJD": "Devils",
    "NSH": "Predators",
    "NYI": "Islanders",
    "NYR": "Rangers",
    "OTT": "Senators",
    "PHI": "Flyers",
    "PIT": "Penguins",
    "SEA": "Kraken",
    "SJS": "Sharks",
    "STL": "Blues",
    "TBL": "Lightning",
    "TOR": "Maple Leafs",
    "UTA": "Utah Hockey Club",
    "VAN": "Canucks",
    "VGK": "Golden Knights",
    "WPG": "Jets",
    "WSH": "Capitals",
}


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
            "User-Agent": "nhl-value-betting-model/00b_build_base_match_fusionnee",
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


def normalize_team_code(value: Any) -> Optional[str]:
    value = normalize_str(value)
    if value is None:
        return None
    return value.upper()


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


def to_int_or_none(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if isinstance(value, str) and value.strip() == "":
            return None
        return int(float(value))
    except Exception:
        return None


def team_nickname_from_obj(team_obj: Any) -> Optional[str]:
    if not isinstance(team_obj, dict):
        return None

    common_name = team_obj.get("commonName")
    if isinstance(common_name, dict):
        default_name = normalize_str(common_name.get("default"))
        if default_name is not None:
            return default_name

    team_name = team_obj.get("teamName")
    if isinstance(team_name, dict):
        default_name = normalize_str(team_name.get("default"))
        if default_name is not None:
            return default_name

    name_block = team_obj.get("name")
    if isinstance(name_block, dict):
        default_name = normalize_str(name_block.get("default"))
        if default_name is not None:
            return default_name

    return None


def team_abbrev_from_obj(team_obj: Any) -> Optional[str]:
    if isinstance(team_obj, dict):
        for key in ["abbrev", "triCode", "teamAbbrev"]:
            value = normalize_team_code(team_obj.get(key))
            if value is not None:
                return value

    if isinstance(team_obj, str):
        value = normalize_team_code(team_obj)
        if value is not None and len(value) == 3:
            return value

    return None


def player_id_from_obj(player_obj: Dict[str, Any]) -> Optional[int]:
    for key in ["id", "playerId", "id_joueur"]:
        value = to_int_or_none(player_obj.get(key))
        if value is not None:
            return value
    return None


def player_position_from_obj(player_obj: Dict[str, Any], bucket_name: Optional[str] = None) -> Optional[str]:
    for key in ["positionCode", "position", "positionAbbrev"]:
        value = normalize_position(player_obj.get(key))
        if value is not None:
            return value

    if bucket_name is not None:
        bucket_name = bucket_name.lower()
        if bucket_name in {"forwards", "forward"}:
            return None
        if bucket_name in {"defense", "defensemen", "defencemen"}:
            return "D"
        if bucket_name in {"goalies", "goalie"}:
            return "G"

    return None


def is_goalie_bucket(bucket_name: str) -> bool:
    return bucket_name.lower() in {"goalies", "goalie"}


def get_stat_int(player_obj: Dict[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        value = to_int_or_none(player_obj.get(key))
        if value is not None:
            return value
    return default


def get_stat_str(player_obj: Dict[str, Any], *keys: str, default: str = "00:00") -> str:
    for key in keys:
        value = normalize_str(player_obj.get(key))
        if value is not None:
            return value
    return default


def fetch_game_payload(session: requests.Session, game_id: int) -> Any:
    # 1) boxscore
    boxscore_url = f"{BASE_URL}/gamecenter/{game_id}/boxscore"
    try:
        return safe_json_get(session, boxscore_url)
    except Exception:
        pass

    # 2) landing
    landing_url = f"{BASE_URL}/gamecenter/{game_id}/landing"
    return safe_json_get(session, landing_url)


def played_matches_only(df_matchs: pd.DataFrame) -> pd.DataFrame:
    df = df_matchs.copy()

    df["id_match"] = pd.to_numeric(df["id_match"], errors="coerce").astype("Int64")
    df["date_match"] = pd.to_datetime(df["date_match"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["buts_domicile"] = pd.to_numeric(df["buts_domicile"], errors="coerce")
    df["buts_exterieur"] = pd.to_numeric(df["buts_exterieur"], errors="coerce")

    played = df[
        df["id_match"].notna()
        & df["date_match"].notna()
        & df["buts_domicile"].notna()
        & df["buts_exterieur"].notna()
    ].copy()

    return played.sort_values(["date_match", "id_match"]).reset_index(drop=True)


def parse_player_rows_for_side(
    payload: Dict[str, Any],
    side_key: str,
    team_code: str,
    opp_code: str,
    team_name: str,
    opp_name: str,
    game_id: int,
    date_match: str,
    season_source: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    player_stats = payload.get("playerByGameStats")
    if not isinstance(player_stats, dict):
        return rows

    side_block = player_stats.get(side_key)
    if not isinstance(side_block, dict):
        return rows

    for bucket_name, bucket_value in side_block.items():
        if not isinstance(bucket_value, list):
            continue

        if is_goalie_bucket(bucket_name):
            continue

        for player_obj in bucket_value:
            if not isinstance(player_obj, dict):
                continue

            player_id = player_id_from_obj(player_obj)
            position = player_position_from_obj(player_obj, bucket_name=bucket_name)

            if player_id is None:
                continue

            # On ignore les gardiens.
            if position == "G":
                continue

            row = {
                "id_joueur": player_id,
                "id_match": game_id,
                "date_match": date_match,
                "season_source": season_source,
                "team_player_match": team_code,
                "adversaire_match": opp_code,
                "home_road_flag": "H" if side_key == "homeTeam" else "R",
                "is_home_player": 1 if side_key == "homeTeam" else 0,
                "team_name_match": team_name,
                "opponent_name_match": opp_name,
                "buts": get_stat_int(player_obj, "goals"),
                "passes": get_stat_int(player_obj, "assists"),
                "points": get_stat_int(player_obj, "points"),
                "tirs": get_stat_int(player_obj, "sog", "shots"),
                "temps_de_glace": get_stat_str(player_obj, "toi", "timeOnIce", default="00:00"),
                "temps_pp": get_stat_str(player_obj, "powerPlayToi", "ppToi", default="00:00"),
                "plus_moins": get_stat_int(player_obj, "plusMinus"),
                "penalty_minutes": get_stat_int(player_obj, "pim", "penaltyMinutes"),
            }
            rows.append(row)

    return rows


def parse_game_to_stats_rows(
    payload: Any,
    game_id: int,
    date_match: str,
    season_source: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if not isinstance(payload, dict):
        return rows

    home_team_obj = payload.get("homeTeam")
    away_team_obj = payload.get("awayTeam")

    home_code = team_abbrev_from_obj(home_team_obj)
    away_code = team_abbrev_from_obj(away_team_obj)

    if home_code is None or away_code is None:
        return rows

    home_name = first_not_none(team_nickname_from_obj(home_team_obj), TEAM_NAME_FALLBACK.get(home_code), home_code)
    away_name = first_not_none(team_nickname_from_obj(away_team_obj), TEAM_NAME_FALLBACK.get(away_code), away_code)

    rows.extend(
        parse_player_rows_for_side(
            payload=payload,
            side_key="homeTeam",
            team_code=home_code,
            opp_code=away_code,
            team_name=home_name,
            opp_name=away_name,
            game_id=game_id,
            date_match=date_match,
            season_source=season_source,
        )
    )

    rows.extend(
        parse_player_rows_for_side(
            payload=payload,
            side_key="awayTeam",
            team_code=away_code,
            opp_code=home_code,
            team_name=away_name,
            opp_name=home_name,
            game_id=game_id,
            date_match=date_match,
            season_source=season_source,
        )
    )

    return rows


def build_stats_dataframe(
    session: requests.Session,
    df_matchs_played: pd.DataFrame,
    sleep_seconds: float,
) -> pd.DataFrame:
    all_rows: List[Dict[str, Any]] = []
    total_games = len(df_matchs_played)

    print(f"[stats] matchs joués à parcourir : {total_games}")

    for idx, match_row in df_matchs_played.iterrows():
        game_id = int(match_row["id_match"])
        date_match = str(match_row["date_match"])
        season_source = str(match_row["saison"])

        payload = fetch_game_payload(session, game_id)
        game_rows = parse_game_to_stats_rows(
            payload=payload,
            game_id=game_id,
            date_match=date_match,
            season_source=season_source,
        )

        if len(game_rows) == 0:
            raise ValueError(f"Aucune ligne joueur trouvée pour id_match={game_id}")

        all_rows.extend(game_rows)

        if (idx + 1) % 200 == 0 or (idx + 1) == total_games:
            print(f"[stats] {idx + 1}/{total_games}")

        time.sleep(sleep_seconds)

    if not all_rows:
        raise ValueError("stats.csv vide : aucune stat joueur collectée.")

    df = pd.DataFrame(all_rows)

    # Dédoublonnage de sécurité
    df["id_joueur"] = pd.to_numeric(df["id_joueur"], errors="coerce").astype("Int64")
    df["id_match"] = pd.to_numeric(df["id_match"], errors="coerce").astype("Int64")
    df["is_home_player"] = pd.to_numeric(df["is_home_player"], errors="coerce").astype("Int64")

    numeric_cols = ["buts", "passes", "points", "tirs", "plus_moins", "penalty_minutes"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["date_match"] = pd.to_datetime(df["date_match"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["season_source"] = df["season_source"].astype(str)

    df = (
        df.sort_values(["date_match", "id_match", "id_joueur"])
        .drop_duplicates(subset=["id_match", "id_joueur"], keep="last")
        .reset_index(drop=True)
    )

    required_cols = [
        "id_joueur",
        "id_match",
        "date_match",
        "season_source",
        "team_player_match",
        "adversaire_match",
        "home_road_flag",
        "is_home_player",
        "team_name_match",
        "opponent_name_match",
        "buts",
        "passes",
        "points",
        "tirs",
        "temps_de_glace",
        "temps_pp",
        "plus_moins",
        "penalty_minutes",
    ]
    df = df[required_cols]

    missing = int(df["id_joueur"].isna().sum() + df["id_match"].isna().sum())
    if missing > 0:
        raise ValueError(f"stats.csv invalide : {missing} ids manquants.")

    return df


def audit_players_against_joueurs_csv(df_stats: pd.DataFrame, df_joueurs: pd.DataFrame) -> Tuple[int, pd.DataFrame]:
    ids_stats = set(pd.to_numeric(df_stats["id_joueur"], errors="coerce").dropna().astype(int))
    ids_joueurs = set(pd.to_numeric(df_joueurs["id_joueur"], errors="coerce").dropna().astype(int))

    missing_ids = sorted(ids_stats - ids_joueurs)
    missing_count = len(missing_ids)

    if missing_count == 0:
        return 0, pd.DataFrame(columns=["id_joueur"])

    return missing_count, pd.DataFrame({"id_joueur": missing_ids})


def build_base_match_fusionnee(df_stats: pd.DataFrame, df_matchs: pd.DataFrame) -> pd.DataFrame:
    df_matchs_local = df_matchs.copy()
    df_matchs_local["id_match"] = pd.to_numeric(df_matchs_local["id_match"], errors="coerce").astype("Int64")
    df_matchs_local["date_match"] = pd.to_datetime(df_matchs_local["date_match"], errors="coerce").dt.strftime("%Y-%m-%d")

    base = df_stats.merge(
        df_matchs_local,
        on="id_match",
        how="left",
        indicator=True,
        suffixes=("", "_match"),
    ).copy()

    base["match_trouve"] = (base["_merge"] == "both").astype(int)

    base["team_attendue_match"] = base.apply(
        lambda r: r["id_equipe_domicile"] if int(r["is_home_player"]) == 1 else r["id_equipe_exterieur"],
        axis=1,
    )
    base["adversaire_attendu_match"] = base.apply(
        lambda r: r["id_equipe_exterieur"] if int(r["is_home_player"]) == 1 else r["id_equipe_domicile"],
        axis=1,
    )

    base["check_team_ok"] = base["team_player_match"] == base["team_attendue_match"]
    base["check_opp_ok"] = base["adversaire_match"] == base["adversaire_attendu_match"]

    ordered_cols = [
        "id_joueur",
        "id_match",
        "date_match",
        "season_source",
        "team_player_match",
        "adversaire_match",
        "home_road_flag",
        "is_home_player",
        "team_name_match",
        "opponent_name_match",
        "buts",
        "passes",
        "points",
        "tirs",
        "temps_de_glace",
        "temps_pp",
        "plus_moins",
        "penalty_minutes",
        "date_match_match",
        "saison",
        "id_equipe_domicile",
        "id_equipe_exterieur",
        "buts_domicile",
        "buts_exterieur",
        "status",
        "_merge",
        "match_trouve",
        "team_attendue_match",
        "adversaire_attendu_match",
        "check_team_ok",
        "check_opp_ok",
    ]
    base = base[ordered_cols].copy()

    nb_match_non_trouve = int((base["match_trouve"] != 1).sum())
    nb_team_bad = int((base["check_team_ok"] != True).sum())
    nb_opp_bad = int((base["check_opp_ok"] != True).sum())

    print("\n=== CONTROLES BLOQUANTS ===")
    print("matchs non trouvés :", nb_match_non_trouve)
    print("mismatch team :", nb_team_bad)
    print("mismatch adversaire :", nb_opp_bad)

    if nb_match_non_trouve > 0:
        raise ValueError("❌ Il reste des matchs non rattachés")
    if nb_team_bad > 0 or nb_opp_bad > 0:
        raise ValueError("❌ Incohérence équipe/adversaire détectée")

    return base


def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"[OK] fichier créé : {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruit stats.csv et base_match_fusionnee.csv depuis joueurs.csv + matchs.csv + API NHL."
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Pause entre appels API.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = get_raw_dir()

    joueurs_path = raw_dir / "joueurs.csv"
    matchs_path = raw_dir / "matchs.csv"
    stats_path = raw_dir / "stats.csv"
    base_match_fusionnee_path = raw_dir / "base_match_fusionnee.csv"

    if not joueurs_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {joueurs_path}")
    if not matchs_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {matchs_path}")

    print("=== 00b_build_base_match_fusionnee.py ===")
    print(f"Repo root : {get_repo_root()}")
    print(f"Raw dir   : {raw_dir}")
    print("")

    df_joueurs = pd.read_csv(joueurs_path)
    df_matchs = pd.read_csv(matchs_path)

    print(f"[inputs] joueurs.csv : {df_joueurs.shape}")
    print(f"[inputs] matchs.csv  : {df_matchs.shape}")

    df_matchs_played = played_matches_only(df_matchs)
    print(f"[inputs] matchs joués retenus : {df_matchs_played.shape}")
    print("")

    session = build_session()

    # 1) stats.csv
    df_stats = build_stats_dataframe(
        session=session,
        df_matchs_played=df_matchs_played,
        sleep_seconds=args.sleep_seconds,
    )

    joueurs_missing_count, joueurs_missing_df = audit_players_against_joueurs_csv(df_stats, df_joueurs)
    print("")
    print("[audit] joueurs vus dans stats mais absents de joueurs.csv :", joueurs_missing_count)

    if joueurs_missing_count > 0:
        preview = joueurs_missing_df.head(20)
        print(preview.to_string(index=False))

    save_csv(df_stats, stats_path)

    print("")
    print(f"[stats] shape : {df_stats.shape}")
    print(f"[stats] id_match min/max : {df_stats['id_match'].min()} / {df_stats['id_match'].max()}")

    # 2) base_match_fusionnee.csv
    base = build_base_match_fusionnee(df_stats=df_stats, df_matchs=df_matchs_played)
    save_csv(base, base_match_fusionnee_path)

    print("")
    print(f"[base_match_fusionnee] shape : {base.shape}")
    print("Terminé.")


if __name__ == "__main__":
    main()
