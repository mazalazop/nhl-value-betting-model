#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model/00b_build_base_match_fusionnee.py

MODIFIÉ — mode incrémental par défaut.

Changement principal :
- Si data/raw/stats.csv existe déjà, on ne refetch que les matchs
  dont l'id_match est absent de stats.csv.
- Flag --rebuild pour forcer un refresh complet.
- Réduit le temps d'exécution de ~6h à quelques minutes en mode quotidien.
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
    "ANA": "Ducks", "ARI": "Coyotes", "BOS": "Bruins", "BUF": "Sabres",
    "CAR": "Hurricanes", "CBJ": "Blue Jackets", "CGY": "Flames", "CHI": "Blackhawks",
    "COL": "Avalanche", "DAL": "Stars", "DET": "Red Wings", "EDM": "Oilers",
    "FLA": "Panthers", "LAK": "Kings", "MIN": "Wild", "MTL": "Canadiens",
    "NJD": "Devils", "NSH": "Predators", "NYI": "Islanders", "NYR": "Rangers",
    "OTT": "Senators", "PHI": "Flyers", "PIT": "Penguins", "SEA": "Kraken",
    "SJS": "Sharks", "STL": "Blues", "TBL": "Lightning", "TOR": "Maple Leafs",
    "UTA": "Utah Hockey Club", "VAN": "Canucks", "VGK": "Golden Knights",
    "WPG": "Jets", "WSH": "Capitals",
}


def get_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def get_raw_dir() -> Path:
    raw_dir = get_repo_root() / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir

def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=5, read=5, connect=5, backoff_factor=1.0,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=frozenset(["GET"]), raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "nhl-value-betting-model/00b",
                            "Accept": "application/json"})
    return session

def safe_json_get(session: requests.Session, url: str) -> Any:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    if response.status_code >= 400:
        response.raise_for_status()
    return response.json()

def first_not_none(*values: Any) -> Any:
    for v in values:
        if v is not None:
            return v
    return None

def normalize_str(value: Any) -> Optional[str]:
    if value is None: return None
    value = str(value).strip()
    return value if value else None

def normalize_team_code(value: Any) -> Optional[str]:
    value = normalize_str(value)
    return value.upper() if value else None

def normalize_position(value: Any) -> Optional[str]:
    value = normalize_str(value)
    if value is None: return None
    value = value.upper()
    return {"LW": "L", "RW": "R", "LD": "D", "RD": "D"}.get(value, value)

def to_int_or_none(value: Any) -> Optional[int]:
    if value is None: return None
    try:
        if isinstance(value, str) and value.strip() == "": return None
        return int(float(value))
    except Exception: return None

def team_nickname_from_obj(team_obj: Any) -> Optional[str]:
    if not isinstance(team_obj, dict): return None
    for key in ["commonName", "teamName", "name"]:
        obj = team_obj.get(key)
        if isinstance(obj, dict):
            default_name = normalize_str(obj.get("default"))
            if default_name: return default_name
    return None

def team_abbrev_from_obj(team_obj: Any) -> Optional[str]:
    if isinstance(team_obj, dict):
        for key in ["abbrev", "triCode", "teamAbbrev"]:
            value = normalize_team_code(team_obj.get(key))
            if value: return value
    if isinstance(team_obj, str):
        value = normalize_team_code(team_obj)
        if value and len(value) == 3: return value
    return None

def player_id_from_obj(player_obj: Dict[str, Any]) -> Optional[int]:
    for key in ["id", "playerId", "id_joueur"]:
        value = to_int_or_none(player_obj.get(key))
        if value is not None: return value
    return None

def player_position_from_obj(player_obj: Dict[str, Any], bucket_name: Optional[str] = None) -> Optional[str]:
    for key in ["positionCode", "position", "positionAbbrev"]:
        value = normalize_position(player_obj.get(key))
        if value: return value
    if bucket_name:
        bucket_name = bucket_name.lower()
        if bucket_name in {"defense", "defensemen", "defencemen"}: return "D"
        if bucket_name in {"goalies", "goalie"}: return "G"
    return None

def is_goalie_bucket(bucket_name: str) -> bool:
    return bucket_name.lower() in {"goalies", "goalie"}

def get_stat_int(player_obj: Dict[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        value = to_int_or_none(player_obj.get(key))
        if value is not None: return value
    return default

def get_stat_str(player_obj: Dict[str, Any], *keys: str, default: str = "00:00") -> str:
    for key in keys:
        value = normalize_str(player_obj.get(key))
        if value: return value
    return default

def fetch_game_payload(session: requests.Session, game_id: int) -> Any:
    for endpoint in ["boxscore", "landing"]:
        try:
            url = f"{BASE_URL}/gamecenter/{game_id}/{endpoint}"
            return safe_json_get(session, url)
        except Exception:
            pass
    raise RuntimeError(f"Impossible de récupérer les données pour le match {game_id}")

def played_matches_only(df_matchs: pd.DataFrame) -> pd.DataFrame:
    df = df_matchs.copy()
    df["id_match"] = pd.to_numeric(df["id_match"], errors="coerce").astype("Int64")
    df["date_match"] = pd.to_datetime(df["date_match"], errors="coerce").dt.strftime("%Y-%m-%d")
    df["buts_domicile"] = pd.to_numeric(df["buts_domicile"], errors="coerce")
    df["buts_exterieur"] = pd.to_numeric(df["buts_exterieur"], errors="coerce")
    played = df[df["id_match"].notna() & df["date_match"].notna()
                & df["buts_domicile"].notna() & df["buts_exterieur"].notna()].copy()
    return played.sort_values(["date_match", "id_match"]).reset_index(drop=True)

def parse_player_rows_for_side(
    payload, side_key, team_code, opp_code, team_name, opp_name,
    game_id, date_match, season_source,
) -> List[Dict[str, Any]]:
    rows = []
    player_stats = payload.get("playerByGameStats")
    if not isinstance(player_stats, dict): return rows
    side_block = player_stats.get(side_key)
    if not isinstance(side_block, dict): return rows
    for bucket_name, bucket_value in side_block.items():
        if not isinstance(bucket_value, list): continue
        if is_goalie_bucket(bucket_name): continue
        for player_obj in bucket_value:
            if not isinstance(player_obj, dict): continue
            player_id = player_id_from_obj(player_obj)
            position = player_position_from_obj(player_obj, bucket_name=bucket_name)
            if player_id is None or position == "G": continue
            rows.append({
                "id_joueur": player_id, "id_match": game_id,
                "date_match": date_match, "season_source": season_source,
                "team_player_match": team_code, "adversaire_match": opp_code,
                "home_road_flag": "H" if side_key == "homeTeam" else "R",
                "is_home_player": 1 if side_key == "homeTeam" else 0,
                "team_name_match": team_name, "opponent_name_match": opp_name,
                "buts": get_stat_int(player_obj, "goals"),
                "passes": get_stat_int(player_obj, "assists"),
                "points": get_stat_int(player_obj, "points"),
                "tirs": get_stat_int(player_obj, "sog", "shots"),
                "temps_de_glace": get_stat_str(player_obj, "toi", "timeOnIce", default="00:00"),
                "temps_pp": get_stat_str(player_obj, "powerPlayToi", "ppToi", default="00:00"),
                "plus_moins": get_stat_int(player_obj, "plusMinus"),
                "penalty_minutes": get_stat_int(player_obj, "pim", "penaltyMinutes"),
            })
    return rows

def parse_game_to_stats_rows(payload, game_id, date_match, season_source) -> List[Dict[str, Any]]:
    rows = []
    if not isinstance(payload, dict): return rows
    home_team_obj = payload.get("homeTeam")
    away_team_obj = payload.get("awayTeam")
    home_code = team_abbrev_from_obj(home_team_obj)
    away_code = team_abbrev_from_obj(away_team_obj)
    if home_code is None or away_code is None: return rows
    home_name = first_not_none(team_nickname_from_obj(home_team_obj), TEAM_NAME_FALLBACK.get(home_code), home_code)
    away_name = first_not_none(team_nickname_from_obj(away_team_obj), TEAM_NAME_FALLBACK.get(away_code), away_code)
    for side_key, tc, oc, tn, on_ in [
        ("homeTeam", home_code, away_code, home_name, away_name),
        ("awayTeam", away_code, home_code, away_name, home_name),
    ]:
        rows.extend(parse_player_rows_for_side(
            payload, side_key, tc, oc, tn, on_, game_id, date_match, season_source))
    return rows

def load_existing_stats(stats_path: Path) -> pd.DataFrame:
    """Charge stats.csv existant et retourne le DataFrame (vide si absent)."""
    if not stats_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(stats_path, low_memory=False)
    df["id_match"] = pd.to_numeric(df["id_match"], errors="coerce").astype("Int64")
    return df

def build_stats_dataframe(
    session: requests.Session,
    df_matchs_played: pd.DataFrame,
    sleep_seconds: float,
    existing_stats: pd.DataFrame,
) -> pd.DataFrame:
    """Mode incrémental : ne fetch que les matchs manquants dans existing_stats."""
    
    if existing_stats.empty:
        matches_to_fetch = df_matchs_played
        print(f"[stats] mode rebuild complet : {len(matches_to_fetch)} matchs à fetcher")
    else:
        existing_ids = set(existing_stats["id_match"].dropna().astype(int).tolist())
        all_played_ids = set(df_matchs_played["id_match"].dropna().astype(int).tolist())
        missing_ids = all_played_ids - existing_ids
        matches_to_fetch = df_matchs_played[
            df_matchs_played["id_match"].astype(int).isin(missing_ids)
        ].copy()
        print(f"[stats] mode incrémental : {len(existing_ids)} matchs déjà en cache, "
              f"{len(missing_ids)} à fetcher")
    
    if matches_to_fetch.empty:
        print("[stats] rien à fetcher, stats.csv est à jour")
        return existing_stats

    new_rows: List[Dict[str, Any]] = []
    total_games = len(matches_to_fetch)
    
    for idx, (_, match_row) in enumerate(matches_to_fetch.iterrows()):
        game_id = int(match_row["id_match"])
        date_match = str(match_row["date_match"])
        season_source = str(match_row["saison"])
        
        try:
            payload = fetch_game_payload(session, game_id)
            game_rows = parse_game_to_stats_rows(payload, game_id, date_match, season_source)
            if len(game_rows) == 0:
                print(f"[stats] WARNING: 0 lignes joueur pour id_match={game_id}, skip")
                continue
            new_rows.extend(game_rows)
        except Exception as e:
            print(f"[stats] ERROR fetching id_match={game_id}: {e}")
            continue

        if (idx + 1) % 100 == 0 or (idx + 1) == total_games:
            print(f"[stats] {idx + 1}/{total_games}")
        time.sleep(sleep_seconds)

    if not new_rows and existing_stats.empty:
        raise ValueError("stats.csv vide : aucune stat joueur collectée.")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([existing_stats, new_df], ignore_index=True)
    else:
        combined = existing_stats.copy()

    # Nettoyage uniforme
    combined["id_joueur"] = pd.to_numeric(combined["id_joueur"], errors="coerce").astype("Int64")
    combined["id_match"] = pd.to_numeric(combined["id_match"], errors="coerce").astype("Int64")
    combined["is_home_player"] = pd.to_numeric(combined["is_home_player"], errors="coerce").astype("Int64")
    numeric_cols = ["buts", "passes", "points", "tirs", "plus_moins", "penalty_minutes"]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0).astype(int)
    combined["date_match"] = pd.to_datetime(combined["date_match"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined["season_source"] = combined["season_source"].astype(str)
    combined = (combined.sort_values(["date_match", "id_match", "id_joueur"])
                .drop_duplicates(subset=["id_match", "id_joueur"], keep="last")
                .reset_index(drop=True))

    required_cols = [
        "id_joueur", "id_match", "date_match", "season_source",
        "team_player_match", "adversaire_match", "home_road_flag", "is_home_player",
        "team_name_match", "opponent_name_match", "buts", "passes", "points",
        "tirs", "temps_de_glace", "temps_pp", "plus_moins", "penalty_minutes",
    ]
    for col in required_cols:
        if col not in combined.columns:
            combined[col] = ""
    combined = combined[required_cols]

    missing = int(combined["id_joueur"].isna().sum() + combined["id_match"].isna().sum())
    if missing > 0:
        raise ValueError(f"stats.csv invalide : {missing} ids manquants.")
    return combined

def build_base_match_fusionnee(df_stats: pd.DataFrame, df_matchs: pd.DataFrame) -> pd.DataFrame:
    df_matchs_local = df_matchs.copy()
    df_matchs_local["id_match"] = pd.to_numeric(df_matchs_local["id_match"], errors="coerce").astype("Int64")
    df_matchs_local["date_match"] = pd.to_datetime(df_matchs_local["date_match"], errors="coerce").dt.strftime("%Y-%m-%d")
    base = df_stats.merge(df_matchs_local, on="id_match", how="left",
                          indicator=True, suffixes=("", "_match")).copy()
    base["match_trouve"] = (base["_merge"] == "both").astype(int)
    base["team_attendue_match"] = base.apply(
        lambda r: r["id_equipe_domicile"] if int(r["is_home_player"]) == 1 else r["id_equipe_exterieur"], axis=1)
    base["adversaire_attendu_match"] = base.apply(
        lambda r: r["id_equipe_exterieur"] if int(r["is_home_player"]) == 1 else r["id_equipe_domicile"], axis=1)
    base["check_team_ok"] = base["team_player_match"] == base["team_attendue_match"]
    base["check_opp_ok"] = base["adversaire_match"] == base["adversaire_attendu_match"]

    nb_match_non_trouve = int((base["match_trouve"] != 1).sum())
    nb_team_bad = int((base["check_team_ok"] != True).sum())
    nb_opp_bad = int((base["check_opp_ok"] != True).sum())
    print(f"\n=== CONTROLES BLOQUANTS ===")
    print(f"matchs non trouvés : {nb_match_non_trouve}")
    print(f"mismatch team : {nb_team_bad}")
    print(f"mismatch adversaire : {nb_opp_bad}")
    if nb_match_non_trouve > 0:
        raise ValueError("❌ Il reste des matchs non rattachés")
    if nb_team_bad > 0 or nb_opp_bad > 0:
        raise ValueError("❌ Incohérence équipe/adversaire détectée")
    return base

def save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"[OK] fichier créé : {path}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sleep-seconds", type=float, default=DEFAULT_SLEEP_SECONDS)
    parser.add_argument("--rebuild", action="store_true",
                        help="Forcer un rebuild complet de stats.csv (ignorer le cache existant).")
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
    print(f"Mode : {'REBUILD' if args.rebuild else 'INCREMENTAL'}")

    df_joueurs = pd.read_csv(joueurs_path)
    df_matchs = pd.read_csv(matchs_path)
    print(f"[inputs] joueurs.csv : {df_joueurs.shape}")
    print(f"[inputs] matchs.csv  : {df_matchs.shape}")

    df_matchs_played = played_matches_only(df_matchs)
    print(f"[inputs] matchs joués retenus : {df_matchs_played.shape}")

    session = build_session()
    
    existing_stats = pd.DataFrame() if args.rebuild else load_existing_stats(stats_path)
    
    df_stats = build_stats_dataframe(
        session=session,
        df_matchs_played=df_matchs_played,
        sleep_seconds=args.sleep_seconds,
        existing_stats=existing_stats,
    )
    save_csv(df_stats, stats_path)
    print(f"\n[stats] shape : {df_stats.shape}")

    base = build_base_match_fusionnee(df_stats=df_stats, df_matchs=df_matchs_played)
    save_csv(base, base_match_fusionnee_path)
    print(f"\n[base_match_fusionnee] shape : {base.shape}")
    print("Terminé.")

if __name__ == "__main__":
    main()
