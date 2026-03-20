from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

INPUT_BASE_MATCH_FUSIONNEE = RAW_DIR / "base_match_fusionnee.csv"
OUTPUT_STANDINGS = RAW_DIR / "team_standings_daily.csv"
SUMMARY_PATH = OUTPUTS_DIR / "00c_refresh_team_standings_summary.json"

NHL_STANDINGS_URL = "https://api-web.nhle.com/v1/standings/{date_str}"
EXPECTED_TEAMS_PER_DAY = 32
REQUEST_TIMEOUT = 30
DEFAULT_SLEEP_SECONDS = 0.15
PARIS_TZ_NAME = "Europe/Paris"


@dataclass
class RefreshConfig:
    base_match_path: Path
    output_path: Path
    summary_path: Path
    start_date: str | None
    end_date: str | None
    extra_dates: list[str]
    include_today: bool
    rebuild: bool
    sleep_seconds: float


def ensure_directories(output_path: Path, summary_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> RefreshConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Télécharge des snapshots journaliers de standings NHL et construit "
            "data/raw/team_standings_daily.csv de manière incrémentale."
        )
    )
    parser.add_argument("--start-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end-date", type=str, default=None, help="YYYY-MM-DD")
    parser.add_argument(
        "--extra-date",
        dest="extra_dates",
        action="append",
        default=[],
        help="Date supplémentaire YYYY-MM-DD à forcer dans le refresh. Répéter l'option si besoin.",
    )
    parser.add_argument(
        "--base-match-path",
        type=str,
        default=str(INPUT_BASE_MATCH_FUSIONNEE),
        help="Chemin vers base_match_fusionnee.csv utilisé pour déduire les dates utiles.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(OUTPUT_STANDINGS),
        help="Chemin du CSV de sortie.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=str(SUMMARY_PATH),
        help="Chemin du JSON de résumé.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reconstruit complètement le CSV au lieu d'un refresh incrémental.",
    )
    parser.add_argument(
        "--no-include-today",
        action="store_true",
        help="N'ajoute pas automatiquement la date du jour Europe/Paris aux dates à récupérer.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=DEFAULT_SLEEP_SECONDS,
        help="Pause entre deux appels API pour rester raisonnable.",
    )

    args = parser.parse_args()
    return RefreshConfig(
        base_match_path=Path(args.base_match_path),
        output_path=Path(args.output_path),
        summary_path=Path(args.summary_path),
        start_date=args.start_date,
        end_date=args.end_date,
        extra_dates=args.extra_dates,
        include_today=not args.no_include_today,
        rebuild=bool(args.rebuild),
        sleep_seconds=float(args.sleep_seconds),
    )


def make_session() -> requests.Session:
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.75,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update(
        {
            "User-Agent": "Henachel-NHL-Standings-Refresh/1.0",
            "Accept": "application/json",
        }
    )
    return session


def today_paris_str() -> str:
    if ZoneInfo is None:
        return pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    return pd.Timestamp.now(tz=ZoneInfo(PARIS_TZ_NAME)).strftime("%Y-%m-%d")


def normalize_date_strings(values: Iterable[str | pd.Timestamp]) -> list[str]:
    out: list[str] = []
    for value in values:
        dt = pd.to_datetime(value, errors="coerce")
        if pd.isna(dt):
            continue
        out.append(dt.strftime("%Y-%m-%d"))
    return sorted(set(out))


def load_target_dates_from_base_match(
    base_match_path: Path,
    start_date: str | None,
    end_date: str | None,
) -> list[str]:
    if not base_match_path.exists():
        if start_date and end_date:
            return normalize_date_strings(pd.date_range(start=start_date, end=end_date, freq="D"))
        raise FileNotFoundError(
            f"Fichier introuvable : {base_match_path}. "
            "Fournis --start-date/--end-date ou place base_match_fusionnee.csv dans data/raw/."
        )

    usecols = ["date_match"]
    base = pd.read_csv(base_match_path, usecols=usecols, low_memory=False)
    if "date_match" not in base.columns:
        raise ValueError("La colonne date_match est absente de base_match_fusionnee.csv")

    base["date_match"] = pd.to_datetime(base["date_match"], errors="coerce")
    base = base[base["date_match"].notna()].copy()

    if start_date is not None:
        base = base[base["date_match"] >= pd.to_datetime(start_date)]
    if end_date is not None:
        base = base[base["date_match"] <= pd.to_datetime(end_date)]

    return normalize_date_strings(base["date_match"].dropna().unique())


def load_existing_output(output_path: Path) -> pd.DataFrame:
    if not output_path.exists():
        return pd.DataFrame()

    existing = pd.read_csv(output_path, low_memory=False)
    required = {"date_snapshot", "team_abbrev"}
    missing = [c for c in required if c not in existing.columns]
    if missing:
        raise ValueError(
            f"Le fichier existant {output_path} ne contient pas les colonnes requises : {missing}"
        )

    existing["date_snapshot"] = pd.to_datetime(existing["date_snapshot"], errors="coerce").dt.strftime("%Y-%m-%d")
    existing = existing[existing["date_snapshot"].notna()].copy()
    existing["team_abbrev"] = existing["team_abbrev"].astype(str).str.upper().str.strip()
    return existing


def extract_nested_default(value) -> str | None:
    if isinstance(value, dict):
        default = value.get("default")
        if default is None:
            return None
        return str(default)
    if value is None:
        return None
    return str(value)


def flatten_team_record(record: dict, requested_date: str) -> dict:
    points = pd.to_numeric(record.get("points"), errors="coerce")
    games_played = pd.to_numeric(record.get("gamesPlayed"), errors="coerce")

    output = {
        "date_snapshot": requested_date,
        "api_date": record.get("date"),
        "season_id": record.get("seasonId"),
        "team_abbrev": extract_nested_default(record.get("teamAbbrev")),
        "team_name": extract_nested_default(record.get("teamName")),
        "team_common_name": extract_nested_default(record.get("teamCommonName")),
        "conference_abbrev": record.get("conferenceAbbrev"),
        "conference_name": record.get("conferenceName"),
        "division_abbrev": record.get("divisionAbbrev"),
        "division_name": record.get("divisionName"),
        "games_played": games_played,
        "games_remaining": 82 - games_played if pd.notna(games_played) else pd.NA,
        "points": points,
        "point_pctg": pd.to_numeric(record.get("pointPctg"), errors="coerce"),
        "wins": pd.to_numeric(record.get("wins"), errors="coerce"),
        "losses": pd.to_numeric(record.get("losses"), errors="coerce"),
        "ot_losses": pd.to_numeric(record.get("otLosses"), errors="coerce"),
        "regulation_wins": pd.to_numeric(record.get("regulationWins"), errors="coerce"),
        "regulation_plus_ot_wins": pd.to_numeric(record.get("regulationPlusOtWins"), errors="coerce"),
        "goal_for": pd.to_numeric(record.get("goalFor"), errors="coerce"),
        "goal_against": pd.to_numeric(record.get("goalAgainst"), errors="coerce"),
        "goal_differential": pd.to_numeric(record.get("goalDifferential"), errors="coerce"),
        "league_sequence": pd.to_numeric(record.get("leagueSequence"), errors="coerce"),
        "conference_sequence": pd.to_numeric(record.get("conferenceSequence"), errors="coerce"),
        "division_sequence": pd.to_numeric(record.get("divisionSequence"), errors="coerce"),
        "wildcard_sequence": pd.to_numeric(record.get("wildcardSequence"), errors="coerce"),
        "waivers_sequence": pd.to_numeric(record.get("waiversSequence"), errors="coerce"),
        "streak_code": record.get("streakCode"),
        "streak_count": pd.to_numeric(record.get("streakCount"), errors="coerce"),
        "l10_games_played": pd.to_numeric(record.get("l10GamesPlayed"), errors="coerce"),
        "l10_wins": pd.to_numeric(record.get("l10Wins"), errors="coerce"),
        "l10_losses": pd.to_numeric(record.get("l10Losses"), errors="coerce"),
        "l10_ot_losses": pd.to_numeric(record.get("l10OtLosses"), errors="coerce"),
        "l10_points": pd.to_numeric(record.get("l10Points"), errors="coerce"),
        "home_games_played": pd.to_numeric(record.get("homeGamesPlayed"), errors="coerce"),
        "home_points": pd.to_numeric(record.get("homePoints"), errors="coerce"),
        "road_games_played": pd.to_numeric(record.get("roadGamesPlayed"), errors="coerce"),
        "road_points": pd.to_numeric(record.get("roadPoints"), errors="coerce"),
        "team_logo": record.get("teamLogo"),
    }
    return output


def fetch_standings_for_date(session: requests.Session, date_str: str) -> tuple[pd.DataFrame, dict]:
    url = NHL_STANDINGS_URL.format(date_str=date_str)
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code} pour {url}")

    payload = response.json()
    raw_rows = payload.get("standings", [])
    if not isinstance(raw_rows, list):
        raise ValueError(f"Payload inattendu pour {date_str} : clé standings absente ou invalide")

    rows = [flatten_team_record(record, requested_date=date_str) for record in raw_rows]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["team_abbrev"] = df["team_abbrev"].astype(str).str.upper().str.strip()
        df["date_snapshot"] = pd.to_datetime(df["date_snapshot"], errors="coerce").dt.strftime("%Y-%m-%d")

    meta = {
        "date": date_str,
        "rows": int(len(df)),
        "status": "ok" if len(df) > 0 else "empty",
        "expected_teams": EXPECTED_TEAMS_PER_DAY,
        "is_complete_day": bool(len(df) == EXPECTED_TEAMS_PER_DAY),
        "url": url,
    }
    return df, meta


def compute_missing_dates(target_dates: list[str], existing: pd.DataFrame) -> tuple[list[str], list[str]]:
    if existing.empty:
        return target_dates, []

    counts = existing.groupby("date_snapshot")["team_abbrev"].nunique().to_dict()
    complete_dates = {date for date, count in counts.items() if int(count) == EXPECTED_TEAMS_PER_DAY}
    incomplete_dates = sorted(date for date, count in counts.items() if int(count) != EXPECTED_TEAMS_PER_DAY)

    to_fetch = sorted(date for date in target_dates if date not in complete_dates)
    return to_fetch, incomplete_dates


def merge_and_save(existing: pd.DataFrame, fresh: pd.DataFrame, output_path: Path) -> pd.DataFrame:
    combined = pd.concat([existing, fresh], ignore_index=True)
    if combined.empty:
        combined.to_csv(output_path, index=False)
        return combined

    combined["date_snapshot"] = pd.to_datetime(combined["date_snapshot"], errors="coerce")
    combined = combined[combined["date_snapshot"].notna()].copy()
    combined["date_snapshot"] = combined["date_snapshot"].dt.strftime("%Y-%m-%d")
    combined["team_abbrev"] = combined["team_abbrev"].astype(str).str.upper().str.strip()

    combined = (
        combined.sort_values(["date_snapshot", "team_abbrev"]).drop_duplicates(
            subset=["date_snapshot", "team_abbrev"], keep="last"
        )
    )

    numeric_cols = [
        "games_played",
        "games_remaining",
        "points",
        "point_pctg",
        "wins",
        "losses",
        "ot_losses",
        "regulation_wins",
        "regulation_plus_ot_wins",
        "goal_for",
        "goal_against",
        "goal_differential",
        "league_sequence",
        "conference_sequence",
        "division_sequence",
        "wildcard_sequence",
        "waivers_sequence",
        "streak_count",
        "l10_games_played",
        "l10_wins",
        "l10_losses",
        "l10_ot_losses",
        "l10_points",
        "home_games_played",
        "home_points",
        "road_games_played",
        "road_points",
    ]
    for col in numeric_cols:
        if col in combined.columns:
            combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined = combined.sort_values(["date_snapshot", "conference_abbrev", "division_abbrev", "team_abbrev"])
    combined.to_csv(output_path, index=False)
    return combined


def write_summary(summary_path: Path, payload: dict) -> None:
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    config = parse_args()
    ensure_directories(config.output_path, config.summary_path)

    target_dates = load_target_dates_from_base_match(
        base_match_path=config.base_match_path,
        start_date=config.start_date,
        end_date=config.end_date,
    )

    if config.include_today:
        target_dates = sorted(set(target_dates + [today_paris_str()]))
    if config.extra_dates:
        target_dates = sorted(set(target_dates + normalize_date_strings(config.extra_dates)))

    if not target_dates:
        raise ValueError("Aucune date cible déterminée pour le refresh standings.")

    existing = pd.DataFrame() if config.rebuild else load_existing_output(config.output_path)
    to_fetch, incomplete_existing_dates = compute_missing_dates(target_dates, existing)

    fetched_frames: list[pd.DataFrame] = []
    fetch_log: list[dict] = []
    session = make_session()

    for idx, date_str in enumerate(to_fetch, start=1):
        df_day, meta = fetch_standings_for_date(session, date_str)
        fetched_frames.append(df_day)
        fetch_log.append(meta)
        print(f"[{idx}/{len(to_fetch)}] standings {date_str} -> {len(df_day)} lignes")
        if config.sleep_seconds > 0:
            time.sleep(config.sleep_seconds)

    fresh = pd.concat(fetched_frames, ignore_index=True) if fetched_frames else pd.DataFrame()

    if not existing.empty and incomplete_existing_dates:
        existing = existing[~existing["date_snapshot"].isin(incomplete_existing_dates)].copy()

    combined = merge_and_save(existing, fresh, config.output_path)

    counts_by_date = (
        combined.groupby("date_snapshot")["team_abbrev"].nunique().sort_index().to_dict()
        if not combined.empty
        else {}
    )
    incomplete_after = sorted(date for date, count in counts_by_date.items() if int(count) != EXPECTED_TEAMS_PER_DAY)

    summary = {
        "status": "ok",
        "api_template": NHL_STANDINGS_URL,
        "base_match_path": str(config.base_match_path),
        "output_path": str(config.output_path),
        "summary_path": str(config.summary_path),
        "target_dates_count": int(len(target_dates)),
        "target_dates_min": min(target_dates) if target_dates else None,
        "target_dates_max": max(target_dates) if target_dates else None,
        "include_today": bool(config.include_today),
        "rebuild": bool(config.rebuild),
        "sleep_seconds": float(config.sleep_seconds),
        "rows_existing_before": int(len(existing)),
        "dates_requested_count": int(len(to_fetch)),
        "dates_requested_min": min(to_fetch) if to_fetch else None,
        "dates_requested_max": max(to_fetch) if to_fetch else None,
        "dates_incomplete_existing_before_refetch": incomplete_existing_dates,
        "rows_fetched": int(len(fresh)),
        "rows_output": int(len(combined)),
        "dates_output_count": int(len(counts_by_date)),
        "dates_incomplete_after": incomplete_after,
        "teams_per_complete_day_expected": EXPECTED_TEAMS_PER_DAY,
        "fetch_log": fetch_log,
    }
    write_summary(config.summary_path, summary)

    print("00c_refresh_team_standings.py")
    print(f"Base match   : {config.base_match_path}")
    print(f"Output CSV   : {config.output_path}")
    print(f"Summary JSON : {config.summary_path}")
    print(f"Dates cibles : {len(target_dates)}")
    print(f"Dates fetch  : {len(to_fetch)}")
    print(f"Rows output  : {len(combined)}")
    if incomplete_after:
        print(f"Dates incomplètes restantes : {len(incomplete_after)}")


if __name__ == "__main__":
    main()
