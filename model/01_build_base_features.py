from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
FINAL_DIR = DATA_DIR / "final"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

INPUT_BASE_MATCH_FUSIONNEE = RAW_DIR / "base_match_fusionnee.csv"
INPUT_PP_STATS_GAME = RAW_DIR / "pp_stats_game.csv"
INPUT_TEAM_STANDINGS = RAW_DIR / "team_standings_daily.csv"

OUTPUT_BASE_CANONIQUE = FINAL_DIR / "base_canonique_v2.csv"
OUTPUT_BASE_FEATURES = FINAL_DIR / "base_features_v2.csv"
OUTPUT_BASE_FEATURES_CONTEXT = FINAL_DIR / "base_features_context_v2.csv"
SUMMARY_PATH = OUTPUTS_DIR / "01_build_base_features_summary.json"

DEFAULT_TEAM_GF = 2.8
DEFAULT_TEAM_GA = 2.8
DEFAULT_TEAM_WINRATE = 0.5
DEFAULT_LAST10_HIT_RATE = 0.45
DEFAULT_LAST20_HIT_RATE = 0.45
DEFAULT_SEASON_HIT_RATE = 0.45
DEFAULT_POINTS_PER_GAME = 0.55
DEFAULT_WILDCARD_DISTANCE = 0.0
DEFAULT_CONFERENCE_RANK = 8.5
DEFAULT_DIVISION_RANK = 4.5
DEFAULT_GAMES_REMAINING = 82.0


TEAM_CODE_ALIASES = {
    "ARI": "UTA",
    "PHX": "UTA",
    "PHO": "UTA",
}


def ensure_directories() -> None:
    INTERMEDIATE_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def verifier_colonnes(df: pd.DataFrame, colonnes: list[str]) -> None:
    manquantes = [c for c in colonnes if c not in df.columns]
    if manquantes:
        raise ValueError(f"Colonnes manquantes : {manquantes}")


def rolling_mean_shifted(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    return (
        df.groupby(group_col)[value_col]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).mean())
    )


def rolling_count_shifted(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    return (
        df.groupby(group_col)[value_col]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).count())
    )


def rolling_mean_shifted_group(
    df: pd.DataFrame,
    group_cols: list[str],
    value_col: str,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    return (
        df.groupby(group_cols)[value_col]
        .transform(lambda s: s.shift(1).rolling(window=window, min_periods=min_periods).mean())
    )


def safe_divide(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")

    out = pd.Series(0.0, index=num.index, dtype=float)
    mask = num.notna() & den.notna() & (den > 0)
    out.loc[mask] = num.loc[mask] / den.loc[mask]
    return out


def safe_ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    num = pd.to_numeric(num, errors="coerce")
    den = pd.to_numeric(den, errors="coerce")

    out = pd.Series(np.nan, index=num.index, dtype=float)
    mask = num.notna() & den.notna() & (den != 0)
    out.loc[mask] = num.loc[mask] / den.loc[mask]
    out = out.replace([np.inf, -np.inf], np.nan)
    return out




def coerce_binary_flag(value) -> float:
    """Coerce legacy boolean-ish values to 1.0 / 0.0.

    Supports ints/floats/bools and strings like True/False, 1/0, yes/no.
    Returns np.nan when the value cannot be interpreted.
    """
    if pd.isna(value):
        return np.nan

    if isinstance(value, (bool, np.bool_)):
        return 1.0 if bool(value) else 0.0

    if isinstance(value, (int, float, np.integer, np.floating)):
        if pd.isna(value):
            return np.nan
        if float(value) == 1.0:
            return 1.0
        if float(value) == 0.0:
            return 0.0
        return float(value)

    txt = str(value).strip().lower()
    if txt in {"1", "1.0", "true", "t", "yes", "y", "ok"}:
        return 1.0
    if txt in {"0", "0.0", "false", "f", "no", "n"}:
        return 0.0

    parsed = pd.to_numeric(pd.Series([txt]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else np.nan


def parse_mmss_to_minutes(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan

    if ":" in s:
        parts = s.split(":")
        try:
            if len(parts) == 2:
                mm, ss = parts
                return float(mm) + float(ss) / 60.0
            if len(parts) == 3:
                hh, mm, ss = parts
                return float(hh) * 60.0 + float(mm) + float(ss) / 60.0
        except ValueError:
            return np.nan

    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_pp_time_to_minutes_from_stats(value):
    if pd.isna(value):
        return np.nan

    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) / 60.0

    s = str(value).strip()
    if s == "" or s.lower() in {"nan", "none", "null"}:
        return np.nan

    if ":" in s:
        return parse_mmss_to_minutes(s)

    try:
        return float(s) / 60.0
    except ValueError:
        return np.nan


def normalize_team_code(value) -> str:
    if pd.isna(value):
        return ""
    code = str(value).strip().upper()
    return TEAM_CODE_ALIASES.get(code, code)


def parse_season_start_year(value) -> float:
    if pd.isna(value):
        return np.nan
    s = str(value).strip()
    if len(s) < 4:
        return np.nan
    try:
        return float(int(s[:4]))
    except ValueError:
        return np.nan


def previous_season_code(value) -> str | None:
    if pd.isna(value):
        return None
    s = str(value).strip()
    if len(s) < 8:
        return None
    try:
        start = int(s[:4]) - 1
        end = int(s[4:8]) - 1
        return f"{start}{end}"
    except ValueError:
        return None


def calc_consecutive_away(is_home_series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(is_home_series, errors="coerce").fillna(1).astype(int).tolist()
    out: list[int] = []
    c = 0
    for is_home in vals:
        if is_home == 0:
            c += 1
        else:
            c = 0
        out.append(c)
    return pd.Series(out, index=is_home_series.index)


def compute_streak_window_features(player_df: pd.DataFrame) -> pd.DataFrame:
    player_df = player_df.sort_values(["date_match", "id_match"]).copy()

    hits = pd.to_numeric(player_df["a_marque_un_point"], errors="coerce").fillna(0).astype(int).tolist()
    seasons = player_df["season_start_year"].astype("Int64").tolist()

    n = len(player_df)
    current_point_streak_pre = [0] * n
    current_no_point_streak_pre = [0] * n
    max_point_streak_last2_pre = [0] * n
    max_no_point_streak_last2_pre = [0] * n
    count_5plus_point_streaks_last2_pre = [0] * n
    count_matches_last2_pre = [0] * n

    prev_season = None
    running_point_streak = 0
    running_no_point_streak = 0

    def scan_streaks(values: list[int]) -> tuple[int, int, int]:
        max_hit = 0
        max_no = 0
        count_5plus = 0
        hit_run = 0
        no_run = 0
        for value in values:
            if value == 1:
                hit_run += 1
                no_run = 0
            else:
                no_run += 1
                hit_run = 0
            if hit_run > max_hit:
                max_hit = hit_run
            if no_run > max_no:
                max_no = no_run
            if hit_run == 5:
                count_5plus += 1
        return max_hit, max_no, count_5plus

    for i in range(n):
        current_season = seasons[i]
        if prev_season is None or current_season != prev_season:
            running_point_streak = 0
            running_no_point_streak = 0

        current_point_streak_pre[i] = running_point_streak
        current_no_point_streak_pre[i] = running_no_point_streak

        if pd.isna(current_season):
            start_idx = 0
        else:
            min_season = int(current_season) - 1
            start_idx = 0
            while start_idx < i:
                season_val = seasons[start_idx]
                if pd.isna(season_val) or int(season_val) >= min_season:
                    break
                start_idx += 1

        previous_values = hits[start_idx:i]
        count_matches_last2_pre[i] = len(previous_values)
        if previous_values:
            max_hit, max_no, count_5plus = scan_streaks(previous_values)
            max_point_streak_last2_pre[i] = max_hit
            max_no_point_streak_last2_pre[i] = max_no
            count_5plus_point_streaks_last2_pre[i] = count_5plus

        if hits[i] == 1:
            running_point_streak += 1
            running_no_point_streak = 0
        else:
            running_no_point_streak += 1
            running_point_streak = 0
        prev_season = current_season

    return pd.DataFrame(
        {
            "current_point_streak_pre": current_point_streak_pre,
            "current_no_point_streak_pre": current_no_point_streak_pre,
            "max_point_streak_last_2_seasons_pre": max_point_streak_last2_pre,
            "max_no_point_streak_last_2_seasons_pre": max_no_point_streak_last2_pre,
            "count_5plus_point_streaks_last_2_seasons_pre": count_5plus_point_streaks_last2_pre,
            "count_matches_last_2_seasons_pre": count_matches_last2_pre,
        },
        index=player_df.index,
    )


def charger_pp_stats_officiels(path_pp: Path) -> tuple[pd.DataFrame, dict]:
    if not path_pp.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path_pp}\n"
            "Place pp_stats_game.csv dans data/raw/. "
            "Ce fichier n'est pas poussé sur GitHub et sert désormais de source officielle pour temps_pp."
        )

    pp = pd.read_csv(path_pp, low_memory=False)
    verifier_colonnes(pp, ["id_joueur", "id_match", "ppTimeOnIce"])

    pp = pp.copy()
    pp["id_joueur"] = pd.to_numeric(pp["id_joueur"], errors="coerce")
    pp["id_match"] = pd.to_numeric(pp["id_match"], errors="coerce")

    nb_rows_before = int(len(pp))
    pp = pp[pp["id_joueur"].notna() & pp["id_match"].notna()].copy()
    nb_rows_after_keys = int(len(pp))
    nb_rows_invalid_keys = nb_rows_before - nb_rows_after_keys

    nb_duplicates = int(pp.duplicated(subset=["id_joueur", "id_match"]).sum())
    if nb_duplicates > 0:
        doublons = pp.loc[
            pp.duplicated(subset=["id_joueur", "id_match"], keep=False),
            ["id_joueur", "id_match"],
        ]
        exemple = doublons.head(10).to_dict(orient="records")
        raise ValueError(
            "Doublons détectés dans pp_stats_game.csv sur (id_joueur, id_match). "
            f"Exemples : {exemple}"
        )

    pp["temps_pp"] = pp["ppTimeOnIce"].apply(parse_pp_time_to_minutes_from_stats)
    nb_temps_pp_non_null = int(pp["temps_pp"].notna().sum())

    if nb_temps_pp_non_null == 0:
        raise ValueError(
            "Aucune valeur exploitable trouvée dans ppTimeOnIce. "
            "La source PP ne peut pas être utilisée telle quelle."
        )

    pp_for_merge = pp[["id_joueur", "id_match", "temps_pp"]].copy()

    summary = {
        "pp_file": str(path_pp),
        "pp_rows_read": nb_rows_before,
        "pp_rows_after_valid_keys": nb_rows_after_keys,
        "pp_rows_invalid_keys_dropped": nb_rows_invalid_keys,
        "pp_duplicate_key_rows": nb_duplicates,
        "pp_rows_with_temps_pp": nb_temps_pp_non_null,
    }
    return pp_for_merge, summary


def charger_standings(path_standings: Path) -> tuple[pd.DataFrame | None, dict]:
    if not path_standings.exists():
        return None, {
            "standings_file": str(path_standings),
            "standings_loaded": False,
            "reason": "file_missing",
        }

    standings = pd.read_csv(path_standings, low_memory=False)
    required = [
        "date_snapshot",
        "team_abbrev",
        "conference_abbrev",
        "division_abbrev",
        "games_played",
        "games_remaining",
        "points",
        "conference_sequence",
        "division_sequence",
    ]
    verifier_colonnes(standings, required)

    standings = standings.copy()
    standings["date_snapshot"] = pd.to_datetime(standings["date_snapshot"], errors="coerce")
    standings = standings[standings["date_snapshot"].notna()].copy()
    standings["team_abbrev"] = standings["team_abbrev"].apply(normalize_team_code)
    standings["conference_abbrev"] = standings["conference_abbrev"].astype(str).str.upper().str.strip()
    standings["division_abbrev"] = standings["division_abbrev"].astype(str).str.upper().str.strip()

    numeric_cols = [
        "games_played",
        "games_remaining",
        "points",
        "conference_sequence",
        "division_sequence",
        "wildcard_sequence",
        "point_pctg",
        "goal_differential",
        "l10_points",
    ]
    for col in numeric_cols:
        if col in standings.columns:
            standings[col] = pd.to_numeric(standings[col], errors="coerce")

    standings = standings.sort_values(["date_snapshot", "conference_abbrev", "conference_sequence", "team_abbrev"]).reset_index(drop=True)

    conf_cutoff = (
        standings.dropna(subset=["conference_abbrev"])  # type: ignore[arg-type]
        .groupby(["date_snapshot", "conference_abbrev"], dropna=False)[["conference_sequence", "points"]]
        .apply(_compute_conference_cutoff_points)
        .reset_index(name="conference_cutoff_points")
    )

    standings = standings.merge(
        conf_cutoff,
        on=["date_snapshot", "conference_abbrev"],
        how="left",
        validate="many_to_one",
    )
    standings["wildcard_distance"] = pd.to_numeric(standings["points"], errors="coerce") - pd.to_numeric(
        standings["conference_cutoff_points"],
        errors="coerce",
    )
    standings["standings_lookup_date"] = standings["date_snapshot"]
    standings = standings.sort_values(["team_abbrev", "standings_lookup_date"]).reset_index(drop=True)

    summary = {
        "standings_file": str(path_standings),
        "standings_loaded": True,
        "standings_rows": int(len(standings)),
        "standings_dates": int(standings["date_snapshot"].nunique()),
        "standings_teams": int(standings["team_abbrev"].nunique()),
        "standings_min_date": standings["date_snapshot"].min().strftime("%Y-%m-%d") if len(standings) else None,
        "standings_max_date": standings["date_snapshot"].max().strftime("%Y-%m-%d") if len(standings) else None,
    }
    return standings, summary


def _compute_conference_cutoff_points(group: pd.DataFrame) -> float:
    conf_seq = pd.to_numeric(group.get("conference_sequence"), errors="coerce")
    points = pd.to_numeric(group.get("points"), errors="coerce")

    mask = conf_seq.notna() & points.notna() & (conf_seq == 8)
    if mask.any():
        return float(points.loc[mask].iloc[0])

    points_sorted = points.dropna().sort_values(ascending=False).tolist()
    if len(points_sorted) >= 8:
        return float(points_sorted[7])
    if len(points_sorted) > 0:
        return float(points_sorted[-1])
    return np.nan


def charger_source_avec_pp() -> tuple[pd.DataFrame, dict]:
    if not INPUT_BASE_MATCH_FUSIONNEE.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {INPUT_BASE_MATCH_FUSIONNEE}\n"
            "Place base_match_fusionnee.csv dans data/raw/."
        )

    source = pd.read_csv(INPUT_BASE_MATCH_FUSIONNEE, low_memory=False)
    verifier_colonnes(source, ["id_joueur", "id_match"])

    source = source.copy()
    source["id_joueur"] = pd.to_numeric(source["id_joueur"], errors="coerce")
    source["id_match"] = pd.to_numeric(source["id_match"], errors="coerce")

    source = source.drop(columns=["temps_pp"], errors="ignore")

    pp_for_merge, pp_summary = charger_pp_stats_officiels(INPUT_PP_STATS_GAME)

    source = source.merge(
        pp_for_merge,
        on=["id_joueur", "id_match"],
        how="left",
        validate="many_to_one",
    )

    pp_found_mask = source["temps_pp"].notna()
    pp_rows_merged = int(pp_found_mask.sum())
    pp_rows_missing_after_merge = int((~pp_found_mask).sum())
    pp_merge_coverage = float(pp_found_mask.mean()) if len(source) > 0 else 0.0

    source["temps_pp"] = source["temps_pp"].fillna(0.0)

    merge_summary = {
        "input_base_file": str(INPUT_BASE_MATCH_FUSIONNEE),
        "input_rows": int(len(source)),
        "pp_rows_merged": pp_rows_merged,
        "pp_rows_missing_after_merge": pp_rows_missing_after_merge,
        "pp_merge_coverage": pp_merge_coverage,
    }
    merge_summary.update(pp_summary)

    return source, merge_summary


def build_base_canonique(df: pd.DataFrame) -> pd.DataFrame:
    colonnes_obligatoires = [
        "id_joueur",
        "id_match",
        "date_match",
        "season_source",
        "team_player_match",
        "adversaire_match",
        "is_home_player",
        "buts",
        "passes",
        "points",
        "tirs",
        "temps_de_glace",
        "temps_pp",
        "plus_moins",
        "penalty_minutes",
        "id_equipe_domicile",
        "id_equipe_exterieur",
        "buts_domicile",
        "buts_exterieur",
        "status",
        "match_trouve",
        "check_team_ok",
        "check_opp_ok",
    ]
    verifier_colonnes(df, colonnes_obligatoires)

    base = df.copy()
    base["date_match"] = pd.to_datetime(base["date_match"], errors="coerce")

    cols_num = [
        "id_joueur",
        "id_match",
        "is_home_player",
        "buts",
        "passes",
        "points",
        "tirs",
        "plus_moins",
        "penalty_minutes",
        "buts_domicile",
        "buts_exterieur",
    ]
    for c in cols_num:
        base[c] = pd.to_numeric(base[c], errors="coerce")

    for c in ["temps_de_glace", "temps_pp"]:
        base[c] = base[c].apply(parse_mmss_to_minutes)

    for c in [
        "season_source",
        "team_player_match",
        "adversaire_match",
        "id_equipe_domicile",
        "id_equipe_exterieur",
    ]:
        base[c] = base[c].astype(str).str.strip().str.upper()

    base["team_player_match"] = base["team_player_match"].apply(normalize_team_code)
    base["adversaire_match"] = base["adversaire_match"].apply(normalize_team_code)
    base["id_equipe_domicile"] = base["id_equipe_domicile"].apply(normalize_team_code)
    base["id_equipe_exterieur"] = base["id_equipe_exterieur"].apply(normalize_team_code)
    base["season_start_year"] = base["season_source"].apply(parse_season_start_year)

    # Legacy files can store these flags as booleans/strings ("True"/"False") instead of 1/0.
    base["match_trouve_flag"] = base["match_trouve"].apply(coerce_binary_flag)
    base["check_team_ok_flag"] = base["check_team_ok"].apply(coerce_binary_flag)
    base["check_opp_ok_flag"] = base["check_opp_ok"].apply(coerce_binary_flag)

    mask_match_bad = base["match_trouve_flag"].isna() | (base["match_trouve_flag"] != 1.0)
    mask_team_bad = base["check_team_ok_flag"].isna() | (base["check_team_ok_flag"] != 1.0)
    mask_opp_bad = base["check_opp_ok_flag"].isna() | (base["check_opp_ok_flag"] != 1.0)

    nb_match_non_trouve = int(mask_match_bad.sum())
    nb_team_bad = int(mask_team_bad.sum())
    nb_opp_bad = int(mask_opp_bad.sum())

    if nb_match_non_trouve > 0:
        sample = base.loc[mask_match_bad, ["id_match", "id_joueur", "match_trouve"]].head(10).to_dict("records")
        raise ValueError(f"Il reste des matchs non rattachés ({nb_match_non_trouve}). Exemples: {sample}")
    if nb_team_bad > 0 or nb_opp_bad > 0:
        sample = base.loc[mask_team_bad | mask_opp_bad, ["id_match", "id_joueur", "check_team_ok", "check_opp_ok"]].head(10).to_dict("records")
        raise ValueError(f"Incohérence équipe/adversaire détectée (team_bad={nb_team_bad}, opp_bad={nb_opp_bad}). Exemples: {sample}")

    base["a_marque_un_point"] = (base["points"] >= 1).astype(int)
    base["a_marque_un_but"] = (base["buts"] >= 1).astype(int)

    base["buts_match_equipe"] = np.where(
        base["is_home_player"] == 1,
        base["buts_domicile"],
        base["buts_exterieur"],
    )
    base["buts_match_adversaire"] = np.where(
        base["is_home_player"] == 1,
        base["buts_exterieur"],
        base["buts_domicile"],
    )
    base["victoire_equipe"] = (base["buts_match_equipe"] > base["buts_match_adversaire"]).astype(int)
    base["defaite_equipe"] = (base["buts_match_equipe"] < base["buts_match_adversaire"]).astype(int)
    base["diff_buts_equipe"] = base["buts_match_equipe"] - base["buts_match_adversaire"]

    colonnes_drop = [
        "_merge",
        "team_attendue_match",
        "adversaire_attendu_match",
        "check_team_ok",
        "check_opp_ok",
        "match_trouve_flag",
        "check_team_ok_flag",
        "check_opp_ok_flag",
    ]
    colonnes_drop = [c for c in colonnes_drop if c in base.columns]
    base = base.drop(columns=colonnes_drop, errors="ignore")

    base = (
        base.drop_duplicates(subset=["id_joueur", "id_match"])
        .sort_values(["id_joueur", "date_match", "id_match"], na_position="last")
        .reset_index(drop=True)
    )

    return base


def creer_features_temporelles_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    colonnes_obligatoires = [
        "id_joueur",
        "id_match",
        "date_match",
        "season_source",
        "season_start_year",
        "buts",
        "passes",
        "points",
        "tirs",
        "a_marque_un_point",
        "a_marque_un_but",
        "is_home_player",
        "team_player_match",
        "adversaire_match",
        "temps_de_glace",
        "temps_pp",
    ]
    verifier_colonnes(df, colonnes_obligatoires)

    df["date_match"] = pd.to_datetime(df["date_match"], errors="coerce")
    df["season_start_year"] = pd.to_numeric(df["season_start_year"], errors="coerce")

    cols_num = [
        "id_joueur",
        "id_match",
        "buts",
        "passes",
        "points",
        "tirs",
        "temps_de_glace",
        "temps_pp",
        "is_home_player",
        "a_marque_un_point",
        "a_marque_un_but",
    ]
    for c in cols_num:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["temps_de_glace", "temps_pp"]:
        df[c] = df[c].apply(parse_mmss_to_minutes)

    df = df.sort_values(["id_joueur", "date_match", "id_match"], na_position="last").reset_index(drop=True)

    same_season_prev = df.groupby("id_joueur")["season_source"].shift(1).eq(df["season_source"])

    df["nb_matchs_avant_match"] = df.groupby("id_joueur").cumcount()
    df["season_games_before_match"] = df.groupby(["id_joueur", "season_source"]).cumcount()

    df["jours_repos_raw"] = df.groupby("id_joueur")["date_match"].diff().dt.days
    df.loc[~same_season_prev, "jours_repos_raw"] = np.nan
    df["is_premier_match_joueur"] = (df["nb_matchs_avant_match"] == 0).astype(int)
    df["is_premier_match_saison_joueur"] = (df["season_games_before_match"] == 0).astype(int)

    df["jours_repos"] = df["jours_repos_raw"].clip(lower=0, upper=14)
    df["jours_repos"] = df["jours_repos"].fillna(7)

    df["tirs_moy_5"] = rolling_mean_shifted(df, "id_joueur", "tirs", 5)
    df["toi_moy_5"] = rolling_mean_shifted(df, "id_joueur", "temps_de_glace", 5)
    df["pp_moy_5"] = rolling_mean_shifted(df, "id_joueur", "temps_pp", 5)
    df["points_moy_5"] = rolling_mean_shifted(df, "id_joueur", "points", 5)
    df["buts_moy_5"] = rolling_mean_shifted(df, "id_joueur", "buts", 5)
    df["passes_moy_5"] = rolling_mean_shifted(df, "id_joueur", "passes", 5)

    df["tirs_moy_10"] = rolling_mean_shifted(df, "id_joueur", "tirs", 10)
    df["toi_moy_10"] = rolling_mean_shifted(df, "id_joueur", "temps_de_glace", 10)
    df["points_moy_10"] = rolling_mean_shifted(df, "id_joueur", "points", 10)
    df["buts_moy_10"] = rolling_mean_shifted(df, "id_joueur", "buts", 10)

    df["nb_matchs_joues_10"] = rolling_count_shifted(df, "id_joueur", "id_match", 10)

    df["point_hit_rate_last_5"] = rolling_mean_shifted(df, "id_joueur", "a_marque_un_point", 5)
    df["point_hit_rate_last_10"] = rolling_mean_shifted(df, "id_joueur", "a_marque_un_point", 10)
    df["point_hit_rate_last_20"] = rolling_mean_shifted(df, "id_joueur", "a_marque_un_point", 20)
    df["goal_hit_rate_last_10"] = rolling_mean_shifted(df, "id_joueur", "a_marque_un_but", 10)

    df["hist_ok_5"] = (df["nb_matchs_avant_match"] >= 5).astype(int)
    df["hist_ok_10"] = (df["nb_matchs_avant_match"] >= 10).astype(int)
    df["hist_ok_20"] = (df["nb_matchs_avant_match"] >= 20).astype(int)

    # Features intra-saison avant match
    player_season_group = df.groupby(["id_joueur", "season_source"], sort=False)
    df["season_point_hits_before_match"] = player_season_group["a_marque_un_point"].cumsum() - df["a_marque_un_point"]
    df["season_goal_hits_before_match"] = player_season_group["a_marque_un_but"].cumsum() - df["a_marque_un_but"]
    df["season_points_before_match"] = player_season_group["points"].cumsum() - df["points"]
    df["season_toi_before_match"] = player_season_group["temps_de_glace"].cumsum() - df["temps_de_glace"]
    df["season_pp_before_match"] = player_season_group["temps_pp"].cumsum() - df["temps_pp"]

    df["point_hit_rate_season_pre"] = safe_ratio(
        df["season_point_hits_before_match"],
        df["season_games_before_match"],
    )
    df["goal_hit_rate_season_pre"] = safe_ratio(
        df["season_goal_hits_before_match"],
        df["season_games_before_match"],
    )
    df["points_per_game_season_pre"] = safe_ratio(
        df["season_points_before_match"],
        df["season_games_before_match"],
    )
    df["toi_moy_season_pre"] = safe_ratio(
        df["season_toi_before_match"],
        df["season_games_before_match"],
    )
    df["pp_moy_season_pre"] = safe_ratio(
        df["season_pp_before_match"],
        df["season_games_before_match"],
    )

    # Features saison précédente complète
    season_summary = (
        df.groupby(["id_joueur", "season_source"], as_index=False)
        .agg(
            prev_full_games=("id_match", "count"),
            prev_full_point_hit_rate=("a_marque_un_point", "mean"),
            prev_full_goal_hit_rate=("a_marque_un_but", "mean"),
            prev_full_points_per_game=("points", "mean"),
            prev_full_toi_moy=("temps_de_glace", "mean"),
            prev_full_pp_moy=("temps_pp", "mean"),
        )
    )
    season_summary = season_summary.rename(columns={"season_source": "prev_season_source"})
    df["prev_season_source"] = df["season_source"].apply(previous_season_code)
    df = df.merge(
        season_summary,
        on=["id_joueur", "prev_season_source"],
        how="left",
        validate="many_to_one",
    )
    df = df.rename(
        columns={
            "prev_full_games": "prev_season_games",
            "prev_full_point_hit_rate": "point_hit_rate_prev_season",
            "prev_full_goal_hit_rate": "goal_hit_rate_prev_season",
            "prev_full_points_per_game": "points_per_game_prev_season",
            "prev_full_toi_moy": "toi_moy_prev_season",
            "prev_full_pp_moy": "pp_moy_prev_season",
        }
    )

    streak_parts = []
    for _, player_df in df.groupby("id_joueur", sort=False):
        streak_parts.append(compute_streak_window_features(player_df))
    streak_features = pd.concat(streak_parts).sort_index()
    for col in streak_features.columns:
        df[col] = streak_features[col]

    df["recent_hit_rate_composite"] = (
        0.6 * df["point_hit_rate_last_10"].fillna(df["point_hit_rate_last_20"])
        + 0.4 * df["point_hit_rate_last_20"].fillna(df["point_hit_rate_last_10"])
    )

    rolling_cols = [
        "tirs_moy_5",
        "toi_moy_5",
        "pp_moy_5",
        "points_moy_5",
        "buts_moy_5",
        "passes_moy_5",
        "tirs_moy_10",
        "toi_moy_10",
        "points_moy_10",
        "buts_moy_10",
        "nb_matchs_joues_10",
        "point_hit_rate_last_5",
        "point_hit_rate_last_10",
        "point_hit_rate_last_20",
        "goal_hit_rate_last_10",
        "season_point_hits_before_match",
        "season_goal_hits_before_match",
        "season_points_before_match",
        "season_toi_before_match",
        "season_pp_before_match",
        "point_hit_rate_season_pre",
        "goal_hit_rate_season_pre",
        "points_per_game_season_pre",
        "toi_moy_season_pre",
        "pp_moy_season_pre",
        "point_hit_rate_prev_season",
        "goal_hit_rate_prev_season",
        "points_per_game_prev_season",
        "toi_moy_prev_season",
        "pp_moy_prev_season",
        "prev_season_games",
        "recent_hit_rate_composite",
        "current_point_streak_pre",
        "current_no_point_streak_pre",
        "max_point_streak_last_2_seasons_pre",
        "max_no_point_streak_last_2_seasons_pre",
        "count_5plus_point_streaks_last_2_seasons_pre",
        "count_matches_last_2_seasons_pre",
    ]
    for c in rolling_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    fill_zero_cols = [
        "tirs_moy_5",
        "toi_moy_5",
        "pp_moy_5",
        "points_moy_5",
        "buts_moy_5",
        "passes_moy_5",
        "tirs_moy_10",
        "toi_moy_10",
        "points_moy_10",
        "buts_moy_10",
        "nb_matchs_joues_10",
        "season_point_hits_before_match",
        "season_goal_hits_before_match",
        "season_points_before_match",
        "season_toi_before_match",
        "season_pp_before_match",
        "current_point_streak_pre",
        "current_no_point_streak_pre",
        "max_point_streak_last_2_seasons_pre",
        "max_no_point_streak_last_2_seasons_pre",
        "count_5plus_point_streaks_last_2_seasons_pre",
        "count_matches_last_2_seasons_pre",
        "prev_season_games",
    ]
    for c in fill_zero_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    hit_rate_defaults = {
        "point_hit_rate_last_5": DEFAULT_LAST10_HIT_RATE,
        "point_hit_rate_last_10": DEFAULT_LAST10_HIT_RATE,
        "point_hit_rate_last_20": DEFAULT_LAST20_HIT_RATE,
        "goal_hit_rate_last_10": 0.20,
        "point_hit_rate_season_pre": DEFAULT_SEASON_HIT_RATE,
        "goal_hit_rate_season_pre": 0.20,
        "points_per_game_season_pre": DEFAULT_POINTS_PER_GAME,
        "toi_moy_season_pre": 15.0,
        "pp_moy_season_pre": 1.5,
        "point_hit_rate_prev_season": DEFAULT_SEASON_HIT_RATE,
        "goal_hit_rate_prev_season": 0.20,
        "points_per_game_prev_season": DEFAULT_POINTS_PER_GAME,
        "toi_moy_prev_season": 15.0,
        "pp_moy_prev_season": 1.5,
        "recent_hit_rate_composite": DEFAULT_LAST10_HIT_RATE,
    }
    for col, val in hit_rate_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    df["tirs_par_60_5"] = safe_divide(df["tirs_moy_5"] * 60.0, df["toi_moy_5"])
    df["points_par_60_5"] = safe_divide(df["points_moy_5"] * 60.0, df["toi_moy_5"])
    df["buts_par_60_5"] = safe_divide(df["buts_moy_5"] * 60.0, df["toi_moy_5"])

    return df


def enrichir_contexte_v2(
    df: pd.DataFrame,
    standings: pd.DataFrame | None = None,
    config: dict | None = None,
) -> pd.DataFrame:
    df = df.copy()
    config = config or {}

    colonnes_obligatoires = [
        "id_joueur",
        "id_match",
        "date_match",
        "season_source",
        "season_start_year",
        "team_player_match",
        "adversaire_match",
        "is_home_player",
        "id_equipe_domicile",
        "id_equipe_exterieur",
        "buts_domicile",
        "buts_exterieur",
        "points",
        "buts",
        "tirs",
        "temps_de_glace",
        "temps_pp",
        "points_moy_10",
        "buts_moy_10",
        "tirs_moy_10",
        "pp_moy_5",
        "toi_moy_10",
        "point_hit_rate_last_10",
        "point_hit_rate_last_20",
        "point_hit_rate_season_pre",
        "point_hit_rate_prev_season",
        "points_per_game_season_pre",
        "points_per_game_prev_season",
        "current_point_streak_pre",
        "current_no_point_streak_pre",
        "max_point_streak_last_2_seasons_pre",
        "max_no_point_streak_last_2_seasons_pre",
        "count_5plus_point_streaks_last_2_seasons_pre",
    ]
    verifier_colonnes(df, colonnes_obligatoires)

    seuil_absence_jours = config.get("seuil_absence_longue_jours", 10)
    ratio_retour_toi_min = config.get("ratio_retour_toi_min", 0.75)
    toi_min_retour = config.get("temps_glace_min_retour", 15.0)
    current_weight_default = config.get("historical_weight_current_season_default", 0.60)
    current_weight_return = config.get("historical_weight_current_season_return", 0.80)

    df["date_match"] = pd.to_datetime(df["date_match"], errors="coerce")

    cols_num = [
        "id_joueur",
        "id_match",
        "season_start_year",
        "is_home_player",
        "buts_domicile",
        "buts_exterieur",
        "points",
        "buts",
        "tirs",
        "temps_de_glace",
        "temps_pp",
        "points_moy_10",
        "buts_moy_10",
        "tirs_moy_10",
        "pp_moy_5",
        "toi_moy_10",
        "point_hit_rate_last_10",
        "point_hit_rate_last_20",
        "point_hit_rate_season_pre",
        "point_hit_rate_prev_season",
        "points_per_game_season_pre",
        "points_per_game_prev_season",
        "current_point_streak_pre",
        "current_no_point_streak_pre",
        "max_point_streak_last_2_seasons_pre",
        "max_no_point_streak_last_2_seasons_pre",
        "count_5plus_point_streaks_last_2_seasons_pre",
    ]
    for c in cols_num:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in [
        "team_player_match",
        "adversaire_match",
        "id_equipe_domicile",
        "id_equipe_exterieur",
        "season_source",
    ]:
        df[c] = df[c].astype(str).str.strip().str.upper()

    df["team_player_match"] = df["team_player_match"].apply(normalize_team_code)
    df["adversaire_match"] = df["adversaire_match"].apply(normalize_team_code)
    df["id_equipe_domicile"] = df["id_equipe_domicile"].apply(normalize_team_code)
    df["id_equipe_exterieur"] = df["id_equipe_exterieur"].apply(normalize_team_code)

    df = df.sort_values(["date_match", "id_match", "id_joueur"]).reset_index(drop=True)

    matchs_uniques = (
        df[
            [
                "id_match",
                "date_match",
                "season_source",
                "id_equipe_domicile",
                "id_equipe_exterieur",
                "buts_domicile",
                "buts_exterieur",
            ]
        ]
        .drop_duplicates(subset=["id_match"])
        .copy()
    )

    home_rows = matchs_uniques.rename(
        columns={
            "id_equipe_domicile": "team_code",
            "id_equipe_exterieur": "opp_code",
            "buts_domicile": "gf",
            "buts_exterieur": "ga",
        }
    )[["id_match", "date_match", "season_source", "team_code", "opp_code", "gf", "ga"]].copy()
    home_rows["is_home_team"] = 1

    away_rows = matchs_uniques.rename(
        columns={
            "id_equipe_exterieur": "team_code",
            "id_equipe_domicile": "opp_code",
            "buts_exterieur": "gf",
            "buts_domicile": "ga",
        }
    )[["id_match", "date_match", "season_source", "team_code", "opp_code", "gf", "ga"]].copy()
    away_rows["is_home_team"] = 0

    team_games = pd.concat([home_rows, away_rows], ignore_index=True)
    team_games["date_match"] = pd.to_datetime(team_games["date_match"], errors="coerce")
    team_games["team_code"] = team_games["team_code"].apply(normalize_team_code)
    team_games["opp_code"] = team_games["opp_code"].apply(normalize_team_code)
    team_games = team_games.sort_values(["team_code", "season_source", "date_match", "id_match"]).reset_index(drop=True)

    team_group = team_games.groupby(["team_code", "season_source"], sort=False)

    team_games["team_games_played_pre_approx"] = team_group.cumcount()
    team_games["jours_repos_team"] = team_group["date_match"].diff().dt.days
    team_games["jours_repos_team"] = team_games["jours_repos_team"].fillna(3).clip(lower=0, upper=14)

    team_games["team_back_to_back"] = (team_games["jours_repos_team"] <= 1).astype(int)
    team_games["team_back_to_back_away"] = (
        (team_games["team_back_to_back"] == 1) & (team_games["is_home_team"] == 0)
    ).astype(int)
    team_games["consecutive_away_games"] = team_group["is_home_team"].transform(calc_consecutive_away)

    team_games["team_win"] = (team_games["gf"] > team_games["ga"]).astype(int)
    team_games["team_winrate_5"] = team_group["team_win"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    team_games["team_gf_moy_5"] = team_group["gf"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )
    team_games["team_ga_moy_5"] = team_group["ga"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).mean()
    )

    team_context = team_games[
        [
            "id_match",
            "team_code",
            "opp_code",
            "is_home_team",
            "jours_repos_team",
            "team_back_to_back",
            "team_back_to_back_away",
            "consecutive_away_games",
            "team_winrate_5",
            "team_gf_moy_5",
            "team_ga_moy_5",
            "team_games_played_pre_approx",
        ]
    ].copy()

    dup_team_context = int(team_context.duplicated(subset=["id_match", "team_code"]).sum())
    if dup_team_context > 0:
        raise ValueError("Doublons détectés dans team_context sur (id_match, team_code)")

    df = df.merge(
        team_context,
        left_on=["id_match", "team_player_match"],
        right_on=["id_match", "team_code"],
        how="left",
        validate="many_to_one",
    )

    df["team_context_found"] = df["team_code"].notna().astype(int)
    df["check_home_team_context_ok"] = (
        pd.to_numeric(df["is_home_player"], errors="coerce")
        == pd.to_numeric(df["is_home_team"], errors="coerce")
    ).astype("float")

    df = df.sort_values(["id_joueur", "date_match", "id_match"]).reset_index(drop=True)

    df["nb_matchs_vs_adv_avant"] = df.groupby(["id_joueur", "adversaire_match"]).cumcount()
    df["points_vs_adv_5"] = rolling_mean_shifted_group(df, ["id_joueur", "adversaire_match"], "points", 5)
    df["buts_vs_adv_5"] = rolling_mean_shifted_group(df, ["id_joueur", "adversaire_match"], "buts", 5)
    df["tirs_vs_adv_5"] = rolling_mean_shifted_group(df, ["id_joueur", "adversaire_match"], "tirs", 5)

    k_shrink = 3.0
    points_emp = df["points_vs_adv_5"].fillna(df["points_moy_10"])
    buts_emp = df["buts_vs_adv_5"].fillna(df["buts_moy_10"])
    tirs_emp = df["tirs_vs_adv_5"].fillna(df["tirs_moy_10"])

    df["points_vs_adv_shrunk"] = (
        points_emp * df["nb_matchs_vs_adv_avant"] + df["points_moy_10"] * k_shrink
    ) / (df["nb_matchs_vs_adv_avant"] + k_shrink)
    df["buts_vs_adv_shrunk"] = (
        buts_emp * df["nb_matchs_vs_adv_avant"] + df["buts_moy_10"] * k_shrink
    ) / (df["nb_matchs_vs_adv_avant"] + k_shrink)
    df["tirs_vs_adv_shrunk"] = (
        tirs_emp * df["nb_matchs_vs_adv_avant"] + df["tirs_moy_10"] * k_shrink
    ) / (df["nb_matchs_vs_adv_avant"] + k_shrink)

    same_season_prev = df.groupby("id_joueur")["season_source"].shift(1).eq(df["season_source"])
    df["jours_absence_pre_match"] = df.groupby("id_joueur")["date_match"].diff().dt.days
    df.loc[~same_season_prev, "jours_absence_pre_match"] = np.nan
    df["jours_absence_pre_match"] = df["jours_absence_pre_match"].fillna(3).clip(lower=0, upper=90)

    df["games_missed_proxy"] = (df["jours_absence_pre_match"] - 4).clip(lower=0)
    df["absence_longue_flag"] = (
        same_season_prev.fillna(False) & (df["jours_absence_pre_match"] >= seuil_absence_jours)
    ).astype(int)
    df["retour_episode"] = df.groupby("id_joueur")["absence_longue_flag"].cumsum()
    df["matchs_depuis_retour_avant_match"] = df.groupby(["id_joueur", "retour_episode"]).cumcount()

    df["toi_pre_absence_ref"] = np.where(df["absence_longue_flag"] == 1, df["toi_moy_10"], np.nan)
    df["pp_pre_absence_ref"] = np.where(df["absence_longue_flag"] == 1, df["pp_moy_5"], np.nan)

    df["toi_pre_absence_ref"] = df.groupby(["id_joueur", "retour_episode"])["toi_pre_absence_ref"].transform(
        lambda s: s.ffill()
    )
    df["pp_pre_absence_ref"] = df.groupby(["id_joueur", "retour_episode"])["pp_pre_absence_ref"].transform(
        lambda s: s.ffill()
    )

    df["toi_moy_retour_3_avant_match"] = df.groupby(["id_joueur", "retour_episode"])["temps_de_glace"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    df["pp_moy_retour_3_avant_match"] = df.groupby(["id_joueur", "retour_episode"])["temps_pp"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

    df["ratio_toi_retour_vs_pre_absence"] = safe_ratio(df["toi_moy_retour_3_avant_match"], df["toi_pre_absence_ref"])
    df["ratio_pp_retour_vs_pre_absence"] = safe_ratio(df["pp_moy_retour_3_avant_match"], df["pp_pre_absence_ref"])

    df["return_from_absence_flag"] = (df["retour_episode"] > 0).astype(int)
    df["return_stabilized_flag"] = (
        (df["retour_episode"] > 0)
        & (df["matchs_depuis_retour_avant_match"] >= 3)
        & (df["toi_moy_retour_3_avant_match"].fillna(0) >= toi_min_retour)
        & (df["ratio_toi_retour_vs_pre_absence"].fillna(0) >= ratio_retour_toi_min)
    ).astype(int)

    # Pondération saison courante / saison précédente
    df["historical_current_weight"] = np.where(
        df["return_stabilized_flag"] == 1,
        current_weight_return,
        current_weight_default,
    )
    df["historical_prev_weight"] = 1.0 - df["historical_current_weight"]

    current_hit_base = df["point_hit_rate_season_pre"].copy()
    prev_hit_base = df["point_hit_rate_prev_season"].copy()
    current_ppg_base = df["points_per_game_season_pre"].copy()
    prev_ppg_base = df["points_per_game_prev_season"].copy()

    df["point_hit_rate_weighted_pre"] = np.where(
        current_hit_base.notna() & prev_hit_base.notna(),
        df["historical_current_weight"] * current_hit_base + df["historical_prev_weight"] * prev_hit_base,
        current_hit_base.fillna(prev_hit_base),
    )
    df["points_per_game_weighted_pre"] = np.where(
        current_ppg_base.notna() & prev_ppg_base.notna(),
        df["historical_current_weight"] * current_ppg_base + df["historical_prev_weight"] * prev_ppg_base,
        current_ppg_base.fillna(prev_ppg_base),
    )

    recent_combo = 0.6 * df["point_hit_rate_last_10"].fillna(df["point_hit_rate_last_20"]) + 0.4 * df[
        "point_hit_rate_last_20"
    ].fillna(df["point_hit_rate_last_10"])
    df["recent_vs_expected_gap"] = df["point_hit_rate_weighted_pre"].fillna(DEFAULT_SEASON_HIT_RATE) - recent_combo.fillna(
        DEFAULT_LAST10_HIT_RATE
    )

    df["hard_exclude_hot_streak_pre"] = (
        (df["current_point_streak_pre"] >= 5)
        & (df["count_5plus_point_streaks_last_2_seasons_pre"] < 2)
        & (df["current_point_streak_pre"] >= df["max_point_streak_last_2_seasons_pre"].fillna(0))
    ).astype(int)

    # Contexte standings / playoffs
    standings_summary = {
        "standings_loaded": False,
        "standings_merge_coverage": 0.0,
    }
    if standings is not None and len(standings) > 0:
        standings_lookup = standings.copy()
        standings_lookup["standings_lookup_date"] = pd.to_datetime(
            standings_lookup["standings_lookup_date"],
            errors="coerce",
        )
        standings_lookup = standings_lookup.sort_values(["standings_lookup_date", "team_abbrev"]).reset_index(drop=True)

        merge_left = df[["team_player_match", "date_match"]].copy()
        merge_left["lookup_date"] = pd.to_datetime(merge_left["date_match"], errors="coerce") - pd.Timedelta(days=1)
        merge_left = merge_left.rename(columns={"team_player_match": "team_abbrev"})
        merge_left = merge_left.sort_values(["lookup_date", "team_abbrev"]).reset_index(drop=False)

        standings_cols = [
            "team_abbrev",
            "standings_lookup_date",
            "conference_abbrev",
            "division_abbrev",
            "games_played",
            "games_remaining",
            "points",
            "conference_sequence",
            "division_sequence",
            "wildcard_sequence",
            "conference_cutoff_points",
            "wildcard_distance",
            "point_pctg",
            "goal_differential",
            "l10_points",
        ]
        standings_lookup = standings_lookup[standings_cols].copy()

        merged_standings = pd.merge_asof(
            merge_left,
            standings_lookup,
            left_on="lookup_date",
            right_on="standings_lookup_date",
            by="team_abbrev",
            direction="backward",
            allow_exact_matches=True,
        ).sort_values("index")

        merged_standings = merged_standings.drop(columns=["index"]).reset_index(drop=True)
        for col in [
            "conference_abbrev",
            "division_abbrev",
            "games_played",
            "games_remaining",
            "points",
            "conference_sequence",
            "division_sequence",
            "wildcard_sequence",
            "conference_cutoff_points",
            "wildcard_distance",
            "point_pctg",
            "goal_differential",
            "l10_points",
            "standings_lookup_date",
        ]:
            df[f"{col}_pre"] = merged_standings[col].values

        coverage = pd.to_numeric(df["points_pre"], errors="coerce").notna().mean() if len(df) else 0.0
        standings_summary = {
            "standings_loaded": True,
            "standings_merge_coverage": float(coverage),
        }
    else:
        for col in [
            "conference_abbrev_pre",
            "division_abbrev_pre",
            "games_played_pre",
            "games_remaining_pre",
            "points_pre",
            "conference_sequence_pre",
            "division_sequence_pre",
            "wildcard_sequence_pre",
            "conference_cutoff_points_pre",
            "wildcard_distance_pre",
            "point_pctg_pre",
            "goal_differential_pre",
            "l10_points_pre",
            "standings_lookup_date_pre",
        ]:
            df[col] = np.nan

    df["games_played_team_pre"] = pd.to_numeric(df.get("games_played_pre"), errors="coerce")
    df["games_played_team_pre"] = df["games_played_team_pre"].fillna(df["team_games_played_pre_approx"])

    df["games_remaining_team_pre"] = pd.to_numeric(df.get("games_remaining_pre"), errors="coerce")
    df["games_remaining_team_pre"] = df["games_remaining_team_pre"].fillna(82 - df["games_played_team_pre"])
    df["games_remaining_team_pre"] = df["games_remaining_team_pre"].clip(lower=0, upper=82)

    df["team_points_pre"] = pd.to_numeric(df.get("points_pre"), errors="coerce")
    df["conference_rank_pre"] = pd.to_numeric(df.get("conference_sequence_pre"), errors="coerce")
    df["division_rank_pre"] = pd.to_numeric(df.get("division_sequence_pre"), errors="coerce")
    df["wildcard_distance_pre"] = pd.to_numeric(df.get("wildcard_distance_pre"), errors="coerce")
    df["conference_cutoff_points_pre"] = pd.to_numeric(df.get("conference_cutoff_points_pre"), errors="coerce")

    df["late_season_flag"] = (df["games_remaining_team_pre"] <= 20).astype(int)

    df["playoff_pressure_simple"] = 0
    pressure_mask = df["late_season_flag"] == 1
    df.loc[pressure_mask & df["wildcard_distance_pre"].between(-4, 4, inclusive="both"), "playoff_pressure_simple"] = 1
    df.loc[pressure_mask & ((df["wildcard_distance_pre"] >= 8) | (df["wildcard_distance_pre"] <= -8)), "playoff_pressure_simple"] = -1
    df["playoff_pressure_simple"] = df["playoff_pressure_simple"].fillna(0).astype(int)

    fill_defaults = {
        "jours_repos_team": 3,
        "team_back_to_back": 0,
        "team_back_to_back_away": 0,
        "consecutive_away_games": 0,
        "team_winrate_5": DEFAULT_TEAM_WINRATE,
        "team_gf_moy_5": DEFAULT_TEAM_GF,
        "team_ga_moy_5": DEFAULT_TEAM_GA,
        "team_games_played_pre_approx": 0,
        "nb_matchs_vs_adv_avant": 0,
        "points_vs_adv_shrunk": df["points_moy_10"].median(),
        "buts_vs_adv_shrunk": df["buts_moy_10"].median(),
        "tirs_vs_adv_shrunk": df["tirs_moy_10"].median(),
        "jours_absence_pre_match": 3,
        "games_missed_proxy": 0,
        "matchs_depuis_retour_avant_match": 0,
        "return_from_absence_flag": 0,
        "return_stabilized_flag": 0,
        "ratio_toi_retour_vs_pre_absence": 0,
        "ratio_pp_retour_vs_pre_absence": 0,
        "point_hit_rate_weighted_pre": DEFAULT_SEASON_HIT_RATE,
        "points_per_game_weighted_pre": DEFAULT_POINTS_PER_GAME,
        "recent_vs_expected_gap": 0,
        "hard_exclude_hot_streak_pre": 0,
        "games_played_team_pre": 0,
        "games_remaining_team_pre": DEFAULT_GAMES_REMAINING,
        "team_points_pre": np.nan,
        "conference_rank_pre": DEFAULT_CONFERENCE_RANK,
        "division_rank_pre": DEFAULT_DIVISION_RANK,
        "conference_cutoff_points_pre": np.nan,
        "wildcard_distance_pre": DEFAULT_WILDCARD_DISTANCE,
        "late_season_flag": 0,
        "playoff_pressure_simple": 0,
    }
    for col, val in fill_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    # indicateurs utiles de disponibilité
    df["standings_context_found"] = pd.to_numeric(df["conference_rank_pre"], errors="coerce").notna().astype(int)
    if standings is None or len(standings) == 0:
        df["standings_context_found"] = 0

    drop_cols = [c for c in ["team_code", "opp_code"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    df.attrs["standings_summary"] = standings_summary
    return df


def write_summary(payload: dict) -> None:
    SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_directories()

    df_source, pp_summary = charger_source_avec_pp()
    standings_df, standings_file_summary = charger_standings(INPUT_TEAM_STANDINGS)

    base_canonique = build_base_canonique(df_source)
    base_features = creer_features_temporelles_v2(base_canonique)
    base_features_context = enrichir_contexte_v2(base_features, standings=standings_df, config={})
    standings_runtime_summary = base_features_context.attrs.get("standings_summary", {})

    base_canonique.to_csv(OUTPUT_BASE_CANONIQUE, index=False)
    base_features.to_csv(OUTPUT_BASE_FEATURES, index=False)
    base_features_context.to_csv(OUTPUT_BASE_FEATURES_CONTEXT, index=False)

    summary = {
        "status": "ok",
        "input_file": str(INPUT_BASE_MATCH_FUSIONNEE),
        "pp_file": str(INPUT_PP_STATS_GAME),
        "standings_file": str(INPUT_TEAM_STANDINGS),
        "outputs": [
            str(OUTPUT_BASE_CANONIQUE),
            str(OUTPUT_BASE_FEATURES),
            str(OUTPUT_BASE_FEATURES_CONTEXT),
        ],
        "rows_base_canonique": int(len(base_canonique)),
        "rows_base_features": int(len(base_features)),
        "rows_base_features_context": int(len(base_features_context)),
        "cols_base_canonique": int(len(base_canonique.columns)),
        "cols_base_features": int(len(base_features.columns)),
        "cols_base_features_context": int(len(base_features_context.columns)),
        "pp_integration": pp_summary,
        "standings_file_integration": standings_file_summary,
        "standings_runtime_integration": standings_runtime_summary,
        "new_feature_checks": {
            "late_season_flag_sum": int(pd.to_numeric(base_features_context.get("late_season_flag"), errors="coerce").fillna(0).sum()),
            "playoff_pressure_non_zero": int((pd.to_numeric(base_features_context.get("playoff_pressure_simple"), errors="coerce").fillna(0) != 0).sum()),
            "hot_streak_excludes": int(pd.to_numeric(base_features_context.get("hard_exclude_hot_streak_pre"), errors="coerce").fillna(0).sum()),
            "return_stabilized_rows": int(pd.to_numeric(base_features_context.get("return_stabilized_flag"), errors="coerce").fillna(0).sum()),
        },
    }
    write_summary(summary)

    print("01_build_base_features.py")
    print(f"Input base      : {INPUT_BASE_MATCH_FUSIONNEE}")
    print(f"Input PP        : {INPUT_PP_STATS_GAME}")
    print(f"Input standings : {INPUT_TEAM_STANDINGS}")
    print(f"Output          : {OUTPUT_BASE_CANONIQUE}")
    print(f"Output          : {OUTPUT_BASE_FEATURES}")
    print(f"Output          : {OUTPUT_BASE_FEATURES_CONTEXT}")
    print(f"Summary         : {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
