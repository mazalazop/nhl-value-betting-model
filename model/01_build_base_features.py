
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

OUTPUT_BASE_CANONIQUE = FINAL_DIR / "base_canonique_v2.csv"
OUTPUT_BASE_FEATURES = FINAL_DIR / "base_features_v2.csv"
OUTPUT_BASE_FEATURES_CONTEXT = FINAL_DIR / "base_features_context_v2.csv"
SUMMARY_PATH = OUTPUTS_DIR / "01_build_base_features_summary.json"


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


def calc_consecutive_away(is_home_series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(is_home_series, errors="coerce").fillna(1).astype(int).tolist()
    out = []
    c = 0
    for v in vals:
        if v == 0:
            c += 1
        else:
            c = 0
        out.append(c)
    return pd.Series(out, index=is_home_series.index)


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
        doublons = pp.loc[pp.duplicated(subset=["id_joueur", "id_match"], keep=False), ["id_joueur", "id_match"]]
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

    df = df.copy()
    df["date_match"] = pd.to_datetime(df["date_match"], errors="coerce")

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
        "match_trouve",
    ]
    for c in cols_num:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["temps_de_glace", "temps_pp"]:
        df[c] = df[c].apply(parse_mmss_to_minutes)

    nb_match_non_trouve = int((df["match_trouve"] != 1).sum())
    nb_team_bad = int((df["check_team_ok"] != True).sum())
    nb_opp_bad = int((df["check_opp_ok"] != True).sum())

    if nb_match_non_trouve > 0:
        raise ValueError("Il reste des matchs non rattachés")
    if nb_team_bad > 0 or nb_opp_bad > 0:
        raise ValueError("Incohérence équipe/adversaire détectée")

    base = df.copy()

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

    base["victoire_equipe"] = (
        base["buts_match_equipe"] > base["buts_match_adversaire"]
    ).astype(int)
    base["defaite_equipe"] = (
        base["buts_match_equipe"] < base["buts_match_adversaire"]
    ).astype(int)
    base["diff_buts_equipe"] = (
        base["buts_match_equipe"] - base["buts_match_adversaire"]
    )

    colonnes_drop = [
        "_merge",
        "team_attendue_match",
        "adversaire_attendu_match",
        "check_team_ok",
        "check_opp_ok",
    ]
    colonnes_drop = [c for c in colonnes_drop if c in base.columns]
    base = base.drop(columns=colonnes_drop, errors="ignore")

    base = (
        base.drop_duplicates(subset=["id_joueur", "id_match"])
        .sort_values(["date_match", "id_match", "id_joueur"], na_position="last")
        .reset_index(drop=True)
    )

    return base


def creer_features_temporelles_v2(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    colonnes_obligatoires = [
        "id_joueur",
        "id_match",
        "date_match",
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
    ]
    for c in cols_num:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in ["temps_de_glace", "temps_pp"]:
        df[c] = df[c].apply(parse_mmss_to_minutes)

    df = df.sort_values(["id_joueur", "date_match", "id_match"], na_position="last").reset_index(
        drop=True
    )

    df["nb_matchs_avant_match"] = df.groupby("id_joueur").cumcount()

    df["jours_repos_raw"] = df.groupby("id_joueur")["date_match"].diff().dt.days
    df["is_premier_match_joueur"] = (df["nb_matchs_avant_match"] == 0).astype(int)

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

    df["hist_ok_5"] = (df["nb_matchs_avant_match"] >= 5).astype(int)
    df["hist_ok_10"] = (df["nb_matchs_avant_match"] >= 10).astype(int)

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
    ]
    for c in rolling_cols:
        df[c] = df[c].fillna(0.0)

    df["tirs_par_60_5"] = safe_divide(df["tirs_moy_5"] * 60.0, df["toi_moy_5"])
    df["points_par_60_5"] = safe_divide(df["points_moy_5"] * 60.0, df["toi_moy_5"])
    df["buts_par_60_5"] = safe_divide(df["buts_moy_5"] * 60.0, df["toi_moy_5"])

    return df


def enrichir_contexte_v2(df: pd.DataFrame, config: dict | None = None) -> pd.DataFrame:
    df = df.copy()
    config = config or {}

    colonnes_obligatoires = [
        "id_joueur",
        "id_match",
        "date_match",
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
    ]
    verifier_colonnes(df, colonnes_obligatoires)

    seuil_absence_jours = config.get("seuil_absence_longue_jours", 10)
    ratio_retour_toi_min = config.get("ratio_retour_toi_min", 0.75)
    toi_min_retour = config.get("temps_glace_min", 8.0)

    df["date_match"] = pd.to_datetime(df["date_match"], errors="coerce")

    cols_num = [
        "id_joueur",
        "id_match",
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
    ]
    for c in cols_num:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in [
        "team_player_match",
        "adversaire_match",
        "id_equipe_domicile",
        "id_equipe_exterieur",
    ]:
        df[c] = df[c].astype(str).str.strip().str.upper()

    df = df.sort_values(["date_match", "id_match", "id_joueur"]).reset_index(drop=True)

    matchs_uniques = (
        df[
            [
                "id_match",
                "date_match",
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
    )[["id_match", "date_match", "team_code", "opp_code", "gf", "ga"]].copy()
    home_rows["is_home_team"] = 1

    away_rows = matchs_uniques.rename(
        columns={
            "id_equipe_exterieur": "team_code",
            "id_equipe_domicile": "opp_code",
            "buts_exterieur": "gf",
            "buts_domicile": "ga",
        }
    )[["id_match", "date_match", "team_code", "opp_code", "gf", "ga"]].copy()
    away_rows["is_home_team"] = 0

    team_games = pd.concat([home_rows, away_rows], ignore_index=True)
    team_games["date_match"] = pd.to_datetime(team_games["date_match"], errors="coerce")
    team_games = team_games.sort_values(["team_code", "date_match", "id_match"]).reset_index(
        drop=True
    )

    team_games["jours_repos_team"] = team_games.groupby("team_code")["date_match"].diff().dt.days
    team_games["jours_repos_team"] = team_games["jours_repos_team"].fillna(3).clip(lower=0, upper=14)

    team_games["team_back_to_back"] = (team_games["jours_repos_team"] <= 1).astype(int)
    team_games["team_back_to_back_away"] = (
        (team_games["team_back_to_back"] == 1) & (team_games["is_home_team"] == 0)
    ).astype(int)

    team_games["consecutive_away_games"] = (
        team_games.groupby("team_code")["is_home_team"].transform(calc_consecutive_away)
    )

    team_games["team_win"] = (team_games["gf"] > team_games["ga"]).astype(int)

    team_games["team_winrate_5"] = (
        team_games.groupby("team_code")["team_win"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    team_games["team_gf_moy_5"] = (
        team_games.groupby("team_code")["gf"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    )

    team_games["team_ga_moy_5"] = (
        team_games.groupby("team_code")["ga"]
        .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
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

    df["points_vs_adv_5"] = rolling_mean_shifted_group(
        df, ["id_joueur", "adversaire_match"], "points", 5
    )
    df["buts_vs_adv_5"] = rolling_mean_shifted_group(
        df, ["id_joueur", "adversaire_match"], "buts", 5
    )
    df["tirs_vs_adv_5"] = rolling_mean_shifted_group(
        df, ["id_joueur", "adversaire_match"], "tirs", 5
    )

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

    df["jours_absence_pre_match"] = df.groupby("id_joueur")["date_match"].diff().dt.days
    df["jours_absence_pre_match"] = df["jours_absence_pre_match"].fillna(3).clip(lower=0, upper=90)

    df["absence_longue_flag"] = (df["jours_absence_pre_match"] >= seuil_absence_jours).astype(int)
    df["retour_episode"] = df.groupby("id_joueur")["absence_longue_flag"].cumsum()

    df["matchs_depuis_retour_avant_match"] = (
        df.groupby(["id_joueur", "retour_episode"]).cumcount()
    )

    df["toi_pre_absence_ref"] = np.where(
        df["absence_longue_flag"] == 1,
        df["toi_moy_10"],
        np.nan,
    )
    df["pp_pre_absence_ref"] = np.where(
        df["absence_longue_flag"] == 1,
        df["pp_moy_5"],
        np.nan,
    )

    df["toi_pre_absence_ref"] = (
        df.groupby(["id_joueur", "retour_episode"])["toi_pre_absence_ref"]
        .transform(lambda s: s.ffill())
    )
    df["pp_pre_absence_ref"] = (
        df.groupby(["id_joueur", "retour_episode"])["pp_pre_absence_ref"]
        .transform(lambda s: s.ffill())
    )

    df["toi_moy_retour_2_avant_match"] = (
        df.groupby(["id_joueur", "retour_episode"])["temps_de_glace"]
        .transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
    )
    df["pp_moy_retour_2_avant_match"] = (
        df.groupby(["id_joueur", "retour_episode"])["temps_pp"]
        .transform(lambda s: s.shift(1).rolling(2, min_periods=1).mean())
    )

    df["ratio_toi_retour_vs_pre_absence"] = safe_ratio(
        df["toi_moy_retour_2_avant_match"],
        df["toi_pre_absence_ref"],
    )
    df["ratio_pp_retour_vs_pre_absence"] = safe_ratio(
        df["pp_moy_retour_2_avant_match"],
        df["pp_pre_absence_ref"],
    )

    df["eligible_post_retour"] = (
        (df["retour_episode"] > 0)
        & (df["matchs_depuis_retour_avant_match"] >= 2)
        & (df["toi_moy_retour_2_avant_match"].fillna(0) >= toi_min_retour)
        & (df["ratio_toi_retour_vs_pre_absence"].fillna(0) >= ratio_retour_toi_min)
    ).astype(int)

    fill_defaults = {
        "jours_repos_team": 3,
        "team_back_to_back": 0,
        "team_back_to_back_away": 0,
        "consecutive_away_games": 0,
        "team_winrate_5": 0.5,
        "team_gf_moy_5": 2.8,
        "team_ga_moy_5": 2.8,
        "nb_matchs_vs_adv_avant": 0,
        "points_vs_adv_shrunk": df["points_moy_10"].median(),
        "buts_vs_adv_shrunk": df["buts_moy_10"].median(),
        "tirs_vs_adv_shrunk": df["tirs_moy_10"].median(),
        "jours_absence_pre_match": 3,
        "matchs_depuis_retour_avant_match": 0,
        "eligible_post_retour": 0,
        "ratio_toi_retour_vs_pre_absence": 0,
        "ratio_pp_retour_vs_pre_absence": 0,
    }

    for col, val in fill_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)

    drop_cols = [c for c in ["team_code", "opp_code"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    return df


def write_summary(payload: dict) -> None:
    SUMMARY_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ensure_directories()

    df_source, pp_summary = charger_source_avec_pp()

    base_canonique = build_base_canonique(df_source)
    base_features = creer_features_temporelles_v2(base_canonique)
    base_features_context = enrichir_contexte_v2(base_features, config={})

    base_canonique.to_csv(OUTPUT_BASE_CANONIQUE, index=False)
    base_features.to_csv(OUTPUT_BASE_FEATURES, index=False)
    base_features_context.to_csv(OUTPUT_BASE_FEATURES_CONTEXT, index=False)

    summary = {
        "status": "ok",
        "input_file": str(INPUT_BASE_MATCH_FUSIONNEE),
        "pp_file": str(INPUT_PP_STATS_GAME),
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
    }
    write_summary(summary)

    print("01_build_base_features.py")
    print(f"Input base : {INPUT_BASE_MATCH_FUSIONNEE}")
    print(f"Input PP   : {INPUT_PP_STATS_GAME}")
    print(f"Output     : {OUTPUT_BASE_CANONIQUE}")
    print(f"Output     : {OUTPUT_BASE_FEATURES}")
    print(f"Output     : {OUTPUT_BASE_FEATURES_CONTEXT}")
    print(f"Summary    : {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
