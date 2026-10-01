#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
00d_refresh_current_season_incremental.py

Refresh daily de l'état historique Henachel.

Le pipeline conserve un snapshot historique validé, puis :
1. récupère le calendrier de la saison courante ;
2. met à jour matchs.csv ;
3. met à jour joueurs.csv depuis les rosters courants ;
4. récupère les boxscores des matchs récents / nouveaux ;
5. construit leurs lignes stats + base_match_fusionnee ;
6. fusionne ces nouvelles lignes avec l'historique existant.

Objectif : intégrer réellement les matchs joués depuis le dernier état,
sans reconstruire plusieurs milliers de boxscores à chaque run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

DEFAULT_SEASON = 20262027
DEFAULT_RECENT_DAYS = 3


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Fichier requis absent : {path}")
    return pd.read_csv(path, low_memory=False)


def merge_latest(existing: pd.DataFrame, fresh: pd.DataFrame, key: str) -> pd.DataFrame:
    if existing.empty:
        out = fresh.copy()
    else:
        old = existing.copy()
        new = fresh.copy()
        old["_source_order"] = 0
        new["_source_order"] = 1
        out = pd.concat([old, new], ignore_index=True)
        out = out.drop_duplicates(subset=[key], keep="last")
        out = out.drop(columns=["_source_order"], errors="ignore")
    return out


def main() -> None:
    season = DEFAULT_SEASON
    recent_days = DEFAULT_RECENT_DAYS

    refresh_sources = load_module(PROJECT_ROOT / "model" / "00_refresh_sources.py", "henachel_refresh_sources")
    build_fusion = load_module(PROJECT_ROOT / "model" / "00b_build_base_match_fusionnee.py", "henachel_build_fusion")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    matchs_path = RAW_DIR / "matchs.csv"
    joueurs_path = RAW_DIR / "joueurs.csv"
    stats_path = RAW_DIR / "stats.csv"
    fusion_path = RAW_DIR / "base_match_fusionnee.csv"

    matchs_existing = read_csv_required(matchs_path)
    joueurs_existing = read_csv_required(joueurs_path)
    fusion_existing = read_csv_required(fusion_path)

    # Production model = regular season only.
    fusion_existing["_game_type"] = fusion_existing["id_match"].astype(str).str.zfill(10).str[4:6]
    fusion_existing = fusion_existing[fusion_existing["_game_type"] == "02"].drop(columns=["_game_type"]).copy()

    session = refresh_sources.build_session()

    print("=== 00d_refresh_current_season_incremental.py ===")
    print(f"Saison courante : {season}")
    print(f"Historique matchs : {len(matchs_existing)}")
    print(f"Historique base fusionnée : {len(fusion_existing)}")

    # ------------------------------------------------------------------
    # 1) Calendrier saison courante
    # ------------------------------------------------------------------
    current_matches = refresh_sources.collect_matches(
        session=session,
        seasons=[season],
        sleep_seconds=refresh_sources.DEFAULT_SLEEP_SECONDS,
    )

    merged_matches = merge_latest(matchs_existing, current_matches, "id_match")
    merged_matches["date_match"] = pd.to_datetime(merged_matches["date_match"], errors="coerce").dt.strftime("%Y-%m-%d")
    merged_matches = merged_matches.sort_values(["date_match", "id_match"]).reset_index(drop=True)
    merged_matches.to_csv(matchs_path, index=False)

    print(f"Matchs après refresh : {len(merged_matches)}")

    # ------------------------------------------------------------------
    # 2) Joueurs de la saison courante
    # ------------------------------------------------------------------
    current_rosters = refresh_sources.collect_players_from_rosters(
        session=session,
        seasons=[season],
        sleep_seconds=refresh_sources.DEFAULT_SLEEP_SECONDS,
    )
    current_rosters = current_rosters[["id_joueur", "nom", "position", "id_equipe", "saison", "source_priority"]].copy()
    current_rosters = current_rosters.sort_values(["id_joueur", "saison", "source_priority"])
    current_rosters = current_rosters.drop_duplicates("id_joueur", keep="last")
    current_rosters = current_rosters[["id_joueur", "nom", "position", "id_equipe"]]

    merged_players = merge_latest(joueurs_existing, current_rosters, "id_joueur")
    merged_players = merged_players[["id_joueur", "nom", "position", "id_equipe"]].sort_values(["nom", "id_joueur"]).reset_index(drop=True)
    merged_players.to_csv(joueurs_path, index=False)

    print(f"Joueurs après refresh : {len(merged_players)}")

    # ------------------------------------------------------------------
    # 3) Matchs courants joués à actualiser
    # ------------------------------------------------------------------
    played_current = merged_matches[
        merged_matches["saison"].astype(str) == str(season)
    ].copy()
    played_current["_game_type"] = played_current["id_match"].astype(str).str.zfill(10).str[4:6]
    played_current = played_current[played_current["_game_type"] == "02"].drop(columns=["_game_type"]).copy()
    played_current["date_match_dt"] = pd.to_datetime(played_current["date_match"], errors="coerce")

    max_existing_date = pd.to_datetime(fusion_existing["date_match"], errors="coerce").max()
    cutoff = max_existing_date - pd.Timedelta(days=recent_days) if pd.notna(max_existing_date) else pd.Timestamp("1900-01-01")

    existing_match_ids = set(pd.to_numeric(fusion_existing["id_match"], errors="coerce").dropna().astype(int).tolist())
    played_current["played_flag"] = (
        played_current["buts_domicile"].notna()
        & played_current["buts_exterieur"].notna()
    )

    refresh_matches = played_current[
        played_current["played_flag"]
        & (
            ~played_current["id_match"].isin(existing_match_ids)
            | (played_current["date_match_dt"] >= cutoff)
        )
    ].copy()

    refresh_matches = refresh_matches.drop(columns=["date_match_dt", "played_flag"], errors="ignore")

    print(f"Matchs courants à actualiser : {len(refresh_matches)}")

    if refresh_matches.empty:
        # Conserver stats.csv cohérent avec la base fusionnée.
        fusion_existing.to_csv(fusion_path, index=False)
        if stats_path.exists():
            print("Aucun nouveau boxscore à intégrer.")
        print("00d terminé sans nouveau match.")
        return

    # ------------------------------------------------------------------
    # 4) Boxscores récents
    # ------------------------------------------------------------------
    stats_new = build_fusion.build_stats_dataframe(
        session=session,
        df_matchs_played=refresh_matches,
        sleep_seconds=build_fusion.DEFAULT_SLEEP_SECONDS,
    )

    fusion_new = build_fusion.build_base_match_fusionnee(
        df_stats=stats_new,
        df_matchs=refresh_matches,
    )

    # Upsert joueur/match sur clé composite.
    fusion_old = fusion_existing.copy()
    fusion_old["_key"] = (
        pd.to_numeric(fusion_old["id_match"], errors="coerce").astype("Int64").astype(str)
        + "|"
        + pd.to_numeric(fusion_old["id_joueur"], errors="coerce").astype("Int64").astype(str)
    )
    fusion_new["_key"] = (
        pd.to_numeric(fusion_new["id_match"], errors="coerce").astype("Int64").astype(str)
        + "|"
        + pd.to_numeric(fusion_new["id_joueur"], errors="coerce").astype("Int64").astype(str)
    )

    combined = pd.concat([fusion_old, fusion_new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["_key"], keep="last").drop(columns=["_key"])
    combined["date_match"] = pd.to_datetime(combined["date_match"], errors="coerce").dt.strftime("%Y-%m-%d")
    combined = combined.sort_values(["date_match", "id_match", "id_joueur"]).reset_index(drop=True)
    combined.to_csv(fusion_path, index=False)

    # stats.csv reste un audit de la partie stats de la base fusionnée.
    stats_cols = [
        "id_joueur", "id_match", "season_source", "team_player_match",
        "adversaire_match", "home_road_flag", "is_home_player",
        "buts", "passes", "points", "tirs", "temps_de_glace", "temps_pp",
        "plus_moins", "penalty_minutes",
    ]
    stats_cols = [c for c in stats_cols if c in combined.columns]
    combined[stats_cols].to_csv(stats_path, index=False)

    summary = {
        "status": "ok",
        "season": season,
        "recent_days": recent_days,
        "existing_matches": int(len(matchs_existing)),
        "merged_matches": int(len(merged_matches)),
        "existing_players": int(len(joueurs_existing)),
        "merged_players": int(len(merged_players)),
        "existing_fusion_rows": int(len(fusion_existing)),
        "refresh_matches": int(len(refresh_matches)),
        "new_stats_rows": int(len(stats_new)),
        "new_fusion_rows": int(len(fusion_new)),
        "final_fusion_rows": int(len(combined)),
        "max_existing_date_before_refresh": str(max_existing_date.date()) if pd.notna(max_existing_date) else None,
        "cutoff_date": str(cutoff.date()) if pd.notna(cutoff) else None,
        "refresh_match_ids": pd.to_numeric(refresh_matches["id_match"], errors="coerce").dropna().astype(int).tolist(),
    }
    (OUTPUTS_DIR / "00d_refresh_current_season_incremental_summary.json").write_text(
        __import__("json").dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(summary)
    print("00d terminé.")


if __name__ == "__main__":
    main()
