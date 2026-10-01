#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Match calibrated BUTEURS probabilities with normalized Unibet odds."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
DEFAULT_MODEL = OUT / "predictions_upcoming_goal_enrichi_calibre_v1.csv"
DEFAULT_ODDS = OUT / "normalized_goals_odds.json"

TEAM_NAME_ALIASES = {
    "ANA": ["anaheim ducks", "ducks", "ana"],
    "BOS": ["boston bruins", "bruins", "bos"],
    "BUF": ["buffalo sabres", "sabres", "buf"],
    "CAR": ["carolina hurricanes", "hurricanes", "car"],
    "CBJ": ["columbus blue jackets", "blue jackets", "cbj"],
    "CGY": ["calgary flames", "flames", "cgy"],
    "CHI": ["chicago blackhawks", "blackhawks", "chi"],
    "COL": ["colorado avalanche", "avalanche", "col"],
    "DAL": ["dallas stars", "stars", "dal"],
    "DET": ["detroit red wings", "red wings", "det"],
    "EDM": ["edmonton oilers", "oilers", "edm"],
    "FLA": ["florida panthers", "panthers", "fla"],
    "LAK": ["los angeles kings", "la kings", "kings", "lak"],
    "MIN": ["minnesota wild", "wild", "min"],
    "MTL": ["montreal canadiens", "canadiens", "mtl"],
    "NJD": ["new jersey devils", "devils", "njd"],
    "NSH": ["nashville predators", "predators", "nsh"],
    "NYI": ["new york islanders", "islanders", "nyi"],
    "NYR": ["new york rangers", "rangers", "nyr"],
    "OTT": ["ottawa senators", "senators", "ott"],
    "PHI": ["philadelphia flyers", "flyers", "phi"],
    "PIT": ["pittsburgh penguins", "penguins", "pit"],
    "SEA": ["seattle kraken", "kraken", "sea"],
    "SJS": ["san jose sharks", "san jose sharks", "sharks", "sjs"],
    "STL": ["st louis blues", "st. louis blues", "blues", "stl"],
    "TBL": ["tampa bay lightning", "lightning", "tbl"],
    "TOR": ["toronto maple leafs", "maple leafs", "tor"],
    "UTA": ["utah mammoth", "utah hockey club", "mammoth", "uta"],
    "VAN": ["vancouver canucks", "canucks", "van"],
    "VGK": ["vegas golden knights", "golden knights", "vgk"],
    "WPG": ["winnipeg jets", "jets", "wpg"],
    "WSH": ["washington capitals", "capitals", "wsh"],
}

NAME_TO_CODE = {}
for code, aliases in TEAM_NAME_ALIASES.items():
    for alias in aliases:
        NAME_TO_CODE[alias] = code


def norm(text: Any) -> str:
    s = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", s.casefold()).strip()


def team_code(value: Any) -> str:
    n = norm(value)
    if n in NAME_TO_CODE:
        return NAME_TO_CODE[n]
    if len(n) == 3 and n.upper() in TEAM_NAME_ALIASES:
        return n.upper()
    for alias, code in NAME_TO_CODE.items():
        if n == alias or n in alias or alias in n:
            return code
    return ""


def matchup_key(home: Any, away: Any) -> str:
    a, b = team_code(home), team_code(away)
    return "|".join(sorted([a, b])) if a and b else ""


def player_key(value: Any) -> str:
    return norm(value)


def bet_id(date_match: str, player: str, home: str, away: str) -> str:
    raw = "|".join([date_match, player_key(player), team_code(home), team_code(away)])
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def load_model(path: Path, run_date: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    required = ["date_match", "id_match", "id_joueur", "nom", "team_player_match", "adversaire_match", "proba_goal_1p_calibree"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes modèle BUT : {missing}")
    df["date_match"] = pd.to_datetime(df["date_match"], errors="coerce")
    df = df[df["date_match"].dt.strftime("%Y-%m-%d") == run_date].copy()
    df["proba_goal_1p_calibree"] = pd.to_numeric(df["proba_goal_1p_calibree"], errors="coerce")
    df["matchup_key"] = df.apply(lambda r: matchup_key(r["team_player_match"], r["adversaire_match"]), axis=1)
    df["player_key"] = df["nom"].map(player_key)
    return df.dropna(subset=["proba_goal_1p_calibree"])


def load_odds(path: Path) -> tuple[dict, pd.DataFrame]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("normalized_rows", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return payload, pd.DataFrame(columns=["matchup_key", "player_key", "odds_decimal"])
    required = ["home_team", "away_team", "player_name_raw", "odds_decimal"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes normalized_goals_odds.json : {missing}")
    df["matchup_key"] = df.apply(lambda r: matchup_key(r["home_team"], r["away_team"]), axis=1)
    df["player_key"] = df["player_name_raw"].map(player_key)
    df["odds_decimal"] = pd.to_numeric(df["odds_decimal"], errors="coerce")
    return payload, df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-date", required=True)
    ap.add_argument("--model-csv", default=str(DEFAULT_MODEL))
    ap.add_argument("--odds-json", default=str(DEFAULT_ODDS))
    args = ap.parse_args()

    model = load_model(Path(args.model_csv), args.run_date)
    payload, odds = load_odds(Path(args.odds_json))

    matched = []
    used = set()

    for _, m in model.sort_values("proba_goal_1p_calibree", ascending=False).iterrows():
        subset = odds[
            (odds["matchup_key"] == m["matchup_key"])
            & (odds["player_key"] == m["player_key"])
        ].copy()
        subset = subset[~subset.index.isin(used)]

        method = "exact"
        fuzzy_score = np.nan

        if subset.empty and m["player_key"]:
            cand = odds[odds["matchup_key"] == m["matchup_key"]].copy()
            if not cand.empty:
                cand["fuzzy_score"] = cand["player_key"].map(
                    lambda x: SequenceMatcher(None, m["player_key"], x).ratio()
                )
                cand = cand.sort_values("fuzzy_score", ascending=False)
                if float(cand.iloc[0]["fuzzy_score"]) >= 0.90:
                    subset = cand.head(1)
                    method = "fuzzy"
                    fuzzy_score = float(subset.iloc[0]["fuzzy_score"])

        if len(subset) != 1:
            continue

        o = subset.iloc[0]
        used.add(subset.index[0])
        p = float(m["proba_goal_1p_calibree"])
        odds_decimal = float(o["odds_decimal"])
        implied = 1.0 / odds_decimal if odds_decimal > 1 else np.nan
        edge = p - implied if pd.notna(implied) else np.nan
        ev = p * (odds_decimal - 1.0) - (1.0 - p) if odds_decimal > 1 else np.nan

        matched.append({
            "bet_id": bet_id(args.run_date, m["nom"], m["team_player_match"], m["adversaire_match"]),
            "run_date": args.run_date,
            "bet_status": "pending",
            "result": "",
            "date_match": args.run_date,
            "id_match": int(m["id_match"]) if pd.notna(m["id_match"]) else None,
            "id_joueur": int(m["id_joueur"]) if pd.notna(m["id_joueur"]) else None,
            "player_name": m["nom"],
            "player_name_bookmaker": o["player_name_raw"],
            "team": m["team_player_match"],
            "opponent": m["adversaire_match"],
            "bookmaker": payload.get("bookmaker", "unibet_fr"),
            "market": payload.get("market_key", "player_to_score_including_ot"),
            "stat": "goal",
            "threshold": 1,
            "odds_decimal": odds_decimal,
            "implied_probability": implied,
            "model_probability": p,
            "edge_probability": edge,
            "edge_probability_pct_points": edge * 100 if pd.notna(edge) else np.nan,
            "ev_per_unit": ev,
            "is_positive_ev": bool(ev > 0) if pd.notna(ev) else False,
            "fair_odds_model": 1.0 / p if p > 0 else np.nan,
            "match_method": method,
            "fuzzy_score": fuzzy_score,
            "home_team": o.get("home_team", ""),
            "away_team": o.get("away_team", ""),
            "event_url": o.get("event_url", ""),
            "event_slug": o.get("event_slug", ""),
        })

    matched_df = pd.DataFrame(matched)
    if not matched_df.empty:
        matched_df = matched_df.sort_values(
            ["model_probability", "edge_probability", "odds_decimal"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

    out_csv = OUT / "06b_matched_goal_edges.csv"
    out_json = OUT / "06b_goal_matching_summary.json"
    matched_df.to_csv(out_csv, index=False)

    summary = {
        "status": "ok",
        "run_date": args.run_date,
        "market": "player_goals",
        "model_rows": int(len(model)),
        "bookmaker_rows": int(len(odds)),
        "matched_rows": int(len(matched_df)),
        "positive_ev_rows": int((matched_df["ev_per_unit"] > 0).sum()) if not matched_df.empty else 0,
        "match_rate_vs_bookmaker": float(len(matched_df) / len(odds)) if len(odds) else 0.0,
        "match_rate_vs_model": float(len(matched_df) / len(model)) if len(model) else 0.0,
        "output": str(out_csv),
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
