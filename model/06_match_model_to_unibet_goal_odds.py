#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Jointure modèle BUT / cotes Unibet marché Buteur."""
from __future__ import annotations
import argparse, importlib.util, json
from datetime import date
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

def load_matcher():
    spec = importlib.util.spec_from_file_location("matcher_points", ROOT / "model/06_match_model_to_unibet_odds.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_goal_odds(path, m):
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("normalized_rows", [])
    if not isinstance(rows, list):
        raise ValueError("normalized_goals_odds.json['normalized_rows'] doit être une liste.")
    required = ["bookmaker","market_key","home_team","away_team","team","player_name_raw","odds_decimal"]
    df = pd.DataFrame(rows, columns=required)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans les cotes BUT: {missing}")
    df = df.copy()
    df["odds_decimal"] = pd.to_numeric(df["odds_decimal"], errors="coerce")
    df = df[df["odds_decimal"] > 1].copy()
    df["implied_probability"] = 1.0 / df["odds_decimal"]
    df["market"] = df["market_key"]
    df["stat"] = "goals"
    df["threshold"] = 1
    df["outcome_label"] = "Buteur"
    df["outcome_key"] = "player_to_score_including_ot"
    df["player_name"] = df["player_name_raw"].astype(str)
    df["team_name_normalized_join"] = df["team"].apply(m.normalize_text)
    df["player_name_normalized_join"] = df["player_name"].apply(m.normalize_player_join_name)
    df["matchup_key"] = df.apply(lambda x: m.matchup_key_from_team_names(x["home_team"], x["away_team"]), axis=1)
    for c in ["home_team","away_team","team","player_name"]:
        df[c] = df[c].fillna("").astype(str)
    return payload, df

def find_odds_path(explicit):
    if explicit:
        p = Path(explicit)
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    candidates = [
        OUT / "normalized_goals_odds.json",
        OUT / "normalized_goal_odds.json",
        ROOT / "normalized_goals_odds.json",
        ROOT / "normalized_goal_odds.json",
        ROOT / "data/external/normalized_goals_odds.json",
        ROOT / "data/external/normalized_goal_odds.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("normalized_goals_odds.json introuvable.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-csv", default=str(OUT / "predictions_upcoming_goal_enrichi_calibre_v1.csv"))
    p.add_argument("--odds-json", default=None)
    p.add_argument("--run-date", default=date.today().isoformat())
    a = p.parse_args()
    m = load_matcher()
    model = pd.read_csv(a.model_csv, low_memory=False)
    if "proba_goal_1p_calibree" not in model.columns:
        raise ValueError("Probabilité BUT absente.")
    odds_path = find_odds_path(a.odds_json)
    _, odds = load_goal_odds(odds_path, m)
    if odds.empty:
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["bet_id","run_date","bet_status","result","actual_stat_value","settled_at","recommended_flag","recommendation_rank","date_match","player_name","team","opponent","bookmaker","market","stat","threshold","odds_decimal","implied_probability","model_probability","edge_probability"]).to_csv(OUT / "06_matched_goal_edges.csv", index=False)
        (OUT / "06_goal_matching_summary.json").write_text(
            json.dumps({"status":"ok","run_date":a.run_date,"matched_rows":0,"odds_file":str(odds_path),"reason":"no_accepted_goal_market_rows"}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("BUT matched rows: 0 (aucune cote BUT acceptée)")
        return

    model["date_match"] = pd.to_datetime(model["date_match"], errors="coerce")
    model["team_player_match"] = model["team_player_match"].astype(str).str.upper().str.strip()
    model["adversaire_match"] = model["adversaire_match"].astype(str).str.upper().str.strip()
    model["nom"] = model["nom"].astype(str)
    model["player_name_normalized"] = model["nom"].apply(m.normalize_player_join_name)
    model["model_name_is_numeric"] = model["nom"].apply(m.is_numeric_like_name)
    model["team_aliases"] = model["team_player_match"].apply(m.canonical_team_aliases)
    model["matchup_key"] = model.apply(
        lambda r: m.matchup_key_from_codes(r["team_player_match"], r["adversaire_match"]), axis=1
    )

    rows = []
    for _, mr in model.sort_values(["date_match", "rank_proba_goal_sur_match"], na_position="last").iterrows():
        sub = odds[
            (odds["matchup_key"] == mr["matchup_key"])
            & (odds["player_name_normalized_join"] == mr["player_name_normalized"])
        ].copy()
        if not sub.empty and mr["team_aliases"]:
            sub = sub[sub["team_name_normalized_join"].isin(set(mr["team_aliases"]))].copy()
        method = "exact"
        score = None
        if sub.empty and not mr["model_name_is_numeric"]:
            sub = odds[
                (odds["matchup_key"] == mr["matchup_key"])
                & (odds["team_name_normalized_join"].isin(set(mr["team_aliases"])))
            ].copy()
            if not sub.empty:
                model_name = mr["player_name_normalized"]
                sub["fuzzy_score"] = sub["player_name_normalized_join"].apply(
                    lambda x: __import__("difflib").SequenceMatcher(None, model_name, x).ratio()
                )
                sub = sub.sort_values(["fuzzy_score","odds_decimal"], ascending=[False,True])
                best = float(sub.iloc[0]["fuzzy_score"])
                sub = sub[sub["fuzzy_score"] >= 0.965]
                sub = sub[sub["fuzzy_score"] == best].copy()
                method = "fuzzy"
                score = best
        if len(sub) != 1:
            continue

        o = sub.iloc[0]
        pmod = float(mr["proba_goal_1p_calibree"])
        odds_decimal = float(o["odds_decimal"])
        implied = float(o["implied_probability"])
        edge = pmod - implied
        ev = pmod * (odds_decimal - 1.0) - (1.0 - pmod)
        rows.append({
            "bet_id": m.make_bet_id(mr["date_match"].strftime("%Y-%m-%d"), o["bookmaker"], o["market"], o["stat"], 1, mr["player_name_normalized"], mr["team_player_match"], mr["adversaire_match"]),
            "run_date": a.run_date, "bet_status": "pending", "result": "", "actual_stat_value": None,
            "settled_at": "", "recommended_flag": False, "recommendation_rank": None,
            "date_match": mr["date_match"].strftime("%Y-%m-%d"), "id_match": mr.get("id_match"),
            "id_joueur": mr.get("id_joueur"), "player_name": mr["nom"], "team": mr["team_player_match"],
            "opponent": mr["adversaire_match"], "bookmaker": o["bookmaker"], "market": o["market"],
            "stat": o["stat"], "threshold": 1, "outcome_label": "Buteur", "outcome_key": o["outcome_key"],
            "event_url": o.get("event_url",""), "event_id": o.get("event_id",""), "event_slug": o.get("event_slug",""),
            "home_team": o["home_team"], "away_team": o["away_team"], "team_name_bookmaker": o["team"],
            "odds_decimal": odds_decimal, "implied_probability": implied,
            "model_probability_raw": float(mr["proba_goal_1p_raw"]), "model_probability": pmod,
            "fair_odds_model": 1.0 / pmod if pmod > 0 else None, "edge_probability": edge,
            "edge_probability_pct_points": edge * 100, "ev_per_unit": ev,
            "kelly_fraction": m.kelly_fraction_decimal_odds(pmod, odds_decimal),
            "is_positive_ev": bool(ev > 0), "match_method": method, "fuzzy_score": score,
            "goal_drought_alert_pre": mr.get("goal_drought_alert_pre", 0),
            "goal_streak_excess_pre": mr.get("goal_streak_excess_pre", 0),
            "toi_last_game_minutes": mr.get("toi_last_game_minutes", None),
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["model_probability","edge_probability"], ascending=[False,False])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "06_matched_goal_edges.csv"
    out.to_csv(path, index=False)
    (OUT / "06_goal_matching_summary.json").write_text(
        json.dumps({"status":"ok","run_date":a.run_date,"matched_rows":len(out),"odds_file":str(odds_path)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"BUT matched rows: {len(out)}")

if __name__ == "__main__":
    main()
