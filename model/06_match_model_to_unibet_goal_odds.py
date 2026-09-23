#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Jointure modèle BUT / cotes Unibet BUT 1+."""
from __future__ import annotations
import argparse, importlib.util, json
from datetime import date
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"

def load_points_matcher():
    spec=importlib.util.spec_from_file_location("matcher_points",ROOT/"model/06_match_model_to_unibet_odds.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model-csv",default=str(OUT/"predictions_upcoming_goal_enrichi_calibre_v1.csv"))
    p.add_argument("--odds-json",default=None)
    p.add_argument("--run-date",default=date.today().isoformat())
    a=p.parse_args()
    m=load_points_matcher()
    model=pd.read_csv(a.model_csv,low_memory=False)
    if "proba_goal_1p_calibree" not in model.columns: raise ValueError("Probabilité BUT absente.")
    odds_path=Path(a.odds_json) if a.odds_json else next((p for p in [OUT/"normalized_goal_odds.json",ROOT/"normalized_goal_odds.json",ROOT/"data/external/normalized_goal_odds.json"] if p.exists()),None)
    if odds_path is None: raise FileNotFoundError("normalized_goal_odds.json introuvable.")
    _,odds=m.load_odds_json(odds_path)
    # Réutiliser les clés de jointure du matcher POINT.
    model["date_match"]=pd.to_datetime(model["date_match"],errors="coerce")
    model["team_player_match"]=model["team_player_match"].astype(str).str.upper().str.strip()
    model["adversaire_match"]=model["adversaire_match"].astype(str).str.upper().str.strip()
    for c in ["id_match","id_joueur","rank_proba_goal_sur_date","rank_proba_goal_sur_match"]:
        if c in model.columns: model[c]=pd.to_numeric(model[c],errors="coerce")
    model["nom"]=model["nom"].astype(str)
    model["player_name_normalized_raw"]=model["nom"].apply(m.normalize_text)
    model["player_name_normalized"]=model["nom"].apply(m.normalize_player_join_name)
    model["model_name_is_numeric"]=model["nom"].apply(m.is_numeric_like_name)
    model["team_aliases"]=model["team_player_match"].apply(m.canonical_team_aliases)
    model["team_primary_name"]=model["team_player_match"].apply(m.canonical_team_primary)
    model["opponent_primary_name"]=model["adversaire_match"].apply(m.canonical_team_primary)
    model["matchup_key"]=model.apply(lambda r:m.matchup_key_from_codes(r["team_player_match"],r["adversaire_match"]),axis=1)
    rows=[]
    for _,mr in model.sort_values(["date_match","rank_proba_goal_sur_match"],na_position="last").iterrows():
        sub=m.exact_candidate_subset(mr,odds);method="exact";score=None
        if len(sub)!=1:
            sub=m.fuzzy_candidate_subset(mr,odds);method="fuzzy"
        if len(sub)!=1: continue
        orow=sub.iloc[0]
        pmod=float(mr["proba_goal_1p_calibree"]);imp=float(orow["implied_probability"]);od=float(orow["odds_decimal"])
        rows.append({
            "bet_id":m.make_bet_id(mr["date_match"].strftime("%Y-%m-%d"),orow["bookmaker"],orow["market"],orow["stat"],orow["threshold"],mr["player_name_normalized"],mr["team_player_match"],mr["adversaire_match"]),
            "run_date":a.run_date,"bet_status":"pending","result":"","actual_stat_value":None,"settled_at":"",
            "recommended_flag":False,"recommendation_rank":None,"date_match":mr["date_match"].strftime("%Y-%m-%d"),
            "id_match":mr.get("id_match"),"id_joueur":mr.get("id_joueur"),"player_name":mr["nom"],
            "team":mr["team_player_match"],"opponent":mr["adversaire_match"],"bookmaker":orow["bookmaker"],
            "market":orow["market"],"stat":orow["stat"],"threshold":orow["threshold"],"outcome_label":orow["outcome_label"],
            "outcome_key":orow["outcome_key"],"event_url":orow["event_url"],"event_id":orow["event_id"],"event_slug":orow["event_slug"],
            "home_team":orow["home_team"],"away_team":orow["away_team"],"team_name_bookmaker":orow["team"],
            "odds_decimal":od,"implied_probability":imp,"model_probability_raw":float(mr["proba_goal_1p_raw"]),
            "model_probability":pmod,"fair_odds_model":1.0/pmod if pmod>0 else None,
            "edge_probability":pmod-imp,"edge_probability_pct_points":(pmod-imp)*100,
            "ev_per_unit":pmod*(od-1)-(1-pmod),"kelly_fraction":m.kelly_fraction_decimal_odds(pmod,od),
            "is_positive_ev":bool(pmod*(od-1)-(1-pmod)>0),"match_method":method,"fuzzy_score":score,
            "goal_drought_alert_pre":mr.get("goal_drought_alert_pre",0),
            "goal_streak_excess_pre":mr.get("goal_streak_excess_pre",0),
            "toi_last_game_minutes":mr.get("toi_last_game_minutes",None),
        })
    out=pd.DataFrame(rows).sort_values(["model_probability","edge_probability"],ascending=[False,False]) if rows else pd.DataFrame()
    path=OUT/"06_matched_goal_edges.csv";out.to_csv(path,index=False)
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/"06_goal_matching_summary.json").write_text(json.dumps({"status":"ok","run_date":a.run_date,"matched_rows":len(out),"odds_file":str(odds_path)},ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"BUT matched rows: {len(out)}")

if __name__=="__main__": main()
