#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prédictions BUT sur le même univers de matchs que le modèle POINT."""
from __future__ import annotations
import argparse, importlib.util, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"
GOAL_FEATURES=[
"buts_moy_5","buts_moy_10","buts_par_60_5","tirs_moy_5","tirs_moy_10","toi_moy_5","toi_moy_10","pp_moy_5",
"buts_vs_adv_5","buts_vs_adv_shrunk","tirs_vs_adv_shrunk","points_moy_5","points_moy_10",
"goal_hit_rate_last_5","goal_hit_rate_last_10","goal_hit_rate_last_20","goal_hit_rate_season_pre",
"goal_hit_rate_prev_season","goal_hit_rate_weighted_pre","current_no_goal_streak_pre",
"max_no_goal_streak_last_2_seasons_pre","goal_streak_expected_pre","goal_streak_excess_pre","goal_drought_alert_pre",
"point_hit_rate_last_10","point_hit_rate_last_20","points_per_game_weighted_pre","is_home_player",
"jours_repos","jours_repos_team","team_back_to_back","team_back_to_back_away","team_winrate_5",
"team_gf_moy_5","team_ga_moy_5","games_remaining_team_pre","team_points_pre","wildcard_distance_pre",
"late_season_flag","playoff_pressure_simple","return_stabilized_flag","historical_current_weight","historical_prev_weight"]

def load_point_module():
    spec=importlib.util.spec_from_file_location("predict_points",ROOT/"model/05_predict_upcoming_games.py")
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--target-date",required=True)
    p.add_argument("--recent-lookback-days",type=int,default=45)
    a=p.parse_args()
    mod=load_point_module()
    matchs=mod.load_matchs(); joueurs=mod.load_joueurs(); hist,_,_=mod.load_history()
    target=pd.Timestamp(a.target_date).normalize()
    future=mod.select_future_matches(matchs,target)
    teams=sorted(set(future["id_equipe_domicile"].dropna().tolist()) | set(future["id_equipe_exterieur"].dropna().tolist()))
    pool=mod.build_recent_player_pool(hist,joueurs,target,teams,recent_lookback_days=a.recent_lookback_days)
    standings,_,standings_by_team=mod.load_standings()
    upcoming=mod.build_upcoming_universe(future,hist,matchs,pool,standings_by_team)

    h=hist[hist["date_match"]<target].copy()
    h["target_goal_1p"]=(pd.to_numeric(h["buts"],errors="coerce").fillna(0)>=1).astype(int)
    dates=sorted(pd.to_datetime(h["date_match"]).dt.strftime("%Y-%m-%d").unique())
    if len(dates)<20: raise ValueError("Pas assez de dates historiques pour le modèle BUT.")
    cut=max(1,int(len(dates)*.85));fit=h[h["date_match"].dt.strftime("%Y-%m-%d").isin(dates[:cut])].copy();cal=h[h["date_match"].dt.strftime("%Y-%m-%d").isin(dates[cut:])].copy()
    features=[c for c in GOAL_FEATURES if c in fit.columns]
    Xfit=fit[features].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    keep=[c for c in features if not Xfit[c].isna().all() and Xfit[c].nunique(dropna=True)>1]
    Xfit=Xfit[keep];Xcal=cal[keep].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    yfit=fit.target_goal_1p.astype(int);ycal=cal.target_goal_1p.astype(int)
    if yfit.nunique()<2 or ycal.nunique()<2: raise ValueError("Fit/calibration BUT sans deux classes.")
    pos=max(1,int(yfit.sum()));neg=max(1,len(yfit)-int(yfit.sum()));w=np.where(yfit.to_numpy()==1,neg/pos,1.0)
    if "goal_drought_alert_pre" in fit.columns: w=w*np.where(pd.to_numeric(fit["goal_drought_alert_pre"],errors="coerce").fillna(0).to_numpy()>0,1.15,1.0)
    model=HistGradientBoostingClassifier(loss="log_loss",learning_rate=.05,max_iter=300,max_depth=6,min_samples_leaf=50,l2_regularization=1.0,early_stopping=False,random_state=42)
    model.fit(Xfit,yfit,sample_weight=w)
    raw_cal=model.predict_proba(Xcal)[:,1]
    cal=LogisticRegression(solver="lbfgs",max_iter=1000,C=1e6,random_state=42);cal.fit(raw_cal.reshape(-1,1),ycal)
    Xup=upcoming[keep].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    raw=model.predict_proba(Xup)[:,1];proba=cal.predict_proba(raw.reshape(-1,1))[:,1]
    upcoming["proba_goal_1p_raw"]=raw;upcoming["proba_goal_1p_calibree"]=proba
    upcoming["rank_proba_goal_sur_date"]=upcoming.groupby("date_match")["proba_goal_1p_calibree"].rank(method="first",ascending=False).astype(int)
    upcoming["rank_proba_goal_sur_match"]=upcoming.groupby("id_match")["proba_goal_1p_calibree"].rank(method="first",ascending=False).astype(int)
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/"predictions_upcoming_goal_enrichi_calibre_v1.csv";upcoming.to_csv(path,index=False)
    summary={"status":"ok","target_date":target.strftime("%Y-%m-%d"),"feature_count":len(keep),"drought_bias":True,"output":str(path)}
    (OUT/"05_predict_upcoming_goals_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
