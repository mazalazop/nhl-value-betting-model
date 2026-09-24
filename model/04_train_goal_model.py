#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Modèle BUT: probabilité qu'un joueur marque au moins 1 but."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,brier_score_loss,log_loss,roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
FEATURES=ROOT/"data/final/base_features_context_v2.csv"
OUT=ROOT/"outputs"
BASE=[
"is_home_player","saison","nb_matchs_avant_match","jours_repos_raw","is_premier_match_joueur","jours_repos",
"tirs_moy_5","toi_moy_5","pp_moy_5","points_moy_5","buts_moy_5","passes_moy_5","tirs_moy_10","toi_moy_10",
"points_moy_10","buts_moy_10","passes_moy_10","nb_matchs_joues_10","hist_ok_5","hist_ok_10","tirs_par_60_5",
"points_par_60_5","buts_par_60_5","nb_matchs_vs_adv_avant","points_vs_adv_5","buts_vs_adv_5","tirs_vs_adv_5",
"points_vs_adv_shrunk","buts_vs_adv_shrunk","tirs_vs_adv_shrunk","jours_absence_pre_match","games_missed_proxy",
"absence_longue_flag","retour_episode","return_from_absence_flag","matchs_depuis_retour_avant_match","toi_pre_absence_ref",
"pp_pre_absence_ref","toi_moy_retour_3_avant_match","pp_moy_retour_3_avant_match","ratio_toi_retour_vs_pre_absence",
"ratio_pp_retour_vs_pre_absence","return_stabilized_flag","historical_current_weight","historical_prev_weight",
"point_hit_rate_last_5","point_hit_rate_last_10","point_hit_rate_last_20","point_hit_rate_season_pre",
"point_hit_rate_prev_season","point_hit_rate_weighted_pre","points_per_game_season_pre","points_per_game_prev_season",
"points_per_game_weighted_pre","recent_vs_expected_gap","current_point_streak_pre","current_no_point_streak_pre",
"max_point_streak_last_2_seasons_pre","max_no_point_streak_last_2_seasons_pre","count_5plus_point_streaks_last_2_seasons_pre",
"is_home_team","jours_repos_team","team_back_to_back","team_back_to_back_away","consecutive_away_games","team_winrate_5",
"team_gf_moy_5","team_ga_moy_5","team_games_played_pre_approx","games_played_team_pre","games_remaining_team_pre",
"team_points_pre","conference_rank_pre","division_rank_pre","conference_cutoff_points_pre","wildcard_distance_pre",
"point_pctg_pre","goal_differential_pre","l10_points_pre","late_season_flag","playoff_pressure_simple"]
GOAL=[
"goal_hit_rate_last_5","goal_hit_rate_last_10","goal_hit_rate_last_20","goal_hit_rate_season_pre",
"goal_hit_rate_prev_season","goal_hit_rate_weighted_pre","current_no_goal_streak_pre",
"max_no_goal_streak_last_2_seasons_pre","goal_streak_expected_pre","goal_streak_excess_pre","goal_drought_alert_pre"]

def metrics(split,y,p):
 y=np.asarray(y,int);p=np.asarray(p,float)
 return {"split":split,"rows":int(len(y)),"positive_rate":float(y.mean()),
 "brier_score":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,p,labels=[0,1])),
 "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None,
 "average_precision":float(average_precision_score(y,p)) if len(np.unique(y))>1 else None}

def main():
 if not FEATURES.exists(): raise FileNotFoundError(FEATURES)
 df=pd.read_csv(FEATURES,low_memory=False)
 date_col=next((c for c in ["date_match","date","match_date","game_date","date_game"] if c in df.columns),None)
 if not date_col or "buts" not in df.columns: raise ValueError("Date et colonne buts requises.")
 df[date_col]=pd.to_datetime(df[date_col],errors="coerce");df["buts"]=pd.to_numeric(df["buts"],errors="coerce")
 df=df[df[date_col].notna()&df["buts"].notna()].sort_values(date_col).reset_index(drop=True)
 df["target_goal_1p"]=(df["buts"]>=1).astype(int)
 features=[c for c in BASE+GOAL if c in df.columns]
 dates=sorted(df[date_col].dt.strftime("%Y-%m-%d").unique())
 if len(dates)<20: raise ValueError(f"Pas assez de dates historiques: {len(dates)}")
 a=max(1,int(len(dates)*.70));b=max(a+1,int(len(dates)*.85));b=min(b,len(dates)-1)
 tr=df[df[date_col].dt.strftime("%Y-%m-%d").isin(dates[:a])].copy()
 ca=df[df[date_col].dt.strftime("%Y-%m-%d").isin(dates[a:b])].copy()
 te=df[df[date_col].dt.strftime("%Y-%m-%d").isin(dates[b:])].copy()
 Xtr=tr[features].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
 keep=[c for c in features if not Xtr[c].isna().all() and Xtr[c].nunique(dropna=True)>1]
 Xtr=Xtr[keep];Xca=ca[keep].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan);Xte=te[keep].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
 ytr=tr.target_goal_1p.astype(int);yca=ca.target_goal_1p.astype(int);yte=te.target_goal_1p.astype(int)
 pos=max(1,int(ytr.sum()));neg=max(1,len(ytr)-int(ytr.sum()));w=np.where(ytr.to_numpy()==1,neg/pos,1.0)
    if "goal_drought_alert_pre" in tr.columns: w=w*np.where(pd.to_numeric(train["goal_drought_alert_pre"],errors="coerce").fillna(0).to_numpy()>0,1.15,1.0)
 model=HistGradientBoostingClassifier(loss="log_loss",learning_rate=.05,max_iter=300,max_depth=6,min_samples_leaf=50,l2_regularization=1.0,early_stopping=False,random_state=42)
 model.fit(Xtr,ytr,sample_weight=w)
 pca=model.predict_proba(Xca)[:,1]
 cal=LogisticRegression(solver="lbfgs",max_iter=1000,C=1e6,random_state=42);cal.fit(pca.reshape(-1,1),yca)
 pte_raw=model.predict_proba(Xte)[:,1];pte=cal.predict_proba(pte_raw.reshape(-1,1))[:,1]
 OUT.mkdir(parents=True,exist_ok=True)
 pd.DataFrame([metrics("validation",yca,cal.predict_proba(pca.reshape(-1,1))[:,1]),metrics("test",yte,pte)]).to_csv(OUT/"metrics_modele_but_enrichi_v1.csv",index=False)
 meta=[c for c in ["date_match","id_match","id_joueur","nom","position","team_player_match","adversaire_match","is_home_player","saison"] if c in te.columns]
 pred=te[meta].copy();pred["target_goal_1p"]=yte.to_numpy();pred["proba_goal_1p_raw"]=pte_raw;pred["proba_goal_1p_calibree"]=pte
 pred["rank_proba_goal_sur_date"]=pred.groupby(date_col)["proba_goal_1p_calibree"].rank(method="first",ascending=False).astype(int)
 pred.to_csv(OUT/"predictions_test_but_enrichi_v1.csv",index=False)
 summary={"status":"ok","target":"goal_1p","feature_count":len(keep),"goal_drought_bias":True,"drought_alert_is_behavioral_bias":True,"train_rows":len(tr),"calib_rows":len(ca),"test_rows":len(te)}
 (OUT/"04_train_goal_model_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
