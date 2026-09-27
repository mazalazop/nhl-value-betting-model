#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,log_loss

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"data/final/base_features_context_v2.csv"
OUT=ROOT/"outputs"
MIN_TRAIN_DATES=60
TEST_DAYS=14
STEP_DAYS=14

POINT=[
"is_home_player","nb_matchs_avant_match","jours_repos","tirs_moy_5","toi_moy_5","pp_moy_5","points_moy_5","buts_moy_5",
"passes_moy_5","tirs_moy_10","toi_moy_10","points_moy_10","buts_moy_10","passes_moy_10","nb_matchs_joues_10",
"hist_ok_5","hist_ok_10","tirs_par_60_5","points_par_60_5","buts_par_60_5","nb_matchs_vs_adv_avant","points_vs_adv_5",
"buts_vs_adv_5","tirs_vs_adv_5","points_vs_adv_shrunk","buts_vs_adv_shrunk","tirs_vs_adv_shrunk",
"jours_absence_pre_match","games_missed_proxy","absence_longue_flag","return_from_absence_flag","matchs_depuis_retour_avant_match",
"toi_pre_absence_ref","pp_pre_absence_ref","ratio_toi_retour_vs_pre_absence","ratio_pp_retour_vs_pre_absence",
"return_stabilized_flag","historical_current_weight","historical_prev_weight","point_hit_rate_last_5","point_hit_rate_last_10",
"point_hit_rate_last_20","point_hit_rate_season_pre","point_hit_rate_prev_season","point_hit_rate_weighted_pre",
"points_per_game_season_pre","points_per_game_prev_season","points_per_game_weighted_pre","recent_vs_expected_gap",
"current_point_streak_pre","current_no_point_streak_pre","no_point_streak_expected_pre","no_point_streak_excess_pre",
"no_point_drought_alert_pre","max_point_streak_last_2_seasons_pre","max_no_point_streak_last_2_seasons_pre",
"count_5plus_point_streaks_last_2_seasons_pre","is_home_team","jours_repos_team","team_back_to_back","team_back_to_back_away",
"consecutive_away_games","team_winrate_5","team_gf_moy_5","team_ga_moy_5","team_games_played_pre_approx",
"games_played_team_pre","games_remaining_team_pre","team_points_pre","conference_rank_pre","division_rank_pre",
"conference_cutoff_points_pre","wildcard_distance_pre","point_pctg_pre","goal_differential_pre","l10_points_pre",
"late_season_flag","playoff_pressure_simple"]

GOAL=[
"buts_moy_5","buts_moy_10","buts_par_60_5","tirs_moy_5","tirs_moy_10","toi_moy_5","toi_moy_10","pp_moy_5",
"buts_vs_adv_5","buts_vs_adv_shrunk","tirs_vs_adv_shrunk","points_moy_5","points_moy_10","goal_hit_rate_last_5",
"goal_hit_rate_last_10","goal_hit_rate_last_20","goal_hit_rate_season_pre","goal_hit_rate_prev_season","goal_hit_rate_weighted_pre",
"current_no_goal_streak_pre","max_no_goal_streak_last_2_seasons_pre","goal_streak_expected_pre","goal_streak_excess_pre",
"goal_drought_alert_pre","point_hit_rate_last_10","point_hit_rate_last_20","points_per_game_weighted_pre","is_home_player",
"jours_repos","jours_repos_team","team_back_to_back","team_back_to_back_away","team_winrate_5","team_gf_moy_5","team_ga_moy_5",
"games_remaining_team_pre","team_points_pre","wildcard_distance_pre","late_season_flag","playoff_pressure_simple",
"return_stabilized_flag","historical_current_weight","historical_prev_weight"]

def metrics(y,p):
 p=np.clip(p,1e-6,1-1e-6); base=float(y.mean())
 out={"rows":len(y),"positive_rate":base,"mean_predicted_probability":float(p.mean()),
      "brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,p,labels=[0,1])),
      "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
      "average_precision":float(average_precision_score(y,p)) if len(np.unique(y))==2 else np.nan}
 for q in (.01,.05,.10):
  k=max(1,int(np.ceil(len(y)*q))); z=np.argsort(-p)[:k]; pr=float(y[z].mean())
  out[f"precision_top_{int(q*100)}pct"]=pr
  out[f"lift_top_{int(q*100)}pct"]=pr/base if base else np.nan
 return out

def run(df,market,target,features):
 d=df.copy(); d["date_match"]=pd.to_datetime(d["date_match"],errors="coerce")
 d=d.dropna(subset=["date_match",target]).sort_values(["date_match","id_match","id_joueur"],kind="stable")
 d[target]=pd.to_numeric(d[target],errors="coerce").fillna(0).astype(int)
 dates=pd.DatetimeIndex(sorted(d.date_match.dt.normalize().unique()))
 if len(dates)<MIN_TRAIN_DATES+TEST_DAYS: raise ValueError(f"{market}: seulement {len(dates)} dates, insuffisant")
 rows=[]; preds=[]; origin=dates[MIN_TRAIN_DATES]; fold=0
 while origin<dates[-1]:
  end=origin+pd.Timedelta(days=TEST_DAYS)
  tr=d[d.date_match<origin]; te=d[(d.date_match>=origin)&(d.date_match<end)]
  if not te.empty and tr[target].nunique()==2:
   cols=[c for c in features if c in tr.columns and c in te.columns]
   X=tr[cols].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
   T=te[cols].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
   cols=[c for c in cols if not X[c].isna().all() and X[c].nunique(dropna=True)>1]
   X=X[cols];T=T[cols]
   for c in cols:
    med=X[c].median(); med=0.0 if pd.isna(med) else med
    X[c]=X[c].fillna(med);T[c]=T[c].fillna(med)
   y=tr[target].to_numpy(); pos=max(1,int(y.sum())); neg=max(1,len(y)-int(y.sum()))
   w=np.where(y==1,neg/pos,1.0)
   alert="no_point_drought_alert_pre" if market=="POINTS" else "goal_drought_alert_pre"
   if alert in tr:
    w*=np.where(pd.to_numeric(tr[alert],errors="coerce").fillna(0).to_numpy()>0,1.15,1.0)
   model=HistGradientBoostingClassifier(loss="log_loss",learning_rate=.05,max_iter=300,max_depth=6,min_samples_leaf=50,l2_regularization=1.0,early_stopping=False,random_state=42)
   model.fit(X,y,sample_weight=w); p=model.predict_proba(T)[:,1]; yy=te[target].to_numpy(); fold+=1
   m=metrics(yy,p);m.update({"market":market,"fold":fold,"train_start":str(tr.date_match.min().date()),"train_end":str(tr.date_match.max().date()),"test_start":str(origin.date()),"test_end":str(te.date_match.max().date()),"features_used":len(cols)});rows.append(m)
   meta=[c for c in ["date_match","id_match","id_joueur","nom","player_name","team_player_match","adversaire_match"] if c in te]
   q=te[meta].copy();q["market"]=market;q["fold"]=fold;q["target"]=yy;q["predicted_probability"]=p;preds.append(q)
  origin+=pd.Timedelta(days=STEP_DAYS)
 return pd.DataFrame(rows),pd.concat(preds,ignore_index=True) if preds else pd.DataFrame()

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if not INPUT.exists(): raise FileNotFoundError(INPUT)
 d=pd.read_csv(INPUT,low_memory=False)
 if "date_match" not in d: raise ValueError("date_match absent")
 if "a_marque_un_point" in d: pt="a_marque_un_point"
 elif "target_point_1p" in d: pt="target_point_1p"
 elif "points" in d: d["audit_point"]=(pd.to_numeric(d.points,errors="coerce").fillna(0)>=1).astype(int);pt="audit_point"
 else: raise ValueError("cible POINT absente")
 d["audit_goal"]=(pd.to_numeric(d.buts,errors="coerce").fillna(0)>=1).astype(int) if "buts" in d else np.nan
 results=[]; predictions=[]
 for market,target,features in [("POINTS",pt,POINT),("BUTS","audit_goal",GOAL)]:
  if target in d and pd.to_numeric(d[target],errors="coerce").notna().any():
   a,b=run(d,market,target,features)
   if not a.empty: results.append(a);predictions.append(b)
 if not results: raise ValueError("aucun marché auditable")
 m=pd.concat(results,ignore_index=True);p=pd.concat(predictions,ignore_index=True)
 m.to_csv(OUT/"walk_forward_audit_metrics.csv",index=False);p.to_csv(OUT/"walk_forward_audit_predictions.csv",index=False)
 summary={"status":"ok","method":"expanding-window walk-forward","minimum_train_dates":MIN_TRAIN_DATES,"test_window_days":TEST_DAYS,"step_days":STEP_DAYS,
          "folds_by_market":m.groupby("market").fold.nunique().to_dict(),
          "aggregate":m.groupby("market")[["brier","log_loss","roc_auc","average_precision","precision_top_10pct","lift_top_10pct"]].mean(numeric_only=True).round(6).to_dict(orient="index")}
 (OUT/"walk_forward_audit_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
 print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
