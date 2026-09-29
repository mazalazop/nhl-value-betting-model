#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import importlib.util, json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"data/final/base_features_context_v2.csv"
OUT=ROOT/"outputs"
MIN_TRAIN_DATES=60
TEST_DAYS=14
STEP_DAYS=14

GOAL_FEATURES=[
"buts_moy_5","buts_moy_10","buts_par_60_5","tirs_moy_5","tirs_moy_10","toi_moy_5","toi_moy_10","pp_moy_5",
"buts_vs_adv_5","buts_vs_adv_shrunk","tirs_vs_adv_shrunk","points_moy_5","points_moy_10","goal_hit_rate_last_5",
"goal_hit_rate_last_10","goal_hit_rate_last_20","goal_hit_rate_season_pre","goal_hit_rate_prev_season","goal_hit_rate_weighted_pre",
"current_no_goal_streak_pre","max_no_goal_streak_last_2_seasons_pre","goal_streak_expected_pre","goal_streak_excess_pre",
"goal_drought_alert_pre","point_hit_rate_last_10","point_hit_rate_last_20","points_per_game_weighted_pre","is_home_player",
"jours_repos","jours_repos_team","team_back_to_back","team_back_to_back_away","team_winrate_5","team_gf_moy_5","team_ga_moy_5",
"games_remaining_team_pre","team_points_pre","wildcard_distance_pre","late_season_flag","playoff_pressure_simple",
"return_stabilized_flag","historical_current_weight","historical_prev_weight"
]

def load_point_training():
    spec=importlib.util.spec_from_file_location("train_point",ROOT/"model/02_train_point_model.py")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def metrics(y,p):
    y=np.asarray(y,dtype=int); p=np.clip(np.asarray(p,float),1e-6,1-1e-6); base=float(y.mean())
    out={"rows":len(y),"positive_rate":base,"mean_predicted_probability":float(p.mean()),
         "brier":float(brier_score_loss(y,p)),"log_loss":float(log_loss(y,p,labels=[0,1])),
         "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else np.nan,
         "average_precision":float(average_precision_score(y,p)) if len(np.unique(y))==2 else np.nan}
    for q in (.01,.05,.10):
        k=max(1,int(np.ceil(len(y)*q))); z=np.argsort(-p)[:k]; pr=float(y[z].mean())
        out[f"precision_top_{int(q*100)}pct"]=pr; out[f"lift_top_{int(q*100)}pct"]=pr/base if base else np.nan
    return out

def fit_predict(tr,te,target,features,train_mod):
    cols,_,_=train_mod.resolve_feature_list(tr,features,target,"date_match")
    train_mod.assert_no_forbidden_features(cols,target,"date_match")
    X=tr[cols].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    T=te[cols].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    keep=~X.isna().all() & (X.nunique(dropna=True)>1); cols=X.columns[keep].tolist(); X=X[cols]; T=T[cols]
    for c in cols:
        med=X[c].median(); med=0.0 if pd.isna(med) else med
        X[c]=X[c].fillna(med); T[c]=T[c].fillna(med)
    y=tr[target].astype(int).to_numpy(); yt=te[target].astype(int).to_numpy()
    pos=max(1,int(y.sum())); neg=max(1,len(y)-int(y.sum()))
    alert_col="no_point_drought_alert_pre" if target=="audit_point" else "goal_drought_alert_pre"
    alert_source=tr[alert_col] if alert_col in tr.columns else pd.Series(0,index=tr.index)\n    alert=pd.to_numeric(alert_source,errors="coerce").fillna(0).to_numpy()
    w=np.where(y==1,neg/pos,1.0)*np.where(alert>0,1.15,1.0)
    model=HistGradientBoostingClassifier(loss="log_loss",learning_rate=.05,max_iter=300,max_depth=6,min_samples_leaf=50,l2_regularization=1.0,early_stopping=False,random_state=42)
    split=int(len(tr)*.85)
    p_cal=None
    if split>=50 and len(np.unique(y[:split]))==2 and len(np.unique(y[split:]))==2:
        # Fit the calibrator only on dates the base model did not train on.
        cal_model=HistGradientBoostingClassifier(loss="log_loss",learning_rate=.05,max_iter=300,max_depth=6,min_samples_leaf=50,l2_regularization=1.0,early_stopping=False,random_state=42)
        cal_model.fit(X.iloc[:split],y[:split],sample_weight=w[:split])
        p_cal_train=cal_model.predict_proba(X.iloc[split:])[:,1]
        calibrator=LogisticRegression(C=1e6,max_iter=1000)
        calibrator.fit(p_cal_train.reshape(-1,1),y[split:])
        # Refit the production model on the complete historical window, then calibrate test probabilities.
        model=HistGradientBoostingClassifier(loss="log_loss",learning_rate=.05,max_iter=300,max_depth=6,min_samples_leaf=50,l2_regularization=1.0,early_stopping=False,random_state=42)
        model.fit(X,y,sample_weight=w)
        p_raw=model.predict_proba(T)[:,1]
        p_cal=calibrator.predict_proba(p_raw.reshape(-1,1))[:,1]
    else:
        model=HistGradientBoostingClassifier(loss="log_loss",learning_rate=.05,max_iter=300,max_depth=6,min_samples_leaf=50,l2_regularization=1.0,early_stopping=False,random_state=42)
        model.fit(X,y,sample_weight=w)
        p_cal=model.predict_proba(T)[:,1]
    return p_cal,yt,len(cols)

def run(df,market,target,features,train_mod):
    d=df.dropna(subset=["date_match",target]).copy(); d[target]=pd.to_numeric(d[target],errors="coerce").fillna(0).astype(int)
    d=d.sort_values(["date_match","id_match","id_joueur"],kind="stable")
    dates=pd.DatetimeIndex(sorted(d.date_match.dt.normalize().unique()))
    if len(dates)<MIN_TRAIN_DATES+TEST_DAYS: return pd.DataFrame(),pd.DataFrame()
    rows=[]; preds=[]; origin=dates[MIN_TRAIN_DATES]; fold=0
    while origin<dates[-1]:
        end=origin+pd.Timedelta(days=TEST_DAYS); tr=d[d.date_match<origin]; te=d[(d.date_match>=origin)&(d.date_match<end)]
        if not te.empty and len(tr)>=500 and tr[target].nunique()==2 and te[target].nunique()==2:
            p,yy,nfeat=fit_predict(tr,te,target,features,train_mod); fold+=1
            m=metrics(yy,p); m.update({"market":market,"fold":fold,"train_end":str(tr.date_match.max().date()),"test_start":str(te.date_match.min().date()),"test_end":str(te.date_match.max().date()),"features_used":nfeat}); rows.append(m)
            meta=[c for c in ["date_match","id_match","id_joueur","nom","player_name","team_player_match","adversaire_match"] if c in te]
            q=te[meta].copy(); q["market"]=market; q["fold"]=fold; q["target"]=yy; q["predicted_probability"]=p; preds.append(q)
        origin+=pd.Timedelta(days=STEP_DAYS)
    return pd.DataFrame(rows),pd.concat(preds,ignore_index=True) if preds else pd.DataFrame()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if not INPUT.exists(): raise FileNotFoundError(INPUT)
    train_mod=load_point_training(); d=pd.read_csv(INPUT,low_memory=False); d=train_mod.normalize_boolean_like_columns(d)
    d,point_target=train_mod.find_target_column(d); d,date_col=train_mod.find_date_column(d)
    if date_col!="date_match": d["date_match"]=d[date_col]
    d["audit_point"]=pd.to_numeric(d[point_target],errors="coerce").fillna(0).astype(int)
    d["audit_goal"]=(pd.to_numeric(d["buts"],errors="coerce").fillna(0)>=1).astype(int) if "buts" in d else np.nan
    point_features=list(dict.fromkeys(train_mod.BASELINE_FEATURE_WHITELIST+train_mod.ENRICHED_EXTRA_WHITELIST))
    results=[]; predictions=[]
    for market,target,features in [("POINTS","audit_point",point_features),("BUTS","audit_goal",GOAL_FEATURES)]:
        if target in d and pd.to_numeric(d[target],errors="coerce").notna().any():
            a,b=run(d,market,target,features,train_mod)
            if not a.empty: results.append(a); predictions.append(b)
    if not results: raise ValueError("aucun marché auditable")
    m=pd.concat(results,ignore_index=True); p=pd.concat(predictions,ignore_index=True)
    m.to_csv(OUT/"walk_forward_audit_metrics.csv",index=False); p.to_csv(OUT/"walk_forward_audit_predictions.csv",index=False)
    summary={"status":"ok","method":"expanding-window walk-forward; production point whitelists; in-window sigmoid calibration","minimum_train_dates":MIN_TRAIN_DATES,"test_window_days":TEST_DAYS,"step_days":STEP_DAYS,"folds_by_market":m.groupby("market").fold.nunique().to_dict(),"aggregate":m.groupby("market")[["brier","log_loss","roc_auc","average_precision","precision_top_10pct","lift_top_10pct"]].mean(numeric_only=True).round(6).to_dict(orient="index")}
    (OUT/"walk_forward_audit_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
