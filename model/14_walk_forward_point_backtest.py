#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/"data/final/base_features_context_v2.csv"

def load_mod():
    spec=importlib.util.spec_from_file_location("train_point",ROOT/"model/02_train_point_model.py")
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def metric(y,p):
    y=np.asarray(y,dtype=int); p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
    return {"rows":len(y),"positive_rate":float(y.mean()),"brier":float(brier_score_loss(y,p)),
            "logloss":float(log_loss(y,p,labels=[0,1])),
            "auc":float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan,
            "average_precision":float(average_precision_score(y,p)) if len(np.unique(y))>1 else np.nan}

def fit_model(X,y):
    y=np.asarray(y,dtype=int)
    pos=max(1,int(y.sum())); neg=max(1,len(y)-int(y.sum()))
    m=HistGradientBoostingClassifier(loss="log_loss",learning_rate=0.05,max_iter=300,max_depth=6,
        min_samples_leaf=50,l2_regularization=1.0,early_stopping=False,random_state=42)
    m.fit(X,y,sample_weight=np.where(y==1,neg/pos,1.0)); return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default=str(INPUT)); ap.add_argument("--output",default="outputs/walk_forward_point_results.csv")
    ap.add_argument("--min-train-days",type=int,default=365); ap.add_argument("--test-days",type=int,default=30)
    ap.add_argument("--step-days",type=int,default=30); a=ap.parse_args()
    mod=load_mod(); df=pd.read_csv(a.input,low_memory=False)
    df=mod.normalize_boolean_like_columns(df); df,target=mod.find_target_column(df); df,date_col=mod.find_date_column(df)
    df=df.sort_values(date_col).reset_index(drop=True)
    base,_,_=mod.resolve_feature_list(df,list(mod.BASELINE_FEATURE_WHITELIST),target,date_col)
    enrich,_,_=mod.resolve_feature_list(df,list(dict.fromkeys(mod.BASELINE_FEATURE_WHITELIST+mod.ENRICHED_EXTRA_WHITELIST)),target,date_col)
    cursor=df[date_col].min()+pd.Timedelta(days=a.min_train_days); end=df[date_col].max(); rows=[]; fold=0
    while cursor+pd.Timedelta(days=a.test_days)<=end:
        test_start=cursor; test_end=cursor+pd.Timedelta(days=a.test_days)
        train=df[df[date_col]<test_start].copy(); test=df[(df[date_col]>=test_start)&(df[date_col]<test_end)].copy()
        if len(train)<500 or len(test)<50: cursor+=pd.Timedelta(days=a.step_days); continue
        fold+=1; split=int(len(train)*0.80); fit_df=train.iloc[:split].copy(); cal_df=train.iloc[split:].copy()
        for variant,cols in [("baseline",base),("enrichi",enrich)]:
            Xfit=mod.to_numeric_frame(fit_df,cols); Xcal=mod.to_numeric_frame(cal_df,cols); Xte=mod.to_numeric_frame(test,cols)
            Xfit,Xcal,Xte,kept,_=mod.keep_train_valid_features_only(Xfit,Xcal,Xte)
            yfit=fit_df[target].astype(int); ycal=cal_df[target].astype(int); yte=test[target].astype(int)
            if yfit.nunique()<2 or yte.nunique()<2: continue
            model=fit_model(Xfit,yfit); p_raw=model.predict_proba(Xte)[:,1]; p_cal=p_raw
            if ycal.nunique()==2:
                calibrator=LogisticRegression(C=1e6,max_iter=1000)
                calibrator.fit(model.predict_proba(Xcal)[:,1].reshape(-1,1),ycal)
                p_cal=calibrator.predict_proba(p_raw.reshape(-1,1))[:,1]
            for kind,p in [("raw",p_raw),("sigmoid",p_cal)]:
                row=metric(yte,p); row.update({"fold":fold,"variant":variant,"probability_type":kind,
                    "fit_end":fit_df[date_col].max().strftime("%Y-%m-%d"),
                    "calibration_start":cal_df[date_col].min().strftime("%Y-%m-%d"),
                    "test_start":test_start.strftime("%Y-%m-%d"),"test_end":test_end.strftime("%Y-%m-%d"),
                    "train_rows":len(train),"test_rows":len(test),"features_kept":len(kept)})
                rows.append(row)
        cursor+=pd.Timedelta(days=a.step_days)
    if not rows: raise SystemExit("Aucun fold exploitable.")
    result=pd.DataFrame(rows); out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); result.to_csv(out,index=False)
    summary={"status":"ok","folds":int(result.fold.nunique()),
             "aggregate":result.groupby(["variant","probability_type"])[["brier","logloss","auc","average_precision"]].mean().reset_index().to_dict("records")}
    out.with_suffix(".json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(result.groupby(["variant","probability_type"])[["brier","logloss","auc","average_precision"]].mean())

if __name__=="__main__": main()
