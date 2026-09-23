#!/usr/bin/env python3
"""Walk-forward audit of the POINT model.

No future rows are used for fitting. Each fold trains on an expanding historical
window and evaluates the next chronological block. The script compares the
existing baseline feature family with the enriched team/playoff context.

It intentionally does not touch bookmaker odds: this is a scientific model
audit, not a cherry-picked betting backtest.
"""
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
OUT=ROOT/"outputs"

def load_train_module():
    spec=importlib.util.spec_from_file_location("train_point", ROOT/"model/02_train_point_model.py")
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod

def metric(y,p):
    y=np.asarray(y,dtype=int); p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
    return {
        "rows":int(len(y)),
        "positive_rate":float(y.mean()),
        "brier":float(brier_score_loss(y,p)),
        "logloss":float(log_loss(y,p,labels=[0,1])),
        "auc":float(roc_auc_score(y,p)) if len(np.unique(y))>1 else np.nan,
        "average_precision":float(average_precision_score(y,p)) if len(np.unique(y))>1 else np.nan,
    }

def fit_model(X,y):
    m=HistGradientBoostingClassifier(
        loss="log_loss",learning_rate=0.05,max_iter=300,max_depth=6,
        min_samples_leaf=50,l2_regularization=1.0,early_stopping=False,random_state=42
    )
    y=np.asarray(y,dtype=int)
    pos=max(1,int(y.sum())); neg=max(1,int(len(y)-y.sum()))
    weights=np.where(y==1,neg/pos,1.0)
    m.fit(X,y,sample_weight=weights)
    return m

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input",default=str(INPUT))
    ap.add_argument("--output",default=str(OUT/"walk_forward_point_results.csv"))
    ap.add_argument("--min-train-days",type=int,default=365)
    ap.add_argument("--test-days",type=int,default=30)
    ap.add_argument("--step-days",type=int,default=30)
    args=ap.parse_args()

    mod=load_train_module()
    df=pd.read_csv(args.input,low_memory=False)
    df=mod.normalize_boolean_like_columns(df)
    df,target=mod.find_target_column(df)
    df,date_col=mod.find_date_column(df)
    df=df.sort_values(date_col).reset_index(drop=True)
    base_cols,_,_=mod.resolve_feature_list(df,list(mod.BASELINE_FEATURE_WHITELIST),target,date_col)
    enriched_cols,_,_=mod.resolve_feature_list(df,list(dict.fromkeys(mod.BASELINE_FEATURE_WHITELIST+mod.ENRICHED_EXTRA_WHITELIST)),target,date_col)

    start=df[date_col].min()+pd.Timedelta(days=args.min_train_days)
    end=df[date_col].max()
    rows=[]
    fold=0
    cursor=start
    while cursor+pd.Timedelta(days=args.test_days)<=end:
        test_start=cursor; test_end=cursor+pd.Timedelta(days=args.test_days)
        train=df[df[date_col]<test_start].copy()
        test=df[(df[date_col]>=test_start)&(df[date_col]<test_end)].copy()
        if len(train)<500 or len(test)<50:
            cursor+=pd.Timedelta(days=args.step_days); continue
        fold+=1
        for variant,cols in [("baseline",base_cols),("enrichi",enriched_cols)]:
            Xtr=mod.to_numeric_frame(train,cols); Xte=mod.to_numeric_frame(test,cols)
            Xtr,Xte,_,kept,dropped=mod.keep_train_valid_features_only(Xtr,Xte,Xte)
            ytr=train[target].astype(int); yte=test[target].astype(int)
            if ytr.nunique()<2 or yte.nunique()<2: continue
            model=fit_model(Xtr,ytr)
            p_raw=model.predict_proba(Xte)[:,1]
            # Calibration fit uses only the most recent 20% of the training period.
            split_idx=max(1,int(len(train)*0.80))
            cal_train=train.iloc[split_idx:].copy()
            cal_X=mod.to_numeric_frame(cal_train,kept)
            cal_y=cal_train[target].astype(int)
            if cal_y.nunique()==2:
                cal_model=LogisticRegression(C=1e6,max_iter=1000)
                cal_model.fit(model.predict_proba(cal_X)[:,1].reshape(-1,1),cal_y)
                p_cal=cal_model.predict_proba(p_raw.reshape(-1,1))[:,1]
            else:
                p_cal=p_raw
            for kind,p in [("raw",p_raw),("sigmoid",p_cal)]:
                m=metric(yte,p)
                m.update({"fold":fold,"variant":variant,"probability_type":kind,
                          "train_end":train[date_col].max().strftime("%Y-%m-%d"),
                          "test_start":test_start.strftime("%Y-%m-%d"),
                          "test_end":test_end.strftime("%Y-%m-%d"),
                          "train_rows":len(train),"test_rows":len(test),
                          "features_kept":len(kept)})
                rows.append(m)
        cursor+=pd.Timedelta(days=args.step_days)

    if not rows: raise SystemExit("Aucun fold exploitable. Réduire --min-train-days ou vérifier les données.")
    result=pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    result.to_csv(args.output,index=False)
    summary={"status":"ok","folds":int(result.fold.nunique()),
             "rows_evaluated":int(result.test_rows.sum()/2) if not result.empty else 0,
             "aggregate":result.groupby(["variant","probability_type"])[["brier","logloss","auc","average_precision"]].mean().reset_index().to_dict("records")}
    (Path(args.output).with_suffix(".json")).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(result.groupby(["variant","probability_type"])[["brier","logloss","auc","average_precision"]].mean())

if __name__=="__main__": main()
