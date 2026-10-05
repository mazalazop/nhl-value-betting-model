"""Expanding-window evaluation; the final 15% of dates are reserved and untouched."""
import math
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss,log_loss,roc_auc_score,average_precision_score
from henachel.calibration import validate_probabilities
from henachel.point import binary_labels


def metrics(y,p):
    y=binary_labels(y).to_numpy();p=validate_probabilities(p)
    if not len(y) or len(y)!=len(p): raise ValueError('Empty or mismatched evaluation')
    top=np.argsort(-p,kind='stable')[:max(1,math.ceil(.1*len(p)))]
    precision=float(y[top].mean());prevalence=float(y.mean())
    return dict(rows=len(y),brier=float(brier_score_loss(y,p)),logloss=float(log_loss(y,np.clip(p,1e-6,1-1e-6),labels=[0,1])),
                auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
                avg_precision=float(average_precision_score(y,p)) if len(np.unique(y))==2 else None,
                precision_top_10pct=precision,lift_top_10pct=precision/prevalence if prevalence else None,positive_rate=prevalence)


def reliability_bins(y,p):
    y=binary_labels(y).to_numpy();p=validate_probabilities(p);rows=[]
    for i in range(10):
        mask=(p>=i/10)&((p<(i+1)/10) if i<9 else (p<=1))
        rows.append(dict(lower=i/10,upper=(i+1)/10,count=int(mask.sum()),mean_probability=float(p[mask].mean()) if mask.any() else None,observed_rate=float(y[mask].mean()) if mask.any() else None))
    return rows


def expanding_evaluation(frame,fit,feature_names,target='target_point_1p',min_train_dates=20,window_dates=10):
    if min_train_dates<20 or window_dates<1:raise ValueError('Insufficient training window')
    df=frame.sort_values(['date_match','id_match','id_joueur']).copy()
    dates=sorted(pd.to_datetime(df.date_match).unique());reserve=int(len(dates)*.85)
    if reserve<=min_train_dates:raise ValueError('Insufficient dates before reserved final test')
    reports=[];predictions=[]
    for start in range(min_train_dates,reserve,window_dates):
        eval_dates=dates[start:min(start+window_dates,reserve)];cut=pd.Timestamp(eval_dates[0])
        train=df[df.date_match<cut];test=df[df.date_match.isin(eval_dates)]
        model,cal,meta=fit(train,target,'date_match',cut)
        x=test.reindex(columns=meta['feature_cols_kept']).apply(pd.to_numeric,errors='coerce')
        prob=cal.predict(model.predict_proba(x)[:,1]);y=test[target].to_numpy()
        baseline=np.full(len(test),train[target].mean())
        fold=dict(fold=len(reports),train_end=str(train.date_match.max()),eval_start=str(cut),eval_end=str(test.date_match.max()),model=metrics(y,prob),baseline=metrics(y,baseline),calibration=reliability_bins(y,prob),fit=meta)
        reports.append(fold)
        scored=test[['date_match','id_match','id_joueur',target]].copy();scored['probability']=prob;scored['fold']=fold['fold'];predictions.append(scored)
    scored=pd.concat(predictions,ignore_index=True)
    daily=[]
    for date,group in scored.groupby('date_match'):
        order=group.sort_values(['probability','id_match','id_joueur'],ascending=[False,True,True])
        daily.append(dict(date=str(date),precision_top5=float(order.head(5)[target].mean()),precision_top10=float(order.head(10)[target].mean())))
    return dict(status='ok',method='expanding_window',final_test_start=str(pd.Timestamp(dates[reserve])),final_test_used=False,
                aggregate=metrics(scored[target],scored.probability),calibration=reliability_bins(scored[target],scored.probability),daily_top=daily,folds=reports),scored
