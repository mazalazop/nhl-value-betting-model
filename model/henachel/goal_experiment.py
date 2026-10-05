"""Locked GOAL experiments. No publication or modification of the POINT model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
from henachel.goal import BASE_FEATURES, FEATURE_FAMILIES, GOAL_PARAMS, load_goal_history, augment_goal_features, fit_goal_model, numeric_features
from henachel.evaluation import metrics, reliability_bins
from henachel.manifest import manifest

TARGET='a_marque_un_but'


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=str))


def logloss_rows(y,p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6)
    return -(y*np.log(p)+(1-y)*np.log1p(-p))


def paired_comparison(reference,candidate,family_size=8):
    keys=['date_match','id_match','id_joueur']
    a=reference[reference.game_type.eq(2)].set_index(keys).sort_index()
    b=candidate[candidate.game_type.eq(2)].set_index(keys).sort_index()
    if not a.index.is_unique or not b.index.is_unique:
        raise ValueError('Duplicate GOAL comparison identities')
    pd.testing.assert_index_equal(a.index,b.index);np.testing.assert_array_equal(a[TARGET],b[TARGET])
    y=a[TARGET].to_numpy();delta=logloss_rows(y,b.probability)-logloss_rows(y,a.probability)
    db=(y-b.probability.to_numpy())**2-(y-a.probability.to_numpy())**2
    dates=a.index.get_level_values('date_match');levels=sorted(dates.unique());n=len(levels)
    codes=pd.Categorical(dates,categories=levels).codes
    daily=np.bincount(codes,weights=delta,minlength=n);counts=np.bincount(codes,minlength=n)
    rng=np.random.default_rng(20261004);starts=rng.integers(0,n,size=(10000,(n+6)//7))
    samples=((starts[:,:,None]+np.arange(7))%n).reshape(10000,-1)[:,:n]
    bootstrap=daily[samples].sum(axis=1)/counts[samples].sum(axis=1)
    lo,hi=np.quantile(bootstrap,[.05/(2*family_size),1-.05/(2*family_size)])
    folds=[]
    for fold in sorted(a.fold.unique()):
        mask=a.fold.to_numpy()==fold
        folds.append(dict(fold=int(fold),delta_logloss=float(delta[mask].mean()),delta_brier=float(db[mask].mean())))
    return dict(delta_logloss=float(delta.mean()),delta_brier=float(db.mean()),ci_lower=float(lo),ci_upper=float(hi),folds=folds,
                passes=bool(hi<0 and all(f['delta_logloss']<0 and f['delta_brier']<0 for f in folds)))


def score(output,name,frame,dates,features,parameters=None,final=False):
    path=output/name
    if path.exists():raise FileExistsError(f'Immutable GOAL experiment exists: {path}')
    path.mkdir();reserve=int(len(dates)*.85)
    windows=[dates[reserve:]] if final else [dates[i:min(i+60,reserve)] for i in range(160,reserve,60)]
    predictions=[];folds=[]
    for number,window in enumerate(windows):
        cutoff=pd.Timestamp(window[0]);train=frame[frame.date_match<cutoff];ev=frame[frame.date_match.isin(window)]
        model,cal,meta=fit_goal_model(train,cutoff,features,parameters)
        raw=model.predict_proba(numeric_features(ev,meta['feature_cols_kept']))[:,1]
        rows=ev[['date_match','id_match','id_joueur','game_type',TARGET]].copy()
        rows['fold']=number;rows['raw_probability']=raw;rows['probability']=cal.predict(raw)
        rows['prevalence_baseline']=float(train[TARGET].mean());rows['player_baseline']=ev.goal_hit_rate_season_pre
        rows['player_smoothed_baseline']=(ev.season_goal_hits_before_match+10*float(train[TARGET].mean()))/(ev.season_games_before_match+10)
        predictions.append(rows)
        regular=rows[rows.game_type.eq(2)]
        folds.append(dict(fold=number,eval_start=str(cutoff),eval_end=str(ev.date_match.max()),fit=meta,
                          regular=metrics(regular[TARGET],regular.probability)))
        print(name,number,folds[-1]['regular'],flush=True)
    predictions=pd.concat(predictions,ignore_index=True)
    if predictions.duplicated(['id_match','id_joueur']).any():raise ValueError('Duplicate GOAL predictions')
    strata={};bins=[]
    for label,subset in [('all',predictions),('regular',predictions[predictions.game_type.eq(2)]),('playoffs',predictions[predictions.game_type.eq(3)])]:
        strata[label]={col:metrics(subset[TARGET],subset[col]) for col in ['probability','raw_probability','prevalence_baseline','player_baseline','player_smoothed_baseline']}
        table=reliability_bins(subset[TARGET],subset.probability)
        strata[label]['ece']=sum(b['count']*abs(b['mean_probability']-b['observed_rate']) for b in table if b['count'])/len(subset)
        for row in table:row['stratum']=label;bins.append(row)
    predictions.to_csv(path/'predictions.csv.gz',index=False)
    pd.DataFrame(bins).to_csv(path/'calibration.csv',index=False)
    write(path/'summary.json',dict(name=name,target='GOAL_1_PLUS',features=features,parameters=parameters or {},final_test_used=final,
          final_start=str(pd.Timestamp(dates[reserve])),folds=folds,strata=strata,manifest=manifest([output/'protocol.json'],GOAL_PARAMS)))
    return predictions


def load_scores(path):
    return pd.read_csv(path/'predictions.csv.gz')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','baseline','families','pk','models','final'],required=True)
    parser.add_argument('--pk',type=Path,help='Verified game-level opponent PK file for the optional predeclared family')
    args=parser.parse_args();out=args.output
    if (out/'decision.json').exists():
        raise ValueError('GOAL holdout already consumed; do not tune or repeat selection here')
    if (out/'candidate_locked.json').exists() and args.phase in {'families','pk','models'}:
        raise ValueError('GOAL candidate already locked')
    frame=load_goal_history(args.features);dates=sorted(frame.date_match.unique());reserve=int(len(dates)*.85)
    if reserve<=160:raise ValueError('Insufficient GOAL historical dates for frozen protocol')
    protocol=dict(target='GOAL_1_PLUS',input_sha256=hashlib.sha256(args.features.read_bytes()).hexdigest(),
                  input=str(args.features.resolve()),rows=len(frame),type_counts=frame.game_type.value_counts().to_dict(),
                  baseline_features=BASE_FEATURES,feature_families=FEATURE_FAMILIES,parameters=GOAL_PARAMS,
                  final_start=str(pd.Timestamp(dates[reserve])),min_train_dates=160,eval_window_dates=60,
                  candidate_models={'depth6':{'max_depth':6},'leaf100':{'min_samples_leaf':100}},family_size=8,
                  smoothed_player_baseline={'prior_weight':10,'prior':'training prevalence only'},
                  optional_pk='Not eligible without full dated coverage and parity tests',
                  selection='Four-fold logloss/Brier improvement plus Bonferroni 7-date block CI upper<0; one family at most',
                  final_acceptance='Beats prevalence, raw and smoothed player rates on logloss/Brier; paired 95% logloss CI upper<0; ECE<=0.03',
                  limitation='Same final dates previously used for POINT, not a completely untouched cross-task sample')
    if args.phase=='prepare':
        out.mkdir(parents=True,exist_ok=True)
        if (out/'protocol.json').exists():raise FileExistsError('GOAL protocol already frozen')
        write(out/'protocol.json',protocol);return
    old=json.loads((out/'protocol.json').read_text())
    if (old['input_sha256']!=protocol['input_sha256'] or old['baseline_features']!=BASE_FEATURES
            or old['feature_families']!=FEATURE_FAMILIES or old['parameters']!=GOAL_PARAMS):
        raise ValueError('GOAL frozen input/features mismatch')
    if args.phase!='final':frame=frame[frame.date_match<pd.Timestamp(old['final_start'])].copy()
    frame=augment_goal_features(frame)
    if args.pk:
        from henachel.goal_pk import add_opponent_pk
        frame=add_opponent_pk(frame,pd.read_csv(args.pk))
    if args.phase=='baseline':score(out,'baseline',frame,dates,BASE_FEATURES)
    elif args.phase=='families':
        baseline=load_scores(out/'baseline');comparisons=[]
        for name,extra in FEATURE_FAMILIES.items():
            scores=score(out,name,frame,dates,BASE_FEATURES+extra);scores['date_match']=scores.date_match.astype(str)
            result=paired_comparison(baseline,scores);result['name']=name;comparisons.append(result)
        eligible=[r for r in comparisons if r['passes']]
        winner=min(eligible,key=lambda r:(r['delta_logloss'],r['delta_brier'],r['name']))['name'] if eligible else 'baseline'
        write(out/'feature_comparisons.json',comparisons)
        write(out/'features_locked.json',{'name':winner,'features':BASE_FEATURES+FEATURE_FAMILIES.get(winner,[])})
    elif args.phase=='pk':
        if not args.pk:raise ValueError('PK experiment requires verified source')
        from henachel.goal_pk import FEATURE
        comparisons=json.loads((out/'feature_comparisons.json').read_text())
        if any(c['name']=='pk' for c in comparisons):raise FileExistsError('PK already compared')
        scores=score(out,'pk',frame,dates,BASE_FEATURES+[FEATURE]);scores['date_match']=scores.date_match.astype(str)
        result=paired_comparison(load_scores(out/'baseline'),scores);result['name']='pk';comparisons.append(result)
        eligible=[r for r in comparisons if r['passes']]
        winner=min(eligible,key=lambda r:(r['delta_logloss'],r['delta_brier'],r['name']))['name'] if eligible else 'baseline'
        write(out/'feature_comparisons.json',comparisons)
        write(out/'features_locked.json',{'name':winner,'features':BASE_FEATURES+([FEATURE] if winner=='pk' else FEATURE_FAMILIES.get(winner,[]))})
    elif args.phase=='models':
        selected=json.loads((out/'features_locked.json').read_text());reference=load_scores(out/selected['name']);candidates=[]
        for name,override in old['candidate_models'].items():
            scores=score(out,name,frame,dates,selected['features'],override);scores['date_match']=scores.date_match.astype(str)
            result=paired_comparison(reference,scores);result['name']=name;candidates.append(result)
        eligible=[r for r in candidates if r['passes']]
        winner=min(eligible,key=lambda r:(r['delta_logloss'],r['delta_brier'],r['name']))['name'] if eligible else selected['name']
        write(out/'model_comparisons.json',candidates)
        write(out/'candidate_locked.json',{'name':winner,'features':selected['features'],'parameters':old['candidate_models'].get(winner,{}),'final_labels_used':False})
    else:
        selected=json.loads((out/'candidate_locked.json').read_text())
        scores=score(out,'final',frame,dates,selected['features'],selected['parameters'],final=True);checks={}
        for column in ['prevalence_baseline','player_baseline','player_smoothed_baseline']:
            reference=scores.copy();reference['probability']=reference[column]
            checks[column]=paired_comparison(reference,scores,family_size=1)
        summary=json.loads((out/'final/summary.json').read_text())
        acceptable=all(c['passes'] for c in checks.values()) and summary['strata']['regular']['ece']<=.03
        write(out/'decision.json',{'scientifically_acceptable':acceptable,'checks':checks,'candidate':selected,'holdout_consumed':True,'no_roi_claim':True})
