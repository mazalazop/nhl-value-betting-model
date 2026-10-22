"""Paired chronological research evaluation; no picks reconstructed or published."""
import hashlib
import json
import platform
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from henachel import but_v2 as v2
from henachel.goal import load_goal_history, augment_goal_features, fit_goal_model, numeric_features
from henachel.goal_operational import FEATURES as V1_FEATURES
from henachel.evaluation import reliability_bins

def losses(y,p):
    p=np.clip(np.asarray(p,dtype=float),1e-6,1-1e-6);y=np.asarray(y)
    return -(y*np.log(p)+(1-y)*np.log1p(-p)),(y-p)**2

def metrics(rows):
    y=rows.y.to_numpy();p=rows.probability.to_numpy();ll,br=losses(y,p)
    selected=p>=.5
    result=dict(n=len(rows),dates=int(rows.date_match.nunique()),games=int(rows.id_match.nunique()),
                brier=float(br.mean()),log_loss=float(ll.mean()),hit_rate=float(y.mean()),
                precision_at_05=float(y[selected].mean()) if selected.any() else None,
                recall_at_05=float(y[selected].sum()/y.sum()) if y.sum() else None)
    for k in [5,10]:
        top=rows.sort_values(['date_match','probability','id_match','id_joueur'],ascending=[True,False,True,True]).groupby('date_match').head(k)
        result[f'top{k}_n']=len(top);result[f'top{k}_hit_rate']=float(top.y.mean())
    result['calibration']=reliability_bins(y,p)
    return result

def paired(a,b,replicates=2000,seed=20261020,family_size=1):
    keys=['date_match','id_match','id_joueur']
    a=a.set_index(keys).sort_index();b=b.set_index(keys).sort_index()
    if not a.index.is_unique or not b.index.is_unique:raise ValueError('Duplicate prediction')
    pd.testing.assert_index_equal(a.index,b.index)
    np.testing.assert_array_equal(a.y,b.y)
    al,ab=losses(a.y,a.probability);bl,bb=losses(b.y,b.probability)
    daily=pd.DataFrame({'ll':bl-al,'br':bb-ab,'n':1},index=a.index.get_level_values('date_match')).groupby(level=0).sum()
    rng=np.random.default_rng(seed);n=len(daily)
    # Seven successive observed NHL dates; dates preserve within-slate dependencies.
    starts=rng.integers(0,n,size=(replicates,(n+6)//7))
    samples=((starts[:,:,None]+np.arange(7))%n).reshape(replicates,-1)[:,:n]
    denom=daily.n.to_numpy()[samples].sum(axis=1)
    result={}
    for key,delta in [('ll',bl-al),('br',bb-ab)]:
        draws=daily[key].to_numpy()[samples].sum(axis=1)/denom
        result[key]={'delta':float(delta.mean()),'ci95':np.quantile(draws,[.025,.975]).tolist(),
                     'family_adjusted_ci':np.quantile(draws,[.025/family_size,1-.025/family_size]).tolist()}
    return result

def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')

def run(source,output,settings=None):
    settings=settings or v2.config()
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    frame=load_goal_history(source)
    frame=frame[frame.date_match<pd.Timestamp(settings['exclude_dates_from'])].copy()
    v2.validate(frame)
    dates=sorted(frame.date_match.unique())
    if len(dates)<=settings['min_train_dates']:raise ValueError('Not enough dates')
    specs={name:(name,v2.FEATURES) for name in settings['models']}
    for family,columns in v2.FAMILIES.items():
        specs['without_'+family]=('logistic',[c for c in v2.FEATURES if c not in columns])
    for w in (3,5,10):
        specs['volume_window_'+str(w)]=('logistic',[c for c in v2.FEATURES if c not in v2.FAMILIES['volume'] or c in [f'shots_mean_{w}',f'shots60_{w}']])
    protocol=dict(settings=settings,source=str(source),source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                  rows=len(frame),dates=len(dates),start=str(frame.date_match.min()),end=str(frame.date_match.max()),
                  variants={k:dict(model=m,features=f) for k,(m,f) in specs.items()},
                  git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  python=platform.python_version(),point_in_time_verified=False,prospective=False,
                  reference='V1 GOAL17 operational algorithm refitted once per fold, not daily historical predictions',
                  prior_holdout_reused=True,roi=None,promotion_allowed=False)
    # Persist exact source and executed code: enough to reproduce without local untracked files.
    pd.DataFrame(frame).to_csv(output/'input.csv.gz',index=False)
    codefiles=[Path(__file__),Path(v2.__file__),Path('model/henachel/goal.py'),Path('model/henachel/calibration.py')]
    protocol['code_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in codefiles}
    write(output/'protocol.json',protocol)
    features=v2.build_features(frame);reference=augment_goal_features(frame)
    missing={c:float(features[c].isna().mean()) for c in v2.FEATURES}
    write(output/'missingness.json',missing)
    predictions=[];folds=[]
    for fold,i in enumerate(range(settings['min_train_dates'],len(dates),settings['test_dates'])):
        window=dates[i:i+settings['test_dates']];cutoff=pd.Timestamp(window[0])
        ev=features[features.date_match.isin(window)]
        meta={}
        model,cal,fit=fit_goal_model(reference,cutoff,V1_FEATURES)
        ref=reference[reference.date_match.isin(window)]
        base=ev[['date_match','id_match','id_joueur','game_type','is_home_player']].copy()
        base['y']=v2.goal_labels(ev).to_numpy();base['fold']=fold
        ref=ref.set_index(v2.KEYS).loc[pd.MultiIndex.from_frame(ev[v2.KEYS])].reset_index()
        r=base.copy();r['model']='v1';r['probability']=cal.predict(model.predict_proba(numeric_features(ref,fit['feature_cols_kept']))[:,1])
        predictions.append(r);meta['v1']=fit
        for name,(kind,cols) in specs.items():
            bundle=v2.fit(features,cutoff,kind,cols,settings)
            r=base.copy();r['model']=name;r['probability']=v2.predict(bundle,ev)
            predictions.append(r);meta[name]=bundle['metadata']
        folds.append(dict(fold=fold,test_start=str(cutoff),test_end=str(window[-1]),n=len(ev),models=meta))
        print(f'fold={fold} rows={len(ev)} completed',flush=True)
    predictions=pd.concat(predictions,ignore_index=True)
    predictions.to_csv(output/'predictions.csv.gz',index=False)
    summaries={};comparisons={};reference=predictions[predictions.model.eq('v1')]
    for name,rows in predictions.groupby('model'):
        summaries[name]=dict(all=metrics(rows),coverage=len(rows)/len(reference),
                            segments={str(k):metrics(g) for k,g in rows.groupby('game_type')},
                            home_away={str(k):metrics(g) for k,g in rows.groupby('is_home_player')},
                            folds={str(k):metrics(g) for k,g in rows.groupby('fold')})
        if name!='v1':comparisons[name]=paired(reference,rows,settings['bootstrap_replicates'],settings['seed'],len(specs))
    full=predictions[predictions.model.eq('logistic')]
    ablations={name:paired(full,predictions[predictions.model.eq(name)],settings['bootstrap_replicates'],settings['seed'],len(specs))
               for name in specs if name.startswith(('without_','volume_window_'))}
    result=dict(protocol=protocol,metrics=summaries,vs_v1=comparisons,ablation_vs_full=ablations,folds=folds,
                unavailable_families=settings['unavailable_families'],promotion='NOT_ELIGIBLE_REUSED_HISTORY_NO_PROSPECTIVE_VALIDATION',roi=None)
    write(output/'report.json',result)
    return result
