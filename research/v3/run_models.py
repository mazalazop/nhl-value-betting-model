"""Bounded predeclared V3 walk-forward; isolated files, no publishing or promotion."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,importlib.util,json,sys,subprocess,platform
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'model'))
import numpy as np,pandas as pd
from core import features,FAMILIES,metrics,paired,fit_logistic,sha,save,sample_size
from henachel.calibration import fit_method
from henachel.goal import augment_goal_features,fit_goal_model,numeric_features
from henachel.goal_operational import FEATURES as GOAL_FEATURES
spec=importlib.util.spec_from_file_location('v3_reference_point',ROOT/'model/05_predict_upcoming_games.py')
prod=importlib.util.module_from_spec(spec);spec.loader.exec_module(prod)

def main():
 p=argparse.ArgumentParser();p.add_argument('--history',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 cfg=json.loads((Path(__file__).parent/'protocol.json').read_text());a.output.mkdir(parents=True,exist_ok=False)
 prod.FEATURES_HISTORY_PATH=a.history;history,target,_=prod.load_history()
 history=history[(history.date_match<pd.Timestamp(cfg['history_exclude_from']))&(history.temps_de_glace>0)].sort_values(['date_match','id_match','id_joueur']).reset_index(drop=True)
 dates=sorted(history.date_match.unique());full=sum(FAMILIES.values(),[])
 specs={'full':full,**{'without_'+fam:[c for c in full if c not in cols] for fam,cols in FAMILIES.items()}}
 manifest=dict(protocol=cfg,source_sha256=sha(a.history),code_hashes={str(p.relative_to(ROOT)):sha(p) for p in (Path(__file__).parent).glob('*.py')},base_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),python=platform.python_version(),rows=len(history),seasons=history.season_source.unique().tolist(),variants=specs)
 save(a.output/'manifest-before-fit.json',manifest)
 frame=features(history);reference=augment_goal_features(history)
 save(a.output/'feature_missingness.json',{s:{c:float(g[c].isna().mean()) for c in full} for s,g in frame.groupby('season_source')})
 folds=[];scored=[]
 for i in range(cfg['minimum_train_dates'],len(dates),cfg['test_window_dates']):
  window=dates[i:i+cfg['test_window_dates']];cut=pd.Timestamp(window[0]);cal_start=dates[int(i*.85)]
  train=frame[frame.date_match<cal_start];cal=frame[(frame.date_match>=cal_start)&(frame.date_match<cut)];test=frame[frame.date_match.isin(window)]
  fold=len(folds);meta=dict(fold=fold,train_start=str(train.date_match.min()),train_end=str(train.date_match.max()),calibration_start=str(cal_start),calibration_end=str(cal.date_match.max()),test_start=str(cut),test_end=str(window[-1]),n=len(test),models={})
  for market,outcome in [('POINT','points'),('BUT','buts')]:
   base=test[['date_match','id_match','id_joueur','season_source','team_player_match','is_home_player','game_type']].copy();base['y']=test[outcome].ge(1).astype(int);base['fold']=fold;base['market']=market
   def add(name,prob):
    row=base.copy();row['variant']=name;row['probability']=prob;scored.append(row)
   if market=='POINT':
    model,calibrator,info=prod.fit_point_model_and_calibrator(history,target,'date_match',cut)
    ref=history[history.date_match.isin(window)]
    raw=model.predict_proba(prod.to_numeric_frame(ref,info['feature_cols_kept']))[:,1]
    crows=history[(history.date_match>=cal_start)&(history.date_match<cut)]
    craw=model.predict_proba(prod.to_numeric_frame(crows,info['feature_cols_kept']))[:,1]
   else:
    model,calibrator,info=fit_goal_model(reference,cut,GOAL_FEATURES)
    ref=reference[reference.date_match.isin(window)];raw=model.predict_proba(numeric_features(ref,info['feature_cols_kept']))[:,1]
    crows=reference[(reference.date_match>=cal_start)&(reference.date_match<cut)]
    craw=model.predict_proba(numeric_features(crows,info['feature_cols_kept']))[:,1]
   assert list(zip(ref.id_match,ref.id_joueur))==list(zip(test.id_match,test.id_joueur))
   add('reference',calibrator.predict(raw));meta['models'][market+'_reference']=info
   for method in cfg['calibration_variants']:
    calibrator_variant=fit_method(method,craw,crows[outcome].ge(1).astype(int))
    add('reference_'+method,calibrator_variant.predict(raw))
   for variant,columns in specs.items():
    fitted,kept=fit_logistic(train,columns,train[outcome].ge(1).astype(int))
    cp=fitted.predict_proba(cal[kept])[:,1];tp=fitted.predict_proba(test[kept])[:,1]
    methods=cfg['calibration_variants'] if variant=='full' else [cfg['ablation_calibration']]
    for method in methods:
     calibrator_variant=fit_method(method,cp,cal[outcome].ge(1).astype(int))
     add(variant+'_'+method,calibrator_variant.predict(tp))
    meta['models'][market+'_'+variant]=dict(features=kept)
   print(market,'fold',fold,'test',str(cut.date()),len(test),'done',flush=True)
  folds.append(meta);save(a.output/'folds.json',folds)
  pd.concat(scored,ignore_index=True).to_csv(a.output/'predictions.partial.csv.gz',index=False)
 pred=pd.concat(scored,ignore_index=True);pred.to_csv(a.output/'predictions.csv.gz',index=False)
 results={};comparison_count=cfg['simultaneous_family_size']
 for market,group in pred.groupby('market'):
  ref=group[group.variant.eq('reference')];full=group[group.variant.eq('full_sigmoid')]
  result=dict(metrics={},vs_reference={},ablation_vs_full={},power_sensitivity={})
  for name,rows in group.groupby('variant'):
   result['metrics'][name]=dict(all=metrics(rows),coverage=len(rows)/len(ref),folds={str(k):metrics(g) for k,g in rows.groupby('fold')},seasons={str(k):metrics(g) for k,g in rows.groupby('season_source')},home_away={str(k):metrics(g) for k,g in rows.groupby('is_home_player')})
   if name!='reference':result['vs_reference'][name]=paired(ref,rows,comparison_count,cfg['bootstrap_replicates'])
   if name.startswith('without_'):result['ablation_vs_full'][name]=paired(full,rows,comparison_count,cfg['bootstrap_replicates'])
  result['power_sensitivity']={str(gain):sample_size(result['vs_reference']['full_sigmoid'],gain) for gain in [.001,.003,.005]}
  results[market]=result
 save(a.output/'report.json',dict(manifest=manifest,folds=folds,results=results,conclusion='EXPLORATORY_ONLY_NO_PROMOTION',odds_available=False))
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
