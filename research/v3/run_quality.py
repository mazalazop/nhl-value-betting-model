"""Predeclared 2x2 volume/geometry experiment on two complete NHL seasons."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json
from pathlib import Path
import pandas as pd
from core import features,FAMILIES,QUALITY,fit_logistic,paired,metrics,save,sha
from run_models import prod,augment_goal_features,fit_goal_model,numeric_features,GOAL_FEATURES
from henachel.calibration import fit_method

def quality_history(h,seasons):
 """Normalize season labels and align the reference with chronological candidate keys."""
 h=h[pd.to_numeric(h.season_source,errors='raise').isin(seasons) & h.temps_de_glace.gt(0)].copy()
 if h.empty:raise ValueError('No rows in declared quality seasons')
 return h.sort_values(['date_match','id_match','id_joueur']).reset_index(drop=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--history',type=Path,required=True);p.add_argument('--quality',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 cfg=json.loads((Path(__file__).parent/'protocol.json').read_text());prod.FEATURES_HISTORY_PATH=a.history;h,target,_=prod.load_history();h=quality_history(h,cfg['quality_scope_seasons'])
 quality=pd.read_csv(a.quality);coverage=quality.id_match.nunique()/h.id_match.nunique()
 if coverage<cfg['quality_min_game_coverage']:
  save(a.output/'report.json',dict(status='INSUFFICIENT_QUALITY_COVERAGE',game_coverage=coverage));return
 f=features(h,quality.drop(columns=['quality_known','source_observed_at_utc','reason']));ref=augment_goal_features(h)
 full=sum(FAMILIES.values(),[]);novol=[c for c in full if c not in FAMILIES['volume']]
 variants={'no_volume_no_quality':novol,'volume':full,'quality':novol+QUALITY,'volume_quality':full+QUALITY,'volume_quality_conversion_gap':full+QUALITY+['conversion_gap']}
 manifest=dict(protocol=cfg,source_sha256=sha(a.history),quality_sha256=sha(a.quality),variants=variants,game_coverage=coverage,geometry_is_xg=False,code_sha256=sha(__file__))
 save(a.output/'manifest-before-fit.json',manifest);dates=sorted(f.date_match.unique());pred=[];folds=[]
 for i in range(cfg['quality_min_train_dates'],len(dates),cfg['quality_test_window_dates']):
  window=dates[i:i+cfg['quality_test_window_dates']];cut=pd.Timestamp(window[0]);boundary=dates[int(i*.85)]
  tr=f[f.date_match<boundary];cal=f[(f.date_match>=boundary)&(f.date_match<cut)];ev=f[f.date_match.isin(window)]
  meta=dict(fold=len(folds),train_end=str(tr.date_match.max()),calibration_start=str(boundary),calibration_end=str(cal.date_match.max()),test_start=str(cut),test_end=str(window[-1]),n=len(ev),quality_feature_nonmissing={c:float(ev[c].notna().mean()) for c in QUALITY})
  for market,outcome in [('POINT','points'),('BUT','buts')]:
   base=ev[['date_match','id_match','id_joueur','season_source','is_home_player','team_player_match']].copy();base['market']=market;base['fold']=len(folds);base['y']=ev[outcome].ge(1).astype(int)
   def add(name,p):
    row=base.copy();row['variant']=name;row['probability']=p;pred.append(row)
   if market=='POINT':
    model,c,info=prod.fit_point_model_and_calibrator(h,target,'date_match',cut);r=h[h.date_match.isin(window)];raw=model.predict_proba(prod.to_numeric_frame(r,info['feature_cols_kept']))[:,1]
   else:
    model,c,info=fit_goal_model(ref,cut,GOAL_FEATURES);r=ref[ref.date_match.isin(window)];raw=model.predict_proba(numeric_features(r,info['feature_cols_kept']))[:,1]
   assert list(zip(r.id_match,r.id_joueur))==list(zip(ev.id_match,ev.id_joueur))
   add('reference',c.predict(raw))
   for variant,cols in variants.items():
    model,kept=fit_logistic(tr,cols,tr[outcome].ge(1).astype(int));cp=model.predict_proba(cal[kept])[:,1];tp=model.predict_proba(ev[kept])[:,1]
    c=fit_method('sigmoid',cp,cal[outcome].ge(1).astype(int));add(variant,c.predict(tp))
   print(market,'quality fold',len(folds),'done',flush=True)
  folds.append(meta)
 pred=pd.concat(pred,ignore_index=True);pred.to_csv(a.output/'predictions.csv.gz',index=False);results={}
 for market,g in pred.groupby('market'):
  reference=g[g.variant.eq('reference')];r=dict(metrics={},vs_reference={},marginal={})
  for name,rows in g.groupby('variant'):
   r['metrics'][name]=dict(all=metrics(rows),folds={str(k):metrics(q) for k,q in rows.groupby('fold')})
   if name!='reference':r['vs_reference'][name]=paired(reference,rows,cfg['simultaneous_family_size'],cfg['bootstrap_replicates'])
  for base,candidate in [('volume','volume_quality'),('quality','volume_quality'),('no_volume_no_quality','quality'),('no_volume_no_quality','volume'),('volume_quality','volume_quality_conversion_gap')]:
   r['marginal'][candidate+'_vs_'+base]=paired(g[g.variant.eq(base)],g[g.variant.eq(candidate)],cfg['simultaneous_family_size'],cfg['bootstrap_replicates'])
  results[market]=r
 save(a.output/'report.json',dict(manifest=manifest,folds=folds,results=results,status='EXPLORATORY_NO_XG_NO_PROMOTION'))
if __name__=='__main__':main()
