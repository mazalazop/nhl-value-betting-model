"""Reproducible read-only inventory and grouped canonical value diagnostics."""
import argparse,json,sys,subprocess
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
from sklearn.linear_model import LogisticRegression
from core import sha,save,metrics,losses
ROOT=Path(__file__).resolve().parents[2]

def value_metrics(d,bootstrap=True):
 if d.empty:return dict(n=0,roi=None,reason='NO_ELIGIBLE_PRICED_SELECTION')
 profit=d.y*d.odds-1;daily=pd.DataFrame({'date':d.date_match,'profit':profit,'n':1}).groupby('date').sum()
 balance=np.r_[0,daily.profit.cumsum().to_numpy()];drawdown=float((np.maximum.accumulate(balance)-balance).max())
 ci=None
 if bootstrap and len(daily)>=7:
  rng=np.random.default_rng(20261101);n=len(daily)
  ix=((rng.integers(0,n,(4000,(n+6)//7))[:,:,None]+np.arange(7))%n).reshape(4000,-1)[:,:n]
  v=daily.profit.to_numpy()[ix].sum(1)/daily.n.to_numpy()[ix].sum(1);ci=np.quantile(v,[.025,.975]).tolist()
 return dict(n=len(d),dates=len(daily),wins=int(d.y.sum()),hit_rate=float(d.y.mean()),mean_probability=float(d.probability.mean()),mean_odds=float(d.odds.mean()),mean_winning_odds=float(d.loc[d.y.eq(1),'odds'].mean()) if d.y.any() else None,
             expected_roi=float((d.probability*d.odds-1).mean()),roi=float(profit.mean()),profit_units=float(profit.sum()),roi_ci95_date_blocks=ci,drawdown_units_daily=drawdown,max_player_share=float(d.id_joueur.value_counts(normalize=True).max()),max_team_share=float(d.team.value_counts(normalize=True).max()))

def value_walkforward(d,versions):
 dates=sorted(d.date_match.unique());out=[]
 for day in dates[7:]:
  train=d[d.date_match<day];test=d[d.date_match==day].copy()
  # Only known before the decision: a prior NHL date alone is insufficient around midnight.
  available=versions[pd.to_datetime(versions.postgame_captured_at_utc,utc=True,errors='coerce')<pd.to_datetime(test.captured_at,utc=True).min()].copy()
  available=available.sort_values('postgame_captured_at_utc').drop_duplicates('bet_id',keep='last').set_index('bet_id')
  train=train[train.bet_id.isin(available.index)].copy()
  if len(train):
   train['y']=train.bet_id.map(available.official_win)
   state=train.bet_id.map(available.settlement_status)
   train=train[state.isin(['WIN','LOSS']) & train.y.notna()].copy();train['y']=train.y.astype(int)
  p=np.clip(train.probability,1e-6,1-1e-6)
  if len(train)>=20 and train.y.nunique()==2:
   fit=LogisticRegression(C=1,random_state=20261101).fit(np.log(p/(1-p)).to_numpy()[:,None],train.y)
   z=np.clip(test.probability,1e-6,1-1e-6);test['recalibrated']=fit.predict_proba(np.log(z/(1-z)).to_numpy()[:,None])[:,1]
  else:test['recalibrated']=test.probability
  test['calibration_train_n']=len(train);out.append(test)
 if not out:return {},pd.DataFrame()
 rows=pd.concat(out)
 selections={}
 for prob in ['probability','recalibrated']:
  for name,margin in [('positive_edge',0),('safety_edge',.05),('conservative_discount',.05)]:
   q=rows[rows[prob]-1/rows.odds>margin].copy();q['probability']=q[prob]
   selections[prob+'_'+name]=value_metrics(q)
  # Fixed top-five EV among positive-EV eligible picks; only this archived selection universe.
  q=rows.assign(ev=rows[prob]*rows.odds-1);q=q[q.ev>0].sort_values(['date_match','ev','id_match','id_joueur'],ascending=[True,False,True,True]).groupby('date_match').head(5).copy();q['probability']=q[prob]
  selections[prob+'_top5_ev']=value_metrics(q)
 return dict(classification='EXPLORATORY_REUSED_SELECTION_UNIVERSE',limitations='Postgame availability timestamp gate may leave no eligible recalibration history; no broader player universe or fresh holdout',policies=selections,policy_count=len(selections)),rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--history',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 h=pd.read_csv(a.history,low_memory=False)
 inventory=dict(captured_at_utc=datetime.now(timezone.utc).isoformat(),git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),source=str(a.history),sha256=sha(a.history),rows=len(h),games=h.id_match.nunique(),players=h.id_joueur.nunique(),start=h.date_match.min(),end=h.date_match.max(),duplicates=int(h.duplicated(['id_match','id_joueur']).sum()),historical_point_in_time_verified=False,
 seasons={str(s):dict(rows=len(g),games=g.id_match.nunique(),players=g.id_joueur.nunique(),start=g.date_match.min(),end=g.date_match.max()) for s,g in h.groupby('season_source')})
 save(a.output/'inventory.json',inventory)
 missing=pd.DataFrame([dict(season=s,variable=c,rows=len(g),missing=int(g[c].isna().sum()),missing_rate=float(g[c].isna().mean())) for s,g in h.groupby('season_source') for c in h])
 missing.to_csv(a.output/'missingness.csv',index=False)
 files=sorted((ROOT/'docs/postgame/results').glob('*-picks.csv'))
 # No revisions/ subdirectory. Legacy daily files include CLASSEUR/EXTRA, canonical files CANONICAL.
 frames=[]
 for f in files:
  d=pd.read_csv(f);d['source_file']=str(f.relative_to(ROOT));frames.append(d)
 allrows=pd.concat(frames,ignore_index=True);allrows['date_match']=allrows.date
 grouped=[]
 for (group,market),g in allrows.groupby(['archive_group','market']):
  settled=g[g.settlement_status.isin(['WIN','LOSS'])]
  grouped.append(dict(group=group,market=market,n=len(g),settled=len(settled),wins=int(settled.settlement_status.eq('WIN').sum()),unresolved=len(g)-len(settled),brier_allowed=group=='CANONICAL',roi_allowed=group=='CANONICAL'))
 save(a.output/'groups.json',grouped)
 d=allrows[allrows.archive_group.eq('CANONICAL') & allrows.market.eq('POINT')].copy()
 assert not d.bet_id.duplicated().any()
 for c in ['model_probability','odds','official_win']:d[c]=pd.to_numeric(d[c],errors='coerce')
 cap=pd.to_datetime(d.captured_at,utc=True,errors='coerce');kick=pd.to_datetime(d.start_time_utc,utc=True,errors='coerce');price=pd.to_datetime(d.odds_verified_at_utc,utc=True,errors='coerce')
 eligible=d.archive_provenance.eq('PREGAME_CANONICAL_LEDGER') & cap.lt(kick) & price.lt(kick) & (kick-cap).le(pd.Timedelta(days=7)) & d.model_probability.between(0,1) & d.odds.ge(1.4) & d.settlement_status.isin(['WIN','LOSS'])
 d=d[eligible].copy();d['probability']=d.model_probability;d['y']=d.official_win.astype(int);d['edge']=d.probability-1/d.odds
 # Match the original ledger import fields; postgame result files cannot silently alter predictions.
 for day,g in d.groupby('date'):
  original=pd.read_csv(ROOT/f'docs/postgame/input/{day}-canonical.csv').set_index('bet_id')
  for row in g.to_dict('records'):
   old=original.loc[row['bet_id']]
   for name in ['id_match','id_joueur','model_probability','odds']:
    assert np.isclose(float(old[name]),float(row[name]),rtol=0,atol=1e-14),(day,name)
   for name in ['captured_at','start_time_utc']:assert str(old[name])==str(row[name]),(day,name)
 d['week']=pd.to_datetime(d.date).dt.to_period('W').astype(str);d['month']=d.date.str[:7]
 d['odds_band']=pd.cut(d.odds,[1.4,1.6,1.8,2,3,np.inf],right=False).astype(str)
 d['probability_band']=pd.cut(d.probability,[0,.5,.6,.7,.8,.9,1.00001],right=False).astype(str)
 d['edge_band']=pd.cut(d.edge,[-np.inf,0,.05,.1,.2,np.inf],right=False).astype(str)
 report=dict(as_of_utc=inventory['captured_at_utc'],through_date=d.date.max(),canonical_point=metrics(d),economics=value_metrics(d),eligible=len(d),excluded=len(allrows[allrows.archive_group.eq('CANONICAL')])-len(d),provenance='Existing pregame ledger contract and unchanged original import checked; independent first-publication timestamps not re-certified',segments={})
 for column in ['date','week','month','id_joueur','team','bookmaker','odds_band','probability_band','edge_band']:
  report['segments'][column]={str(k):value_metrics(g,False) for k,g in d.groupby(column)}
 version_files=list((ROOT/'docs/postgame/results/revisions').glob('*-canonical-picks.csv'))+list((ROOT/'docs/postgame/results').glob('*-canonical-picks.csv'))
 versions=pd.concat([pd.read_csv(p) for p in version_files],ignore_index=True)
 versions=versions[versions.bet_id.notna()].copy();versions['official_win']=pd.to_numeric(versions.official_win,errors='coerce')
 save(a.output/'revision_inputs.json',{str(p.relative_to(ROOT)):sha(p) for p in version_files})
 report['value_walkforward'],policies=value_walkforward(d,versions)
 d.to_csv(a.output/'canonical-analysis.csv',index=False)
 if not policies.empty:policies.to_csv(a.output/'value-replay.csv',index=False)
 save(a.output/'canonical-report.json',report)
 save(a.output/'inputs.json',{str(f.relative_to(ROOT)):sha(f) for f in files})
 # Retrospective diagnostic joins are not silently accepted as pregame contextual information.
 save(a.output/'unavailable_segments.json',{k:'No linked, independently timestamped pregame snapshot in this canonical pick ledger' for k in ['PP1','goalie','rest','home_away','historical_shot_volume','line_partners','xG']})
 print(inventory);print(report['economics'])
if __name__=='__main__':main()
