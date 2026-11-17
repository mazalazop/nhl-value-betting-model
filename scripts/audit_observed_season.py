"""Read-only metrics on archived canonical picks, never on reconstructed prices.

Ranking/threshold comparisons are simulations within an already selected universe.
No assertion is made that these are the full model Top 3/5/10.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd


def metrics(d):
    settled = d[d.status.eq('SETTLED')].copy()
    result = dict(proposed=len(d), settled=len(settled), unresolved=len(d)-len(settled))
    if settled.empty:
        return dict(result, hit_rate=None, brier=None, log_loss=None, roi_simulated=None)
    y, p = settled.official_win.astype(float), settled.model_probability.astype(float)
    profit = y*settled.odds.astype(float)-1
    result.update(wins=int(y.sum()), hit_rate=float(y.mean()), mean_probability=float(p.mean()),
                  calibration_gap=float(p.mean()-y.mean()), brier=float(((p-y)**2).mean()),
                  log_loss=float(-(y*np.log(p.clip(1e-15,1-1e-15))+(1-y)*np.log1p(-p.clip(1e-15,1-1e-15))).mean()),
                  profit_units=float(profit.sum()), roi_simulated=float(profit.mean()),
                  mean_ev=float((p*settled.odds-1).mean()))
    daily = pd.DataFrame({'profit':profit, 'n':1, 'date':settled.date}).groupby('date')[['profit','n']].sum()
    if len(daily) >= 7:
        rng=np.random.default_rng(20261117); n=len(daily)
        ix=((rng.integers(0,n,(10000,(n+6)//7))[:,:,None]+np.arange(7))%n).reshape(10000,-1)[:,:n]
        draws=daily.profit.to_numpy()[ix].sum(1)/daily.n.to_numpy()[ix].sum(1)
        result['roi_ci95_exploratory']=np.quantile(draws,[.025,.975]).tolist()
    return result


def audit(paths):
    frames=[]; sources=[]
    for p in sorted(paths):
        sources.append(dict(path=p.name, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
        frames.append(pd.read_csv(p))
    d=pd.concat(frames, ignore_index=True)
    if d.bet_id.isna().any() or d.bet_id.duplicated().any(): raise ValueError('Ambiguous bet identity')
    if not d.archive_group.eq('CANONICAL').all(): raise ValueError('Mixed provenance')
    for col in ['captured_at','odds_verified_at_utc','start_time_utc']:
        d[col]=pd.to_datetime(d[col],utc=True,errors='raise')
    if not ((d.captured_at<d.start_time_utc)&(d.odds_verified_at_utc<d.start_time_utc)).all():
        raise ValueError('Unproven pregame observation')
    if not d.bookmaker.str.lower().eq('unibet').all(): raise ValueError('Other bookmaker')
    if not (d.model_probability.between(0,1)&np.isfinite(d.odds)&d.odds.gt(1)).all():
        raise ValueError('Invalid probability or price')
    settled=d.status.eq('SETTLED')
    if not d.loc[settled,'official_win'].isin([0,1]).all(): raise ValueError('Missing result')
    d['implied_probability']=1/d.odds
    d['edge']=d.model_probability-d.implied_probability
    d['ev']=d.model_probability*d.odds-1
    d['week']=pd.to_datetime(d.date).dt.to_period('W-SUN').astype(str)
    d['probability_bin']=pd.cut(d.model_probability, np.linspace(0,1,11), include_lowest=True)
    d['odds_bin']=pd.cut(d.odds,[1,1.35,1.4,1.5,1.6,2,3,np.inf],right=False)
    d['edge_bin']=pd.cut(d.edge,[-np.inf,-.1,0,.05,.1,np.inf],right=False)
    result=dict(sources=sources, date_min=d.date.min(),date_max=d.date.max(),
                method='Observed canonical picks; unit-stake ROI simulation; 10000 circular 7-date bootstrap draws; exploratory, no multiplicity adjustment',
                limitations=['Full prediction universe unavailable in this cohort',
                             'Historical import excluded odds below 1.40: lowering to 1.35 cannot be evaluated here',
                             'Ranking simulations are within proposed picks, not original full-universe Top K',
                             'No evidence of actual stakes or executed bets'], markets={})
    for market in ['POINT','GOAL']:
        q=d[d.market.eq(market)].copy()
        r=dict(overall=metrics(q),segments={}, rankings={},thresholds={})
        for name in ['week','probability_bin','odds_bin','edge_bin']:
            r['segments'][name]={str(k):metrics(g) for k,g in q.groupby(name,observed=True)}
        for policy, cols in [('probability',['model_probability','edge']),('edge',['edge','model_probability']),('ev',['ev','model_probability'])]:
            rank=q.sort_values(cols+['bet_id'],ascending=[False,False,True])
            r['rankings'][policy]={str(k):metrics(rank.groupby('date',sort=False).head(k)) for k in [3,5,10]}
        for threshold in [1.35,1.4,1.5,1.6]: r['thresholds'][str(threshold)]=metrics(q[q.odds.ge(threshold)])
        result['markets'][market]=r
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=audit(a.inputs.glob('*-canonical-picks.csv'))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v['overall'] for k,v in result['markets'].items()},indent=2))
