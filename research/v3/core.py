"""V3 research-only features and inference. No imports from operational workflows."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm

KEYS=['date_match','id_match','id_joueur']
FAMILIES={
 'volume':[f'shots_{w}' for w in [3,5,10]]+['shots60_10'],
 'usage':[f'{v}_{w}' for v in ['toi','pp'] for w in [3,5,10]],
 'role_change':['toi_change','pp_change'],
 'offense':['points_10','goals_10','assists_10','prior_point_rate','prior_goal_rate'],
 'context':['team_goals_pre','opponent_ga_pre'],
 'calendar':['home','rest_days','team_rest','opponent_rest'],
 'drought':['goal_drought','point_drought'],
 'hot_streak':['goal_hot','point_hot'],
}
QUALITY=['attempts_pre10','unblocked_close_pre10','distance_pre10','angle_pre10','rebound_pre10']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False,default=str)+'\n')
def validate(d):
 if d.duplicated(['id_match','id_joueur']).any() or d[KEYS+['season_source','team_player_match']].isna().any().any():raise ValueError('Invalid/duplicate identity')
 if d.duplicated(['date_match','id_joueur']).any():raise ValueError('Ambiguous within-date ordering')
 for c in ['buts','points','passes','tirs','temps_de_glace','temps_pp']:
  x=pd.to_numeric(d[c],errors='raise')
  if ((x.dropna()<0)|~np.isfinite(x.dropna())).any():raise ValueError('Invalid measurement')
 for c in ['buts','points']:
  if d[c].isna().any() or (d[c]%1!=0).any():raise ValueError('Unknown target')
 if not d.status.isin(['OFF','FINAL']).all() or not d.game_type.isin([2,3]).all():raise ValueError('Non-final/unsupported game')
 if not (d.temps_de_glace>0).all():raise ValueError('Historical participant universe only')

def features(d,quality=None):
 d=d.copy();d['date_match']=pd.to_datetime(d.date_match);validate(d)
 d=d.sort_values(['id_joueur','season_source','date_match','id_match']).reset_index(drop=True)
 g=d.groupby(['id_joueur','season_source'],sort=False)
 cols=[]
 for src,label in [('tirs','shots'),('temps_de_glace','toi'),('temps_pp','pp'),('points','points'),('buts','goals'),('passes','assists')]:
  for w in ([3,5,10] if label in ['shots','toi','pp'] else [10]):
   name=f'{label}_{w}';d[name]=g[src].transform(lambda s:s.shift().rolling(w,min_periods=1).mean());cols.append(name)
 for source,label in [('buts','goal'),('points','point')]:
  d['_hit']=d[source].ge(1).astype(float)
  d['prior_'+label+'_rate']=d.groupby(['id_joueur','season_source'])._hit.transform(lambda s:s.shift().expanding().mean())
  drought=np.full(len(d),np.nan);hot=drought.copy()
  for _,indices in g.indices.items():
   n0=n1=0
   for count,i in enumerate(indices):
    if count:drought[i]=n0;hot[i]=n1
    if d.at[i,source]>0:n0=0;n1+=1
    else:n0+=1;n1=0
  d[label+'_drought']=drought;d[label+'_hot']=hot
 # Historical conversion residual, not xG: earlier individual shooting rate excludes the recent ten games.
 past_goals=g.buts.transform(lambda s:s.shift().expanding().sum())
 past_shots=g.tirs.transform(lambda s:s.shift().expanding().sum())
 recent_goals=g.buts.transform(lambda s:s.shift().rolling(10,min_periods=10).sum())
 recent_shots=g.tirs.transform(lambda s:s.shift().rolling(10,min_periods=10).sum())
 earlier_shots=past_shots-recent_shots
 earlier_rate=(past_goals-recent_goals)/earlier_shots.where(earlier_shots>=50)
 d['conversion_gap']=recent_goals-recent_shots*earlier_rate
 d['shots60_10']=60*d.shots_10/d.toi_10.where(d.toi_10>0)
 d['toi_change']=d.toi_3-d.toi_10;d['pp_change']=d.pp_3-d.pp_10
 d['rest_days']=g.date_match.diff().dt.days;d['home']=d.is_home_player
 teams=d.drop_duplicates(['id_match','team_player_match']).sort_values(['date_match','id_match']).copy()
 tg=teams.groupby(['team_player_match','season_source'])
 teams['team_goals_pre']=tg.buts_match_equipe.transform(lambda s:s.shift().rolling(10,min_periods=1).mean())
 teams['ga_pre']=tg.buts_match_adversaire.transform(lambda s:s.shift().rolling(10,min_periods=1).mean())
 teams['team_rest']=tg.date_match.diff().dt.days
 d=d.merge(teams[['id_match','team_player_match','team_goals_pre','team_rest']],on=['id_match','team_player_match'],validate='many_to_one')
 opp=teams[['id_match','team_player_match','ga_pre','team_rest']].rename(columns={'team_player_match':'adversaire_match','ga_pre':'opponent_ga_pre','team_rest':'opponent_rest'})
 d=d.merge(opp,on=['id_match','adversaire_match'],how='left',validate='many_to_one')
 if quality is not None:
  d=d.merge(quality,on=['id_match','id_joueur'],how='left',validate='one_to_one').sort_values(['id_joueur','season_source','date_match'])
  for src,label in [('attempts','attempts'),('unblocked_close','unblocked_close'),('distance','distance'),('angle','angle'),('rebounds','rebound')]:
   d[label+'_pre10']=d.groupby(['id_joueur','season_source'])[src].transform(lambda s:s.shift().rolling(10,min_periods=3).mean())
 return d.sort_values(KEYS).reset_index(drop=True)

def losses(y,p):
 y=np.asarray(y);p=np.clip(np.asarray(p),1e-6,1-1e-6)
 return (y-p)**2,-(y*np.log(p)+(1-y)*np.log1p(-p))

def metrics(d):
 if d.empty:return dict(n=0,reason='NO_ELIGIBLE_ROWS')
 br,ll=losses(d.y,d.probability);bins=[]
 for low in np.arange(0,1,.1):
  q=d[(d.probability>=low)&(d.probability<low+.1+ (1e-12 if low>.89 else 0))]
  bins.append(dict(lower=float(low),n=len(q),predicted=float(q.probability.mean()) if len(q) else None,observed=float(q.y.mean()) if len(q) else None))
 out=dict(n=len(d),dates=d.date_match.nunique(),games=d.id_match.nunique(),brier=float(br.mean()),log_loss=float(ll.mean()),hit_rate=float(d.y.mean()),ece=sum(b['n']*abs(b['predicted']-b['observed']) for b in bins if b['n'])/len(d),calibration=bins,roi=None,roi_reason='No historical pregame odds in this participant replay')
 for k in [5,10]:
  top=d.sort_values(['date_match','probability','id_match','id_joueur'],ascending=[True,False,True,True]).groupby('date_match').head(k)
  out[f'top{k}']=dict(n=len(top),hit_rate=float(top.y.mean()))
 return out

def paired(a,b,family_size=1,replicates=4000):
 a=a.set_index(KEYS).sort_index();b=b.set_index(KEYS).sort_index();pd.testing.assert_index_equal(a.index,b.index)
 if not a.index.is_unique:raise ValueError('Duplicate paired keys')
 np.testing.assert_array_equal(a.y,b.y)
 ba,la=losses(a.y,a.probability);bb,lb=losses(b.y,b.probability)
 daily=pd.DataFrame({'brier':bb-ba,'log_loss':lb-la,'n':1},index=a.index.get_level_values(0)).groupby(level=0).sum()
 n=len(daily);rng=np.random.default_rng(20261101)
 # Bounded-memory blocks; retain entire date slates and their nested games.
 draws={name:[] for name in ['brier','log_loss']}
 for start in range(0,replicates,500):
  batch=min(500,replicates-start)
  ix=((rng.integers(0,n,(batch,(n+6)//7))[:,:,None]+np.arange(7))%n).reshape(batch,-1)[:,:n]
  denom=daily.n.to_numpy()[ix].sum(1)
  for name in draws:draws[name].append(daily[name].to_numpy()[ix].sum(1)/denom)
 out=dict(n=len(a),dates=n,method='paired circular blocks of 7 dates; entire games kept inside dates',family_size=family_size,replicates=replicates)
 for name in ['brier','log_loss']:
  values=np.concatenate(draws[name])
  out[name]=dict(delta=float(daily[name].sum()/daily.n.sum()),ci95=np.quantile(values,[.025,.975]).tolist(),simultaneous_ci=np.quantile(values,[.025/family_size,1-.025/family_size]).tolist(),bootstrap_se=float(values.std(ddof=1)))
 return out

def fit_logistic(train,columns,y):
 kept=[c for c in columns if train[c].notna().any() and train[c].nunique()>1]
 if not kept:raise ValueError('No training features')
 model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True),StandardScaler(),LogisticRegression(C=1,max_iter=2000,random_state=20261101))
 model.fit(train[kept],y)
 return model,kept

def sample_size(comparison,gain=.001,alpha=.025,power=.8):
 # Extrapolation of observed date-cluster uncertainty; sensitivity only, not guaranteed power.
 se=comparison['log_loss']['bootstrap_se'];dates=comparison['dates']
 need=int(np.ceil(dates*((norm.ppf(1-alpha/2)+norm.ppf(power))*se/gain)**2))
 return dict(target_gain=gain,alpha=alpha,power=power,estimated_dates=need,assumption='same date-cluster variance and composition; no variance drift',observed_dates=dates)
