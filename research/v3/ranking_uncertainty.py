"""Secondary descriptive intervals for fixed primary comparators; no selection/p-values."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path('outputs/v3'));p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 out=[]
 for experiment,folder,candidate in [('main','models-main','full_sigmoid'),('quality','models-quality-aligned','volume_quality')]:
  frames=[]
  for chunk in pd.read_csv(a.workspace/folder/'predictions.csv.gz',chunksize=200000,usecols=['date_match','id_match','id_joueur','market','variant','probability','y']):frames.append(chunk[chunk.variant.isin(['reference',candidate])])
  d=pd.concat(frames,ignore_index=True)
  for market,g in d.groupby('market'):
   for k in [5,10]:
    rates={}
    for variant,z in g.groupby('variant'):
     top=z.sort_values(['date_match','probability','id_match','id_joueur'],ascending=[True,False,True,True]).groupby('date_match').head(k)
     rates[variant]=top.groupby('date_match').y.agg(['sum','count']).sort_index()
    ref=rates['reference'];cand=rates[candidate];pd.testing.assert_index_equal(ref.index,cand.index);np.testing.assert_array_equal(ref['count'],cand['count'])
    daily=(cand['sum']-ref['sum']).to_numpy();count=ref['count'].to_numpy();n=len(ref);rng=np.random.default_rng(20261101)
    ix=((rng.integers(0,n,(5000,(n+6)//7))[:,:,None]+np.arange(7))%n).reshape(5000,-1)[:,:n]
    values=daily[ix].sum(1)/count[ix].sum(1);lo,hi=np.quantile(values,[.025,.975])
    out.append(dict(experiment=experiment,market=market,candidate=candidate,top=k,selected=int(count.sum()),dates=n,delta_hit=float(daily.sum()/count.sum()),ci95_low=lo,ci95_high=hi,classification='SECONDARY_DESCRIPTIVE_NOT_MULTIPLICITY_ADJUSTED_NOT_PROSPECTIVE'))
 pd.DataFrame(out).to_csv(a.output,index=False)
if __name__=='__main__':main()
