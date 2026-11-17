"""Post-hoc descriptive strata, not a test of an xG deficit or a selection rule."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from core import features,losses

def main():
 p=argparse.ArgumentParser();p.add_argument('--history',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 h=pd.read_csv(a.history);h=h[h.season_source.isin([20242025,20252026]) & h.temps_de_glace.gt(0) & h.status.isin(['OFF','FINAL']) & h.game_type.isin([2,3])]
 f=features(h);f['stratum']=np.select([f.conversion_gap.isna(),f.conversion_gap.lt(-1),f.conversion_gap.gt(1)],['UNKNOWN','BELOW_MINUS_ONE_GOAL','ABOVE_PLUS_ONE_GOAL'],default='WITHIN_ONE_GOAL')
 pred=pd.read_csv(a.predictions);pred=pred.merge(f[['id_match','id_joueur','stratum','shots_10','toi_10','pp_10']],on=['id_match','id_joueur'],validate='many_to_one');out=[]
 for (market,variant,stratum),g in pred.groupby(['market','variant','stratum']):
  br,ll=losses(g.y,g.probability);out.append(dict(market=market,variant=variant,stratum=stratum,n=len(g),dates=g.date_match.nunique(),hit_rate=g.y.mean(),mean_probability=g.probability.mean(),brier=br.mean(),log_loss=ll.mean(),mean_shots10=g.shots_10.mean(),mean_toi10=g.toi_10.mean(),mean_pp10=g.pp_10.mean(),classification='POSTHOC_DESCRIPTIVE_NOT_XG_NOT_CAUSAL'))
 pd.DataFrame(out).to_csv(a.output,index=False)
if __name__=='__main__':main()
