"""Selected temporal diagnostics of archived native features, not full PIT certification."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from core import save

def main():
 p=argparse.ArgumentParser();p.add_argument('--history',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 h=pd.read_csv(a.history,low_memory=False).sort_values(['id_joueur','season_source','date_match','id_match']);dates=pd.to_datetime(h.date_match);r={}
 for c in ['standings_lookup_date_pre','team_standings_date_pre','tirs_moy_5','tirs_moy_10']:
  if c not in h:continue
  if 'date' in c:
   t=pd.to_datetime(h[c],errors='coerce');r[c]=dict(known=int(t.notna().sum()),strictly_after_game_date=int((t>dates).sum()),same_date=int((t==dates).sum()))
  else:
   w=int(c.rsplit('_',1)[1]);shifted=h.groupby(['id_joueur','season_source']).tirs.transform(lambda s:s.shift().rolling(w,min_periods=1).mean());known=shifted.notna() & h[c].notna()
   r[c]=dict(season_reset_compared=int(known.sum()),season_reset_differences_over_1e_6=int((np.abs(shifted[known]-h.loc[known,c])>1e-6).sum()))
   across=h.groupby('id_joueur').tirs.transform(lambda s:s.shift().rolling(w,min_periods=1).mean());known=across.notna() & h[c].notna()
   r[c].update(native_across_seasons_compared=int(known.sum()),native_across_seasons_differences_over_1e_6=int((np.abs(across[known]-h.loc[known,c])>1e-6).sum()))
 save(a.output,r)
if __name__=='__main__':main()
