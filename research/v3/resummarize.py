"""Apply declared global multiplicity to stored paired forecasts; no refit/selection."""
import argparse,json,sys
from pathlib import Path
import pandas as pd
from core import paired,save,sample_size,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError(a.output)
 cfg=json.loads((Path(__file__).parent/'protocol.json').read_text());report=json.loads((a.experiment/'report.json').read_text());pred=pd.read_csv(a.experiment/'predictions.csv.gz')
 for market,g in pred.groupby('market'):
  r=report['results'][market];ref=g[g.variant.eq('reference')];full=g[g.variant.eq('full_sigmoid')]
  for name,rows in g.groupby('variant'):
   if name!='reference':r['vs_reference'][name]=paired(ref,rows,cfg['simultaneous_family_size'],cfg['bootstrap_replicates'])
   if name.startswith('without_'):r['ablation_vs_full'][name]=paired(full,rows,cfg['simultaneous_family_size'],cfg['bootstrap_replicates'])
  r['power_sensitivity']={str(gain):sample_size(r['vs_reference']['full_sigmoid'],gain) for gain in [.001,.003,.005]}
 report['inference_manifest']=dict(protocol=cfg,script_sha256=sha(__file__),predictions_sha256=sha(a.experiment/'predictions.csv.gz'),no_predictions_changed=True)
 save(a.output,report)
if __name__=='__main__':main()
