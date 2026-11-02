"""Publish aggregate research evidence only; never copy raw third-party observations."""
import argparse,json,shutil,sys
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import norm
from core import sha,save

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path('outputs/v3'));p.add_argument('--output',type=Path,default=Path('research/v3/results'));p.add_argument('--a01',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
 sources={
 'inventory.json':a.workspace/'audit-verified/inventory.json',
 'missingness.csv':a.workspace/'audit-verified/missingness.csv',
 'canonical.json':a.workspace/'audit-verified/canonical-report.json',
 'groups.json':a.workspace/'audit-verified/groups.json',
 'main.json':a.workspace/'models-main/report-adjusted.json',
 'quality.json':a.workspace/'models-quality-aligned/report.json',
 'quality_coverage.json':a.workspace/'shot-features-complete.coverage.json',
 'original_prospective.json':a.workspace/'original-prospective/report.json',
 'a01.json':a.a01,
 'history_receipt.json':a.workspace/'durable-source/receipt.json',
 'postgame_inputs.json':a.workspace/'audit-verified/inputs.json',
 'postgame_revisions.json':a.workspace/'audit-verified/revision_inputs.json',
 'main_feature_missingness.json':a.workspace/'models-main/feature_missingness.json'}
 for name,source in sources.items():shutil.copyfile(source,a.output/name)
 rows=[];contrasts=[];folds=[];power=[]
 for experiment in ['main','quality']:
  report=json.loads((a.output/(experiment+'.json')).read_text())
  for market,r in report['results'].items():
   for variant,m in r['metrics'].items():
    v=m['all'];rows.append(dict(experiment=experiment,market=market,variant=variant,**{k:v[k] for k in ['n','dates','games','brier','log_loss','hit_rate','ece']},top5_n=v['top5']['n'],top5_hit=v['top5']['hit_rate'],top10_n=v['top10']['n'],top10_hit=v['top10']['hit_rate'],coverage_paired=1,roi='NA_NO_PREGAME_ODDS'))
    for fold,f in m['folds'].items():folds.append(dict(experiment=experiment,market=market,variant=variant,fold=fold,n=f['n'],brier=f['brier'],log_loss=f['log_loss'],top10_hit=f['top10']['hit_rate']))
   for kind in ['vs_reference','ablation_vs_full','marginal']:
    for variant,c in r.get(kind,{}).items():
     for metric in ['brier','log_loss']:
      v=c[metric];contrasts.append(dict(experiment=experiment,market=market,kind=kind,variant=variant,metric=metric,n=c['n'],dates=c['dates'],delta=v['delta'],low=v['simultaneous_ci'][0],high=v['simultaneous_ci'][1],se=v['bootstrap_se'],family=c['family_size']))
   if experiment in ['main','quality']:
    candidate='full_sigmoid' if experiment=='main' else 'volume_quality'
    c=r['vs_reference'][candidate]
    for scale in [.5,1,2]:
     need={metric:int(np.ceil(c['dates']*((norm.ppf(1-.025/2)+norm.ppf(.9))*c[metric]['bootstrap_se']/(gain*scale))**2)) for metric,gain in [('brier',.001),('log_loss',.003)]}
     power.append(dict(experiment=experiment,candidate=candidate,market=market,brier_gain=.001*scale,log_loss_gain=.003*scale,required_dates_brier=need['brier'],required_dates_log_loss=need['log_loss'],required_dates_joint=max(need.values()),approximate_pairs=max(need.values())*c['n']/c['dates'],alpha_per_endpoint=.025,power_per_endpoint=.9,joint_power_lower_bound=.8,assumption='Historical paired 7-date-block variance; conditional on same composition, not a guarantee'))
 pd.DataFrame(rows).to_csv(a.output/'comparisons.csv',index=False);pd.DataFrame(contrasts).to_csv(a.output/'contrasts.csv',index=False);pd.DataFrame(folds).to_csv(a.output/'fold_metrics.csv',index=False);pd.DataFrame(power).to_csv(a.output/'power_sensitivity.csv',index=False)
 evidence={str(source):sha(source) for source in sources.values()}
 for path in [a.workspace/'models-main/predictions.csv.gz',a.workspace/'models-quality-aligned/predictions.csv.gz',a.workspace/'shot-features-complete.csv']:
  evidence[str(path)]=sha(path)

 if (a.workspace/'plot-libs').exists():sys.path.insert(0,str(a.workspace/'plot-libs'))
 import matplotlib
 matplotlib.use('Agg');matplotlib.rcParams['svg.hashsalt']='henachel-v3-1'
 import matplotlib.pyplot as plt
 for experiment in ['main','quality']:
  report=json.loads((a.output/(experiment+'.json')).read_text());fig,axes=plt.subplots(1,2,figsize=(10,4))
  for ax,market in zip(axes,['POINT','BUT']):
   names=['reference','full_sigmoid'] if experiment=='main' else ['reference','volume','volume_quality']
   for name in names:
    bins=report['results'][market]['metrics'][name]['all']['calibration'];bins=[b for b in bins if b['n']>=30]
    ax.plot([b['predicted'] for b in bins],[b['observed'] for b in bins],'o-',label=name)
   ax.plot([0,1],[0,1],'k--',linewidth=.7);ax.set(title=market,xlabel='Probabilité moyenne',ylabel='Fréquence observée',xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8)
  fig.suptitle('Exploration historique — intervalles avec au moins 30 observations');fig.tight_layout();fig.savefig(a.output/(experiment+'_calibration.svg'),metadata={'Date':None});plt.close(fig)
 save(a.output/'evidence_manifest.json',dict(inputs=evidence,classification='EXPLORATORY_REUSED_PERIODS',active_models_changed=False,raw_data_redistributed=False,files={p.name:sha(p) for p in a.output.iterdir() if p.is_file() and p.name!='evidence_manifest.json'}))
 print('Published aggregate evidence')
if __name__=='__main__':main()
