"""Preregistered isolated structural study; production is never modified."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'model'))
from henachel.features import build_feature_frame, charger_standings, build_future_features
from henachel.data import validate_player_games, final_mask
from henachel.goal import augment_goal_features, goal_labels, numeric_features
from henachel.calibration import select_calibrator
from henachel.evaluation import metrics, reliability_bins
from henachel.goal_experiment import paired_comparison
from henachel.manifest import manifest
from research_goal_temporal import augment_research_features
from structural_features import structural_features

PROTOCOL=ROOT/'docs/structural_20261012/protocol.json'
# The preregistration is immutable, independent of subsequent runner commits.
PROTOCOL_SHA='1b45f8b5e5b6b25b0bf953e9254d9b897de0f4dbfab4257f83692b56a8fe26fc'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol():
    if digest(PROTOCOL)!=PROTOCOL_SHA:raise ValueError('Preregistered protocol changed')
    return json.loads(PROTOCOL.read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=str,allow_nan=False)+'\n')


def labels(frame,target):
    if target=='GOAL':return goal_labels(frame)
    values=pd.to_numeric(frame.points,errors='raise')
    if values.isna().any() or not np.isfinite(values).all() or values.lt(0).any() or values.mod(1).ne(0).any():raise ValueError('Invalid POINT target')
    return values.ge(1).astype(int)


def prepare(history,context,standings_path,output):
    p=protocol()
    for path in [history,context]:
        if digest(path)!=p['inputs'].get(str(path)):raise ValueError('Preregistered input mismatch')
    if output.exists():raise FileExistsError(output)
    raw=pd.read_csv(history,low_memory=False);ctx=pd.read_csv(context)
    validate_player_games(raw)
    if not final_mask(raw).all() or not raw.game_type.eq(2).all():raise ValueError('Invalid research universe')
    standings,info=charger_standings(standings_path)
    if standings is None:raise ValueError('POINT standings required')
    base=build_feature_frame(raw,standings)
    base=base.merge(raw[['id_match','id_joueur','game_type']],on=['id_match','id_joueur'],validate='one_to_one')
    if len(base)!=len(raw):raise ValueError('Silent row filtering')
    goal=augment_goal_features(base);derived=augment_research_features(goal)
    result=structural_features(derived,ctx)
    all_columns=sorted(set(sum(p['variants'].values(),[])+p['baselines']['POINT']+p['baselines']['GOAL']))
    if set(all_columns)-set(result):raise ValueError(f'Missing features {set(all_columns)-set(result)}')
    parity=[]
    # No new fits; pregame values checked at cold start, season turnover and development.
    for requested in ['2022-10-07','2023-10-10','2023-11-01','2024-02-01']:
        dates=sorted(d for d in pd.to_datetime(raw.date_match).unique() if d>=pd.Timestamp(requested))
        date=pd.Timestamp(dates[0]);today=raw[pd.to_datetime(raw.date_match).eq(date)]
        matches=today.drop_duplicates('id_match')[['id_match','date_match','season_source','id_equipe_domicile','id_equipe_exterieur']].rename(columns={'season_source':'saison'})
        players=today[['id_joueur','team_player_match','nom','position']].drop_duplicates('id_joueur')
        upcoming=build_future_features(raw,matches,players,standings)
        past=base[base.date_match.lt(date)]
        upcoming=augment_goal_features(upcoming,past)
        upcoming=augment_research_features(upcoming,goal[goal.date_match.lt(date)])
        upcoming=structural_features(upcoming,ctx,derived[derived.date_match.lt(date)])
        a=result[result.date_match.eq(date)].set_index(['id_match','id_joueur']).sort_index()
        b=upcoming.set_index(['id_match','id_joueur']).sort_index()
        pd.testing.assert_index_equal(a.index,b.index)
        np.testing.assert_allclose(a[all_columns],b[all_columns],rtol=0,atol=1e-10,equal_nan=True)
        parity.append(dict(date=str(date),rows=len(a),features=len(all_columns),divergences=0))
    output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(output,index=False)
    report=dict(rows=len(result),dates=result.date_match.nunique(),players=result.id_joueur.nunique(),sha256=digest(output),parity=parity,standings=info,
                standings_context_found=result.groupby('season_source').standings_context_found.mean().to_dict(),
                missing_by_season={str(s):g[all_columns].isna().mean().to_dict() for s,g in result.groupby('season_source')},
                provenance=manifest([history,context,standings_path,PROTOCOL,Path(__file__),ROOT/'scripts/structural_features.py']))
    write(output.with_suffix('.report.json'),report)
    print(json.dumps({k:report[k] for k in ['rows','dates','players','parity','standings_context_found']},default=str),flush=True)


def fit(frame,cutoff,features,target):
    p=protocol();allowed=set(p['baselines'][target]+p['baselines']['C19']+sum(p['variants'].values(),[]))
    if not features or len(set(features))!=len(features) or set(features)-allowed:raise ValueError('Undeclared features')
    past=frame[pd.to_datetime(frame.date_match).lt(pd.Timestamp(cutoff))].copy()
    past.date_match=pd.to_datetime(past.date_match);past=past.sort_values(['date_match','id_match','id_joueur'])
    y=labels(past,target);dates=sorted(past.date_match.unique())
    if len(dates)<20:raise ValueError('Insufficient training dates')
    cut=dates[int(.85*len(dates))];train=past.date_match.lt(cut);cal=~train
    x=numeric_features(past.loc[train],features);kept=x.columns[x.notna().any() & x.nunique().gt(1)].tolist()
    yt=y.loc[train]
    if not kept or yt.nunique()!=2:raise ValueError('Insufficient features/classes')
    model=HistGradientBoostingClassifier(**p['parameters'])
    model.fit(x[kept],yt,sample_weight=np.where(yt.eq(1),yt.eq(0).sum()/yt.eq(1).sum(),1.))
    raw=model.predict_proba(numeric_features(past.loc[cal],kept))[:,1]
    calibrator,info=select_calibrator(raw,y.loc[cal],past.loc[cal,'date_match'])
    return model,calibrator,dict(kept=kept,fit_end=str(past.loc[train,'date_match'].max()),calibration_end=str(past.loc[cal,'date_match'].max()),calibration=info)


def summary(rows):
    bins=reliability_bins(rows.y,rows.probability)
    return dict(metrics=metrics(rows.y,rows.probability),ece=sum(b['count']*abs(b['mean_probability']-b['observed_rate']) for b in bins if b['count'])/len(rows),calibration=bins)


def compare(a,b):
    return paired_comparison(a.rename(columns={'y':'a_marque_un_but'}),b.rename(columns={'y':'a_marque_un_but'}),family_size=protocol()['selection']['bonferroni_family_size'])


def eligible(comparison,reference,candidate):
    p=protocol()['selection']
    return bool(comparison['passes'] and comparison['delta_logloss']<=-p['minimum_logloss_gain'] and comparison['delta_brier']<0 and candidate['ece']<=reference['ece']+p['max_ece_degradation'])


def run(source,output,target):
    p=protocol();frame=pd.read_csv(source,low_memory=False);frame.date_match=pd.to_datetime(frame.date_match)
    features_report=json.loads(source.with_suffix('.report.json').read_text())
    if features_report['sha256']!=digest(source):raise ValueError('Features changed after parity')
    if frame.duplicated(['id_match','id_joueur']).any() or not frame.game_type.eq(2).all():raise ValueError('Invalid evaluation identities/universe')
    variants={'V0':p['baselines'][target],**p['variants']}
    if target=='GOAL':variants['C19']=p['baselines']['C19']
    output.mkdir(parents=True,exist_ok=True)
    identity=dict(source_sha256=digest(source),protocol_sha256=digest(PROTOCOL),runner_sha256=digest(__file__),target=target)
    ledger=output/'identity.json'
    if ledger.exists():
        if json.loads(ledger.read_text())!=identity:raise ValueError('Incompatible experiment resume')
    else:write(ledger,identity)
    results={};reports={}
    for name,cols in variants.items():
        parts=[];fold_reports=[]
        for fold,(start,end) in enumerate(p['folds']):
            dest=output/f'{name}_{fold}.csv.gz';meta_path=output/f'{name}_{fold}.json'
            if dest.exists() and meta_path.exists():
                meta=json.loads(meta_path.read_text())
                if digest(dest)!=meta['prediction_sha256']:raise ValueError('OOF cache changed')
                rows=pd.read_csv(dest,parse_dates=['date_match'])
            else:
                if dest.exists():raise ValueError('Incomplete OOF checkpoint; inspect before restart')
                model,cal,meta=fit(frame,start,cols,target)
                ev=frame[frame.date_match.ge(start)&frame.date_match.lt(end)]
                if ev.empty:raise ValueError('Empty preregistered fold')
                rows=ev[['date_match','id_match','id_joueur','game_type','position']].copy();rows['y']=labels(ev,target);rows['fold']=fold
                rows['probability']=cal.predict(model.predict_proba(numeric_features(ev,meta['kept']))[:,1])
                past=frame[frame.date_match.lt(start)];prevalence=float(labels(past,target).mean())
                prefix='goal' if target=='GOAL' else 'point'
                rows['player_smoothed']=(ev[f'season_{prefix}_hits_before_match']+10*prevalence)/(ev.season_games_before_match+10)
                rows.to_csv(dest,index=False);meta.update(prediction_sha256=digest(dest),**summary(rows));write(meta_path,meta)
            parts.append(rows);fold_reports.append(meta)
            print(target,name,fold,meta['metrics']['logloss'],flush=True)
        rows=pd.concat(parts,ignore_index=True);results[name]=rows
        reports[name]=dict(**summary(rows),folds=fold_reports,features=cols,rows=len(rows),dates=rows.date_match.nunique(),players=rows.id_joueur.nunique(),positions={label:summary(g) for label,g in rows.assign(segment=np.where(rows.position.eq('D'),'D','F')).groupby('segment')})
    benchmark=results['V0'].copy();benchmark['probability']=benchmark.player_smoothed
    reports['player_smoothed']=summary(benchmark)
    comparisons={};passing=[]
    for name in p['variants']:
        cmp=compare(results['V0'],results[name]);ok=eligible(cmp,reports['V0'],reports[name])
        if target=='GOAL':
            cmp['vs_C19']=compare(results['C19'],results[name]);cmp['vs_player_smoothed']=compare(benchmark,results[name])
            ok=ok and eligible(cmp['vs_C19'],reports['C19'],reports[name]) and eligible(cmp['vs_player_smoothed'],reports['player_smoothed'],reports[name])
        cmp['development_eligible']=ok;comparisons[name]=cmp
        if ok:passing.append(name)
    selected=min(passing,key=lambda n:(len(variants[n]),reports[n]['metrics']['logloss'])) if passing else None
    contrasts={f'{b}_vs_{a}':compare(results[a],results[b]) for a,b in [('V1','V2'),('V1','V3'),('V2','V4'),('V3','V4'),('V4','V5')]}
    report=dict(target=target,reports=reports,comparisons=comparisons,diagnostic_contrasts=contrasts,candidate=selected,production_approved=False,
                independent_confirmation=False,manifest=manifest([source,PROTOCOL,Path(__file__)],p['parameters']))
    write(output/'summary.json',report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    build=sub.add_parser('prepare')
    for name in ['history','context','standings','output']:build.add_argument('--'+name,type=Path,required=True)
    train=sub.add_parser('run');train.add_argument('--features',type=Path,required=True);train.add_argument('--output',type=Path,required=True);train.add_argument('--target',choices=['POINT','GOAL'],required=True)
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.history,a.context,a.standings,a.output)
    else:run(a.features,a.output,a.target)
