"""Rebuild development features via production functions; quarantine reserved dates."""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model'))
from henachel.features import build_feature_frame, build_future_features
from henachel.data import validate_player_games, final_mask
from henachel.goal import BASE_FEATURES,FEATURE_FAMILIES,augment_goal_features
from research_goal_temporal import augment_research_features,FAMILIES,development_only

def build(source):
    source=development_only(source)
    validate_player_games(source)
    if not final_mask(source).all():raise ValueError('Non-final source')
    features=build_feature_frame(source)
    features=features.merge(source[['id_match','id_joueur','game_type']],on=['id_match','id_joueur'],validate='one_to_one')
    if len(features)!=len(source):raise ValueError('Unexpected row filtering')
    return features

def verify_parity(source,features):
    source=development_only(source);all_goal=augment_goal_features(features);all_new=augment_research_features(all_goal)
    columns=BASE_FEATURES+FEATURE_FAMILIES['drought']+sum(FAMILIES.values(),[])
    available=sorted(features.date_match.unique());dates=[]
    for wanted in ['2022-10-07','2023-01-01','2023-10-10','2024-01-15']:
        candidates=[d for d in available if pd.Timestamp(d)>=pd.Timestamp(wanted)]
        if candidates:dates.append(candidates[0])
    reports=[]
    for date in dates:
        current=source[pd.to_datetime(source.date_match)==pd.Timestamp(date)]
        matches=current.drop_duplicates('id_match')[['id_match','date_match','season_source','id_equipe_domicile','id_equipe_exterieur']].rename(columns={'season_source':'saison'})
        players=current[['id_joueur','team_player_match','nom','position']].drop_duplicates('id_joueur')
        future=build_future_features(source,matches,players)
        history=features[features.date_match<pd.Timestamp(date)]
        future=augment_goal_features(future,history)
        future=augment_research_features(future,all_goal[all_goal.date_match<pd.Timestamp(date)])
        a=all_new[all_new.date_match==pd.Timestamp(date)].set_index(['id_match','id_joueur']).sort_index()
        b=future.set_index(['id_match','id_joueur']).sort_index()
        pd.testing.assert_index_equal(a.index,b.index)
        for column in columns:np.testing.assert_allclose(a[column].to_numpy(float),b[column].to_numpy(float),rtol=0,atol=1e-12,equal_nan=True,err_msg=column)
        reports.append(dict(date=str(date),rows=len(a),features=len(columns),divergences=0))
    return reports

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    source=pd.read_csv(a.source,low_memory=False);features=build(source);parity=verify_parity(source,features)
    a.output.parent.mkdir(parents=True,exist_ok=True);features.to_csv(a.output,index=False)
    report=dict(source_sha256=hashlib.sha256(a.source.read_bytes()).hexdigest(),rows=len(features),players=features.id_joueur.nunique(),dates=features.date_match.nunique(),start=str(features.date_match.min()),end=str(features.date_match.max()),parity=parity,holdout_used=False,standings_used=False,standings_note='No baseline GOAL17 or registered feature uses standings; no POINT evaluation claimed.')
    a.output.with_suffix('.report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
