"""target date -> GOAL17 operational research probabilities; no tuning or bets."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model'))
from henachel.goal_operational import refit,score,top_ranking,STATUS,upcoming_slate
from henachel.goal import numeric_features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target-date',required=True);p.add_argument('--top',type=int,default=10)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--future-features',type=Path);p.add_argument('--output-dir',type=Path,default=Path('outputs'))
    p.add_argument('--pp-status',choices=['PP1','PP2','UNKNOWN']);a=p.parse_args()
    root=a.root.resolve();out=a.output_dir;out.mkdir(parents=True,exist_ok=True)
    module_path=Path(__file__).resolve().parents[1]/'model/05_predict_upcoming_games.py'
    spec=importlib.util.spec_from_file_location('point_universe',module_path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.RAW_DIR=root/'data/raw';m.MATCHS_PATH=m.RAW_DIR/'matchs.csv';m.JOUEURS_PATH=m.RAW_DIR/'joueurs.csv'
    m.TEAM_STANDINGS_PATH=m.RAW_DIR/'team_standings_daily.csv'
    history=pd.read_csv(root/'data/final/base_features_context_v2.csv',low_memory=False)
    matches=m.load_matchs();date=pd.Timestamp(a.target_date);slate=upcoming_slate(matches,date)
    if slate.empty:
        columns=['id_match','id_joueur','team_player_match','nom','proba_goal_1p_calibree','probability_goal','model_status','game_id','player_id','player_name','team','opponent','game_date','event_start_utc','goal_probability_rank']
        pd.DataFrame(columns=columns).to_csv(out/'predictions_upcoming_goal_research.csv',index=False)
        pd.DataFrame(columns=columns).to_csv(out/'goal_probability_top.csv',index=False)
        (out/'goal_operational_summary.json').write_text(json.dumps(dict(status='no_games',model_status=STATUS)))
        return
    bundle,past=refit(history,a.target_date)
    if a.future_features:
        future=pd.read_json(a.future_features,orient='table')
    else:
        teams=sorted(set(slate.id_equipe_domicile)|set(slate.id_equipe_exterieur))
        pool=m.build_recent_player_pool(past,m.load_joueurs(),date,teams,False,45)
        _,_,standings=m.load_standings_optional()
        future=m.build_upcoming_universe(slate,past,matches,pool,standings)
    predictions,derived=score(bundle,past,future,a.target_date)
    artifact=out/'goal_operational_bundle.joblib';joblib.dump(bundle,artifact)
    model_sha=hashlib.sha256(artifact.read_bytes()).hexdigest();predictions['model_sha256']=model_sha
    predictions.to_csv(out/'predictions_upcoming_goal_research.csv',index=False)
    top_ranking(predictions,a.top,a.pp_status).to_csv(out/'goal_probability_top.csv',index=False)
    # Diagnostic only: never use observed slate results or choose a scorer here.
    frozen_path=Path(__file__).resolve().parents[1]/'model/research_artifacts/goal17_frozen_20240201.joblib'
    frozen=joblib.load(frozen_path)
    probabilities=frozen['calibrator'].predict(frozen['model'].predict_proba(numeric_features(derived,frozen['features']))[:,1])
    diagnostic=derived[['id_match','id_joueur']].copy();diagnostic['frozen_probability']=probabilities
    comparison=predictions.merge(diagnostic,on=['id_match','id_joueur'],validate='one_to_one')
    comparison.to_csv(out/'goal_refit_vs_frozen_diagnostic.csv',index=False)
    frozen_top=set(map(tuple,comparison.nlargest(a.top,'frozen_probability')[['id_match','id_joueur']].to_numpy()))
    refit_top=set(map(tuple,predictions.head(a.top)[['id_match','id_joueur']].to_numpy()))
    summary=dict(bundle['summary'],model_sha256=model_sha,slate_games=int(predictions.game_id.nunique()),scored_players=len(predictions),
                 frozen_reference_cutoff=frozen['cutoff'],top_overlap=len(frozen_top&refit_top),
                 probability_distribution=predictions.probability_goal.describe().to_dict(),
                 frozen_distribution=comparison.frozen_probability.describe().to_dict(),
                 refit_frozen_correlation=float(comparison.probability_goal.corr(comparison.frozen_probability)))
    (out/'goal_operational_summary.json').write_text(json.dumps(summary,indent=2,default=str,allow_nan=False))
    print(json.dumps(summary,default=str),flush=True)
    print(top_ranking(predictions,a.top,a.pp_status)[['player_name','team','opponent','probability_goal','pp_status']].to_string(index=False))


if __name__=='__main__':main()
