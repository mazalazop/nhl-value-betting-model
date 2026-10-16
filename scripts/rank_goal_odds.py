"""Join strict GOAL 1+ odds to research rankings without changing probabilities."""
import argparse,json,sys
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'model'))
from henachel.goal_odds import match_goal_odds,annotate_probability_ranking


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions',type=Path,default=Path('outputs/predictions_upcoming_goal_research.csv'))
    p.add_argument('--odds',type=Path,default=Path('outputs/structured_bookmaker/normalized_goals_odds.json'))
    p.add_argument('--output-dir',type=Path,default=Path('outputs'));a=p.parse_args()
    predictions=pd.read_csv(a.predictions)
    rows=json.loads(a.odds.read_text())['rows'] if a.odds.exists() else []
    matched,rejected=match_goal_odds(predictions,rows)
    a.output_dir.mkdir(parents=True,exist_ok=True)
    matched.to_csv(a.output_dir/'goal_value_ranking.csv',index=False)
    annotate_probability_ranking(predictions,matched).to_csv(a.output_dir/'goal_probability_ranking.csv',index=False)
    matched[matched.experimental_value_eligible].to_csv(a.output_dir/'goal_value_shortlist.csv',index=False)
    report=dict(source_available=a.odds.exists(),accepted=len(matched),rejected=rejected,value_eligible=int(matched.experimental_value_eligible.sum()),model_status='research_only_operational')
    (a.output_dir/'goal_matching_summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
