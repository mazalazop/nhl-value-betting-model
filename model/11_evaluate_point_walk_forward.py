#!/usr/bin/env python3
"""Separate scientific evaluation; never participates in daily pick selection."""
import argparse
import importlib.util
import json
from pathlib import Path
from henachel.evaluation import expanding_evaluation

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--features-csv',type=Path,default=ROOT/'data/final/base_features_context_v2.csv');parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs');args=parser.parse_args()
    spec=importlib.util.spec_from_file_location('henachel_future',ROOT/'model/05_predict_upcoming_games.py');future=importlib.util.module_from_spec(spec);spec.loader.exec_module(future)
    future.FEATURES_HISTORY_PATH=args.features_csv
    data,target,date=future.load_history()
    report,predictions=expanding_evaluation(data.rename(columns={date:'date_match'}),future.fit_point_model_and_calibrator,future.FEATURE_WHITELIST,target=target)
    from henachel.manifest import manifest
    report["manifest"] = manifest([args.features_csv])
    args.output_dir.mkdir(parents=True,exist_ok=True)
    predictions.to_csv(args.output_dir/'11_walk_forward_predictions.csv',index=False)
    (args.output_dir/'11_walk_forward_summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))

if __name__=='__main__': main()
