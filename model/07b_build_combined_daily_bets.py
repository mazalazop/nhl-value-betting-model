#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sélection finale: 5 marqueurs de points + 5 buteurs maximum."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from importlib.util import spec_from_file_location,module_from_spec

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"

def load_selection():
    spec=spec_from_file_location("selection",ROOT/"model/07_build_daily_bets.py")
    mod=module_from_spec(spec);spec.loader.exec_module(mod);return mod

def prepare_goal(df):
    df=df.copy()
    df["no_point_drought_alert_pre"]=pd.to_numeric(df.get("goal_drought_alert_pre",0),errors="coerce").fillna(0)
    df["no_point_streak_excess_pre"]=pd.to_numeric(df.get("goal_streak_excess_pre",0),errors="coerce").fillna(0)
    return df

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-date",required=True)
    p.add_argument("--point-csv",default=str(OUT/"06_matched_point_edges.csv"))
    p.add_argument("--goal-csv",default=str(OUT/"06_matched_goal_edges.csv"))
    p.add_argument("--max-points",type=int,default=5)
    p.add_argument("--max-goals",type=int,default=5)
    p.add_argument("--min-model-proba",type=float,default=.50)
    p.add_argument("--min-edge",type=float,default=-1.0)
    a=p.parse_args()
    m=load_selection()
    frames=[]
    for market,path,cap in [("points",Path(a.point_csv),a.max_points),("goals",Path(a.goal_csv),a.max_goals)]:
        if not path.exists(): continue
        df=pd.read_csv(path,low_memory=False)
        if df.empty: continue
        if market=="goals": df=prepare_goal(df)
        selected,_=m.build_daily_bets(df,a.run_date,cap,1.01,a.min_model_proba,a.min_edge,.02,.90,True,False)
        if not selected.empty:
            selected["pick_market_group"]=market
            frames.append(selected)
    combined=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    if not combined.empty:
        combined=combined.sort_values(["pick_market_group","model_probability","edge_probability"],ascending=[True,False,False],kind="stable").reset_index(drop=True)
        combined["recommendation_rank"]=combined.groupby("pick_market_group", sort=False).cumcount()+1
        combined["display_rank"]=combined["recommendation_rank"]
    path=OUT/"07_daily_bets.csv";combined.to_csv(path,index=False)
    print(f"Points: {sum(len(x[x['pick_market_group']=='points']) for x in frames) if frames else 0}")
    print(f"Goals: {sum(len(x[x['pick_market_group']=='goals']) for x in frames) if frames else 0}")
    print(f"Total: {len(combined)}")

if __name__=="__main__": main()
