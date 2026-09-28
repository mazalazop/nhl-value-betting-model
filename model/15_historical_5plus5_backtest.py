#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backtest historique de la sélection Henachel 5 POINTS + 5 BUTS.

Ce script ne fabrique jamais d'historique de cotes. Il consomme uniquement des
CSV historiques de candidats déjà matchés par les scripts 06.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"

def load_files(pattern: str) -> pd.DataFrame:
    files=sorted(Path(ROOT).glob(pattern))
    if not files:
        return pd.DataFrame()
    frames=[]
    for p in files:
        df=pd.read_csv(p,low_memory=False)
        if not df.empty: frames.append(df)
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()

def settle(df: pd.DataFrame, group: str) -> pd.DataFrame:
    if df.empty: return df
    out=df.copy()
    stat=pd.to_numeric(out.get("actual_stat_value"),errors="coerce")
    threshold=pd.to_numeric(out.get("threshold"),errors="coerce").fillna(1)
    out["won"]=(stat>=threshold).astype("Int64")
    out["pnl"]=out["won"].map({1:out["odds_decimal"]-1,0:-1})
    out.loc[out["won"].isna(),"pnl"]=pd.NA
    out["pick_market_group"]=group
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--points-glob",default="outputs/history/matched_points/*.csv")
    ap.add_argument("--goals-glob",default="outputs/history/matched_goals/*.csv")
    ap.add_argument("--output",default="outputs/historical_5plus5_backtest.csv")
    a=ap.parse_args()
    points=load_files(a.points_glob); goals=load_files(a.goals_glob)
    if points.empty and goals.empty:
        summary={"status":"not_run","reason":"no_historical_matched_odds_files","points_glob":a.points_glob,"goals_glob":a.goals_glob}
        (OUT/"historical_5plus5_backtest_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
        print(json.dumps(summary,ensure_ascii=False,indent=2)); return
    # Import the production selector.
    import importlib.util
    spec=importlib.util.spec_from_file_location("selection",ROOT/"model/07_build_daily_bets.py")
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    selected=[]
    for group,df in [("points",points),("goals",goals)]:
        if df.empty: continue
        if group=="goals":
            df=df.copy()
            df["no_point_drought_alert_pre"]=pd.to_numeric(df.get("goal_drought_alert_pre",0),errors="coerce").fillna(0)
            df["no_point_streak_excess_pre"]=pd.to_numeric(df.get("goal_streak_excess_pre",0),errors="coerce").fillna(0)
        for run_date in sorted(df["run_date"].dropna().astype(str).unique()):
            x,_=mod.build_daily_bets(df,run_date,5,1.01,.50,-1.0,.02,.90,True,False)
            if not x.empty:
                x["pick_market_group"]=group; selected.append(x)
    if not selected:
        summary={"status":"not_run","reason":"no_historical_picks_after_selection"}
        (OUT/"historical_5plus5_backtest_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(summary,ensure_ascii=False,indent=2)); return
    result=pd.concat(selected,ignore_index=True)
    result=settle(result,result["pick_market_group"])
    result.to_csv(a.output,index=False)
    summary={
        "status":"ok","rows":int(len(result)),"days":int(result["run_date"].nunique()),
        "points_picks":int((result.pick_market_group=="points").sum()),
        "goals_picks":int((result.pick_market_group=="goals").sum()),
        "settled_rows":int(result.pnl.notna().sum()),
        "win_rate":float(result.won.dropna().mean()) if result.won.notna().any() else None,
        "roi_per_unit":float(result.pnl.dropna().mean()) if result.pnl.notna().any() else None,
        "note":"ROI is descriptive historical performance only; it is not evidence of future profitability."
    }
    (OUT/"historical_5plus5_backtest_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
