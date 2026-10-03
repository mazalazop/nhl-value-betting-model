#!/usr/bin/env python3
"""Auditable pregame context overlay. No unvalidated probability boosts.

Input context CSV (optional): run_date,player_name,team,opponent,market,
participation_status,source_url,checked_at_utc,goalie_status,goalie_name,
line_status,pp_role_status,injury_status,rest_status,press_note.
One row per player/market; participation_status: confirmed_out, confirmed_in,
probable, uncertain. Only confirmed_out with dated source triggers exclusion.
Unknown context is explicitly flagged, never interpreted as favorable.
"""
import argparse
import json
from pathlib import Path
import pandas as pd

CONTEXT_FIELDS = ["participation_status","source_url","checked_at_utc","goalie_status",
                  "goalie_name","line_status","pp_role_status","injury_status",
                  "rest_status","press_note"]
KEY = ["run_date","player_name","team","opponent","market"]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--run-date",required=True)
    p.add_argument("--context-csv",default="data/context/pregame_context.csv")
    p.add_argument("--points-csv",default="outputs/06_matched_point_edges.csv")
    p.add_argument("--goals-csv",default="outputs/06b_matched_goal_edges.csv")
    p.add_argument("--output-dir",default="outputs/context_overlay")
    p.add_argument("--min-odds",type=float,default=1.40)
    p.add_argument("--max-picks",type=int,default=10)
    a=p.parse_args()
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    cp=Path(a.context_csv)
    ctx=pd.read_csv(cp,dtype=str).fillna("") if cp.exists() else pd.DataFrame(columns=KEY+CONTEXT_FIELDS)
    if not ctx.empty:
        missing=set(KEY)-set(ctx.columns)
        if missing: raise ValueError(f"Context keys missing: {missing}")
        if ctx.duplicated(KEY).any(): raise ValueError("Duplicate context records: resolve sources first")
        for col in CONTEXT_FIELDS:
            if col not in ctx:ctx[col]=""
    report={}
    for market,path in [("POINTS",a.points_csv),("BUTEURS",a.goals_csv)]:
        df=pd.read_csv(path,low_memory=False)
        df=df[df.run_date.astype(str)==a.run_date].copy()
        df["market"]=market
        for col in KEY:df[col]=df[col].fillna("").astype(str)
        subset=ctx[ctx.market==market] if not ctx.empty else ctx
        df=df.merge(subset[KEY+CONTEXT_FIELDS],on=KEY,how="left",validate="many_to_one")
        for col in CONTEXT_FIELDS:df[col]=df[col].fillna("")
        df["odds_decimal"]=pd.to_numeric(df.odds_decimal,errors="coerce")
        df["model_probability"]=pd.to_numeric(df.model_probability,errors="coerce")
        df["source_verified"]=df.source_url.str.startswith(("https://","http://")) & (df.checked_at_utc!="")
        df["excluded_confirmed_absent"]=(df.participation_status=="confirmed_out") & df.source_verified
        df["context_complete"]=df.source_verified & (df.goalie_status!="") & (df.line_status!="") & (df.injury_status!="")
        df["context_alert"]=df.apply(lambda r: "; ".join(
            (["confirmed_absent"] if r.excluded_confirmed_absent else [])+
            (["unverified_absence_report"] if r.participation_status=="confirmed_out" and not r.source_verified else [])+
            (["participation_unconfirmed"] if r.participation_status not in ("confirmed_in","confirmed_out") else [])+
            (["goalie_unconfirmed"] if r.goalie_status not in ("confirmed","confirmed_starter") else [])+
            (["line_unconfirmed"] if r.line_status!="confirmed" else [])+
            (["injury_unverified"] if not r.injury_status else [])
        ),axis=1)
        eligible=df[(df.odds_decimal>=a.min_odds)&~df.excluded_confirmed_absent].copy()
        eligible=eligible.sort_values(["model_probability","edge_probability"],ascending=False,na_position="last")
        eligible=eligible.drop_duplicates(["player_name"],keep="first").head(a.max_picks)
        df.to_csv(out/f"{market.lower()}_context_audit.csv",index=False)
        eligible.to_csv(out/f"{market.lower()}_context_picks.csv",index=False)
        report[market]={"candidates":len(df),"selected":len(eligible),
            "verified_context":int(df.context_complete.sum()),
            "excluded_confirmed_absent":int(df.excluded_confirmed_absent.sum()),
            "warning":"Probabilities unchanged: goalie/line/news effects need backtested calibration."}
    (out/"context_summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
