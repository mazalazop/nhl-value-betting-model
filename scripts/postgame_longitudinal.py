#!/usr/bin/env python3
"""Longitudinal HENACHEL review: observed results vs opportunity quality.

Reads only immutable-style archived daily official-NHL audits.  Observations
are descriptive and NEVER change model weights, probabilities or bet settlement.
"""
from __future__ import annotations
import argparse,csv,datetime,json,math,pathlib,statistics
from collections import defaultdict

START=datetime.date(2026,10,17)
COLS=["date","archive_group","market","signal","number","settled","wins",
      "unresolved","hit_rate","mean_shots","mean_attempts","failed_with_activity",
      "failed_with_low_activity","missing_shot_data","sample_warning"]
def number(x):
    try:
        s=str(x or "").strip()
        n=float(s)
        return n if math.isfinite(n) else None
    except(ValueError,TypeError):return None
def load(paths):
    result=[];seen=set()
    for path in sorted(paths):
        date=path.name.split("-picks.csv")[0]
        d=datetime.date.fromisoformat(date)
        if d<START: continue
        with path.open(encoding="utf-8-sig",newline="") as f:
            rows=list(csv.DictReader(f))
        for r in rows:
            if r.get("date")!=date:raise ValueError(f"Mixed dates in {path}")
            required=["market","player","team","archive_group","status","official_win"]
            if any(k not in r for k in required):raise ValueError(f"Malformed archive {path}")
            market=str(r["market"]).strip().upper()
            if market not in {"POINT","GOAL"}:raise ValueError(f"Unknown market {market}")
            group=r["archive_group"].strip().upper()
            key=(date,group,market,r["player"].casefold().strip(),r["team"].upper().strip())
            if key in seen:raise ValueError(f"Duplicate prediction identity {key}")
            seen.add(key)
            if r["status"]=="SETTLED" and r["official_win"] not in {"0","1"}:
                raise ValueError(f"Settled without official outcome {key}")
            result.append(r)
    return result
def wilson(w,n,z=1.96):
    if not n:return None
    p=w/n;den=1+z*z/n
    delta=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    center=(p+z*z/(2*n))/den
    return {"low":round(max(0,center-delta),4),"high":round(min(1,center+delta),4)}
def metrics(rows):
    settled=[x for x in rows if x["status"]=="SETTLED"]
    wins=sum(int(x["official_win"]) for x in settled)
    misses=[x for x in settled if x["official_win"]=="0"]
    shots=[number(x.get("shots")) for x in settled]
    attempts=[number(x.get("attempts_pbp")) for x in settled]
    with_activity=sum((number(x.get("shots")) or 0)>=3 or
                      (number(x.get("attempts_pbp")) or 0)>=5 for x in misses
                      if number(x.get("shots")) is not None or number(x.get("attempts_pbp")) is not None)
    low_activity=sum((number(x.get("shots")) or 0)<3 and
                      (number(x.get("attempts_pbp")) or 0)<5 for x in misses
                      if number(x.get("shots")) is not None and number(x.get("attempts_pbp")) is not None)
    vshots=[x for x in shots if x is not None]
    vats=[x for x in attempts if x is not None]
    return dict(number=len(rows),settled=len(settled),wins=wins,
                unresolved=len(rows)-len(settled),
                hit_rate=round(wins/len(settled),4) if settled else None,
                hit_rate_wilson95=wilson(wins,len(settled)),
                mean_shots=round(statistics.mean(vshots),3) if vshots else None,
                mean_attempts=round(statistics.mean(vats),3) if vats else None,
                failed_with_activity=with_activity,
                failed_with_low_activity=low_activity,
                missing_shot_data=len(settled)-len(vshots),
                sample_warning="EXPLORATORY_LOW_N" if len(settled)<100 else "")
def grouping(rows,parts):
    groups=defaultdict(list)
    for r in rows:groups[tuple(r.get(k,"") for k in parts)].append(r)
    out=[]
    for key,items in sorted(groups.items()):
        v=metrics(items);out.append({**dict(zip(parts,key)),**v})
    return out
def write_csv(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for x in data for k in x))
    if not keys:return
    with path.open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in data:
            flat={k:(json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v) for k,v in r.items()}
            w.writerow(flat)
def report(rows):
    dates=sorted({r["date"] for r in rows})
    if not dates:raise ValueError("No archived postgame audits")
    current=datetime.date.fromisoformat(dates[-1]);week_start=current-datetime.timedelta(days=6)
    datasets={
       "last_date":[r for r in rows if r["date"]==dates[-1]],
       "last_7_calendar_days":[r for r in rows if r["date"]>=week_start.isoformat()],
       "since_2026_10_17":rows,
    }
    summary={"through_date":dates[-1],"archive_dates":dates,"coverage_days":len(dates),
             "notes":["Group CLASSEUR and EXTRA never combined for a headline win rate.",
                      "Missing or absent skaters never classified as losers.",
                      "Attempt counts and approximate <=20/40ft distances are NOT xG.",
                      "No ROI calculated without independently verifiable pregame odds and stakes.",
                      "This is postgame descriptive learning, not an out-of-sample model change.",
                      "Game-day selections transcribed after play cannot substantiate pregame provenance."],
             "windows":{}}
    for period,data in datasets.items():
        summary["windows"][period]={g:{market:metrics([r for r in data if r["archive_group"]==g and r["market"]==market])
                                     for market in ["POINT","GOAL"]} for g in ["CLASSEUR","EXTRA"]}
    return summary,grouping(rows,["date","archive_group","market"]),grouping(rows,["archive_group","market","signal"])
def build(resultdir,outdir):
    rows=load(resultdir.glob("????-??-??-picks.csv"))
    summary,daily,signals=report(rows)
    outdir.mkdir(parents=True,exist_ok=True)
    write_csv(outdir/"daily_metrics.csv",daily)
    write_csv(outdir/"signal_metrics.csv",signals)
    (outdir/"cumulative_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    header=["# HENACHEL – apprentissage longitudinal post-match","",
            f"Dates couvertes : {len(summary['archive_dates'])} (jusqu'au {summary['through_date']}).",
            "","Les observations avec moins de 100 matchs sont exploratoires. Il n'y a pas de coefficient modifié.",
            "","## Dernière journée"]
    for m,s in summary["windows"]["last_date"]["CLASSEUR"].items():
        header.append(f"- {m}: {s['wins']}/{s['settled']} (non résolus : {s['unresolved']}); tentatives moyennes : {s['mean_attempts']}")
    header+=["","## Points à préserver","",
             "- Ne pas confondre échec du pari et absence d'occasions de qualité.",
             "- Un joueur absent n'est pas un pick officiellement perdant.",
             "- Une tentative proche du filet n'est pas une estimation xG.",
             "- Les cotes rapportées a posteriori ne suffisent pas à un calcul crédible du ROI.",
             "- Aucun apprentissage automatique sur les mêmes matchs que ceux ayant servi à définir les critères.",
             "","Reproductibilité : daily_metrics.csv, signal_metrics.csv et cumulative_summary.json."]
    (outdir/"README.md").write_text("\n".join(header)+"\n",encoding="utf-8")
    return summary
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--results-dir",type=pathlib.Path,default=pathlib.Path("docs/postgame/results"))
    parser.add_argument("--output-dir",type=pathlib.Path,default=pathlib.Path("docs/postgame/longitudinal"))
    a=parser.parse_args();s=build(a.results_dir,a.output_dir)
    print("LONGITUDINAL_SUMMARY="+json.dumps(s["windows"],separators=(",",":"),ensure_ascii=False))
if __name__=="__main__":main()
