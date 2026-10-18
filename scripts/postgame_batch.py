#!/usr/bin/env python3
"""Daily idempotent research batch for all available HENACHEL pick snapshots."""
from __future__ import annotations
import argparse,datetime,pathlib,re
from postgame_pick_diagnostics import collect
from postgame_longitudinal import build
START=datetime.date(2026,10,17)
PAT=re.compile(r"^(20\d\d-\d\d-\d\d)(?:-(canonical))?\.csv$")
def run(inp,out,summary,cutoff,max_age_days=45):
    files=sorted(inp.glob("*.csv"))
    selected=[];skipped=[]
    for path in files:
        m=PAT.fullmatch(path.name)
        if not m:
            skipped.append(path.name+": invalid filename");continue
        d=datetime.date.fromisoformat(m.group(1))
        if d<START or d>cutoff:
            skipped.append(path.name+": out of date");continue
        result=out/(str(d)+("-canonical" if m.group(2) else "")+"-picks.csv")
        if result.exists() and (cutoff-d).days>max_age_days:
            skipped.append(path.name+": frozen older audit");continue
        selected.append((d,path,m.group(2) or ""))
    if not selected and not list(out.glob("????-??-??*-picks.csv")):
        print("No source snapshots/results yet: no claims, no empty artifact");return None
    for d,path,source in selected:
        collect(str(d),path,out,source)
    result=build(out,summary)
    print(f"DAILY_AUDIT: processed={len(selected)} archived dates={len(result['archive_dates'])}")
    return result
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input-dir",type=pathlib.Path,default=pathlib.Path("docs/postgame/input"))
    p.add_argument("--results-dir",type=pathlib.Path,default=pathlib.Path("docs/postgame/results"))
    p.add_argument("--longitudinal-dir",type=pathlib.Path,default=pathlib.Path("docs/postgame/longitudinal"))
    p.add_argument("--as-of",type=datetime.date.fromisoformat,default=datetime.datetime.now(datetime.timezone.utc).date()-datetime.timedelta(days=1))
    a=p.parse_args();run(a.input_dir,a.results_dir,a.longitudinal_dir,a.as_of)
if __name__=="__main__":main()
