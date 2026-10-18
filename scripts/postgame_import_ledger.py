#!/usr/bin/env python3
"""Extract genuine pregame-priced selections from HENACHEL canonical ledger.

Different from a retrospective spreadsheet: only rows with a verified UTC
capture strictly BEFORE NHL puck drop may enter this longitudinal cohort.
No odds are fetched or reconstructed after the event.
"""
from __future__ import annotations
import argparse,csv,datetime,json,math,pathlib
from collections import defaultdict

UTC=datetime.timezone.utc
START=datetime.date(2026,10,17)
def parse_utc(value):
    if not value or str(value).strip().lower() in {"nan","none","nat"}:return None
    try:t=datetime.datetime.fromisoformat(str(value).replace("Z","+00:00"))
    except ValueError:return None
    return t.astimezone(UTC) if t.tzinfo is not None else None
def dateval(x):
    try:return datetime.date.fromisoformat(str(x)[:10])
    except (ValueError,TypeError):return None
def positive_float(x,minimum):
    try:r=float(x)
    except (ValueError,TypeError):return None
    return r if math.isfinite(r) and r>=minimum else None
def goal_market(row):
    market=str(row.get("market","")).strip().lower()
    stat=str(row.get("stat","")).strip().lower()
    threshold=positive_float(row.get("threshold"),0)
    if threshold!=1:return None
    if market in {"player_points","points"} and stat=="points":return "POINT"
    if market in {"player_goals","goals"} and stat=="goals":return "GOAL"
    return None
def extract(ledger):
    grouped=defaultdict(list);excluded=defaultdict(int);seen=set()
    for x in ledger:
        d=dateval(x.get("date_match"))
        if d is None or d<START:excluded["before_scope_or_invalid_date"]+=1;continue
        market=goal_market(x)
        if not market:excluded["unsupported_market"]+=1;continue
        if str(x.get("recommended_flag","")).strip().lower() not in {"true","1","1.0","yes"}:
            excluded["not_recommended"]+=1;continue
        ts=parse_utc(x.get("captured_at"));kickoff=parse_utc(x.get("start_time_utc"))
        if ts is None or kickoff is None or ts>=kickoff:
            excluded["not_provably_pregame"]+=1;continue
        if (kickoff-ts).total_seconds()>7*86400:
            excluded["stale_capture"]+=1;continue
        odds=positive_float(x.get("odds_decimal"),1.40)
        probability=positive_float(x.get("model_probability"),0)
        if odds is None or probability is None or probability>1:
            excluded["bad_price_or_probability"]+=1;continue
        if not all(str(x.get(k,"")).strip() for k in ("bet_id","id_match","id_joueur","player_name","team","bookmaker")):
            excluded["missing_identity"]+=1;continue
        bet_id=x["bet_id"]
        if bet_id in seen:excluded["duplicate_bet_id"]+=1;continue
        seen.add(bet_id)
        grouped[str(d)].append({
            "date":str(d),"market":market,"player":x["player_name"],
            "team":x["team"],"odds":str(odds),"signal":"CANONICAL_PREGAME",
            "archive_group":"CANONICAL","archive_provenance":"PREGAME_CANONICAL_LEDGER",
            "bet_id":bet_id,"bookmaker":x["bookmaker"],
            "id_match":x["id_match"],"id_joueur":x["id_joueur"],
            "model_probability":str(probability),
            "captured_at":ts.isoformat(),"start_time_utc":kickoff.isoformat(),
            "odds_verified_at_utc":ts.isoformat(),
        })
    return grouped,dict(excluded)
def export(input_csv,outdir):
    with pathlib.Path(input_csv).open(encoding="utf-8-sig",newline="") as f:
        records=list(csv.DictReader(f))
    groups,excluded=extract(records)
    outdir.mkdir(parents=True,exist_ok=True)
    for day,rows in groups.items():
        path=outdir/(day+"-canonical.csv")
        if path.exists():
            with path.open(encoding="utf-8-sig",newline="") as f:
                old=list(csv.DictReader(f))
            old_by_id={r["bet_id"]:r for r in old}
            new_by_id={r["bet_id"]:r for r in rows}
            for id_,existing in old_by_id.items():
                if id_ in new_by_id and existing!=new_by_id[id_]:
                    raise RuntimeError(f"Refusing alteration of frozen pregame bet {id_}: {path}")
            additions=[r for r in rows if r["bet_id"] not in old_by_id]
            if not additions:
                continue
            # A later legitimate intraday capture may append new bet IDs;
            # never overwrite or delete an existing snapshot.
            with path.open("a",encoding="utf-8",newline="") as f:
                csv.DictWriter(f,fieldnames=list(rows[0])).writerows(additions)
        else:
            with path.open("x",encoding="utf-8",newline="") as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={"input_rows":len(records),"dates":{date:len(x) for date,x in sorted(groups.items())},
            "exclusions":excluded,"archival_rule":"pregame UTC timestamp strictly earlier than puckdrop; original price frozen"}
    print("CANONICAL_IMPORT="+json.dumps(result,separators=(",",":")))
    return result
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--ledger",type=pathlib.Path,required=True)
    p.add_argument("--output-dir",type=pathlib.Path,default=pathlib.Path("docs/postgame/input"))
    a=p.parse_args();export(a.ledger,a.output_dir)
if __name__=="__main__":main()
