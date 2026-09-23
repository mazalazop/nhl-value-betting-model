#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs/api/daily_picks.json"
DEFAULT_REPORT = ROOT / "outputs/api/validation.json"

def validate(p: dict) -> list[str]:
    errors = []
    required = ["schema_version","model_version","generated_at_utc","timezone","slate_date","picks"]
    errors += [f"missing:{k}" for k in required if k not in p]
    if errors: return errors
    try: datetime.fromisoformat(p["generated_at_utc"].replace("Z","+00:00"))
    except Exception: errors.append("generated_at_utc is not ISO-8601")
    if p["timezone"] != "Europe/Paris": errors.append("timezone must be Europe/Paris")
    if not isinstance(p["picks"], list): return errors + ["picks must be a list"]
    for i,x in enumerate(p["picks"]):
        for k in ["rank","player","team","opponent","match_date","market","odds","model_probability","edge"]:
            if k not in x: errors.append(f"pick[{i}] missing:{k}")
        if "odds" in x and (not isinstance(x["odds"],(int,float)) or x["odds"] <= 1): errors.append(f"pick[{i}] invalid odds")
        if "model_probability" in x and not (0 <= x["model_probability"] <= 1): errors.append(f"pick[{i}] invalid probability")
    if p.get("pick_count") != len(p["picks"]): errors.append("pick_count mismatch")
    return errors

def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument("--input-json",default=str(DEFAULT_INPUT))
    p.add_argument("--report-json",default=str(DEFAULT_REPORT))
    a=p.parse_args()
    payload=json.loads(Path(a.input_json).read_text(encoding="utf-8"))
    errors=validate(payload)
    report={"status":"ok" if not errors else "error","errors":errors}
    Path(a.report_json).parent.mkdir(parents=True,exist_ok=True)
    Path(a.report_json).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    if errors: raise SystemExit("Payload invalide: "+"; ".join(errors))
    print("Payload API valide.")

if __name__=="__main__": main()
