#!/usr/bin/env python3
"""17 October 2026 official NHL score/boxscore supplement for PP1 season matrix.
Research export, independent of production predictions.
"""
import csv,json,time,urllib.request,pathlib
from datetime import date
URL="https://api-web.nhle.com/v1"
DAY="2026-10-17"
def get(path):
    err=None
    for i in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(URL+path,headers={"User-Agent":"HENACHEL-Matrix/1.0","Accept":"application/json"}),timeout=30) as f:
                return json.load(f)
        except Exception as exc:
            err=exc;time.sleep(i+1)
    raise RuntimeError(str(err))
def main():
    slate=get("/score/"+DAY)
    matches=[x for x in slate.get("games",[]) if x.get("gameDate")==DAY]
    if len(matches)!=14:raise RuntimeError("Expected 14 official NHL games 17 Oct, got "+str(len(matches)))
    rows=[]
    for g in matches:
        if str(g.get("gameState")).upper() not in ("OFF","FINAL"):raise RuntimeError("Game not final "+str(g["id"]))
        b=get(f"/gamecenter/{g['id']}/boxscore")
        players=b.get("playerByGameStats") or {}
        for side in ["home","away"]:
            team=(b.get(side+"Team") or g[side+"Team"]).get("abbrev")
            squad=players.get(side+"Team",{})
            for item in (squad.get("forwards") or [])+(squad.get("defense") or []):
                v=item.get("name",{})
                name=v.get("default","") if isinstance(v,dict) else str(v)
                goals=item.get("goals");assists=item.get("assists")
                if goals is None or assists is None:raise RuntimeError("Missing score in boxscore for "+name)
                rows.append({"date":DAY,"game_id":g["id"],"player_id":item["playerId"],
                             "team":team,"display_name":name,"goals":goals,"assists":assists,
                             "points":item.get("points",int(goals)+int(assists)),
                             "source_url":f"https://www.nhl.com/gamecenter/{g['id']}"})
    if len(rows)<400:raise RuntimeError("Incomplete skater export "+str(len(rows)))
    output=pathlib.Path("docs/pp1_history/source/2026-10-17-official.csv")
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print("EXPORTED_ROWS",len(rows),"GAMES",len(matches),"TEAMS",len(set(r["team"] for r in rows)))
if __name__=="__main__":main()
