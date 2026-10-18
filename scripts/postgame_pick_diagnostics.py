#!/usr/bin/env python3
"""Postgame diagnostic of archived HENACHEL picks. Research only, not a model update."""
import argparse,csv,datetime,json,math,pathlib,re,time,unicodedata
from collections import Counter
from urllib.request import Request,urlopen

API="https://api-web.nhle.com/v1"
def fetch(url):
    error=None
    for attempt in range(3):
        try:
            with urlopen(Request(url,headers={"Accept":"application/json","User-Agent":"Mozilla/5.0 HENACHEL-Research"}),timeout=35) as f:
                return json.load(f)
        except Exception as exc:
            error=exc
            if attempt<2: time.sleep(2+attempt*2)
    raise RuntimeError(f"NHL API error: {url}: {error}")
def norm(s):
    s=unicodedata.normalize("NFKD",str(s or "")).casefold()
    return re.sub("[^a-z0-9]","", "".join(c for c in s if not unicodedata.combining(c)))
def nm(x):
    v=x.get("name",{})
    if isinstance(v,dict): v=v.get("default","")
    if v:return str(v)
    return " ".join((x.get(k,{}).get("default","") if isinstance(x.get(k),dict) else str(x.get(k) or "")) for k in ["firstName","lastName"]).strip()
def minute(x):
    if x in [None,""]:return None
    if ":" in str(x):
        a,b=str(x).split(":",1);return round(int(a)+int(b)/60,3)
    try:return round(float(x),3)
    except: return None
def lookup_player(box,team,player):
    for side in ["home","away"]:
        meta=box.get(side+"Team") or {}
        abbr=str(meta.get("abbrev") or meta.get("triCode") or "").upper()
        if abbr!=team.upper():continue
        squad=box.get("playerByGameStats",{}).get(side+"Team") or {}
        rows=sum((squad.get(k) or [] for k in ["forwards","defense","defensemen"]),[])
        exact=[x for x in rows if norm(nm(x))==norm(player)]
        if len(exact)==1:return exact[0],"EXACT"
        # NHL boxscores abbreviate first names. Disambiguate siblings and
        # same-surname teammates with first initial + complete surname.
        first=norm(player.split()[0])[:1]
        surname=norm(player.split()[-1])
        initial_matches=[x for x in rows if norm(nm(x))==first+surname]
        if len(initial_matches)==1:return initial_matches[0],"INITIAL_SURNAME"
        # Team-unique surname is allowed only when unambiguous.
        loose=[x for x in rows if norm(nm(x)).endswith(surname)]
        if len(loose)==1:return loose[0],"TEAM_SURNAME"
        return None,"ABSENT_OR_NAME_MISMATCH"
    return None,"TEAM_NOT_IN_BOXSCORE"
def extract_attempts(pbp,pid):
    r=[]
    for play in pbp.get("plays",[]):
        typ=play.get("typeDescKey")
        if typ not in ["shot-on-goal","missed-shot","blocked-shot","goal"]:continue
        d=play.get("details") or {}
        shooter=d.get("shootingPlayerId") or d.get("scoringPlayerId")
        if str(shooter)!=str(pid):continue
        x=d.get("xCoord");y=d.get("yCoord")
        dist=round(math.hypot(89-abs(x),y),1) if isinstance(x,(int,float)) and isinstance(y,(int,float)) else None
        r.append({"type":typ,"dist_ft_proxy":dist,"period":(play.get("periodDescriptor") or {}).get("number")})
    return r
def collect(day,input_csv,outdir):
    slate=fetch(f"{API}/score/{day}")
    games=[g for g in slate.get("games",[]) if str(g.get("gameDate"))==day]
    if not games: raise ValueError("No official NHL slate available: "+day)
    team_to_game={}
    for g in games:
        for side in ["home","away"]:
            team=(g.get(side+"Team") or {}).get("abbrev")
            if team:team_to_game[str(team).upper()]=g
    picks=list(csv.DictReader(open(input_csv,encoding="utf-8-sig",newline="")))
    cache={};out=[]
    for pick in picks:
        if pick["date"]!=day:continue
        row=dict(pick)
        row.update(status="UNRESOLVED",official_win="",goals="",assists="",points="",shots="",toi_min="",pp_toi_min="",
                   missed_attempts="",blocked_attempts="",attempts_pbp="",close_20ft_proxy="",close_40ft_proxy="",
                   xg_individual="NOT_AVAILABLE",disallowed_goal="NOT_VERIFIED",evidence="",source="")
        g=team_to_game.get(row["team"].upper())
        if not g:
            row["status"]="TEAM_NOT_IN_SLATE";out.append(row);continue
        gid=g["id"];row["source"]=f"https://www.nhl.com/gamecenter/{gid}"
        if gid not in cache:
            box=fetch(f"{API}/gamecenter/{gid}/boxscore")
            try: pbp=fetch(f"{API}/gamecenter/{gid}/play-by-play")
            except Exception as exc:pbp={"plays":[],"source_error":str(exc)}
            cache[gid]=(box,pbp)
        box,pbp=cache[gid]
        player,method=lookup_player(box,row["team"],row["player"])
        if player is None:
            row["status"]=method;out.append(row);continue
        row["status"]="SETTLED"
        row["matched_name"]=nm(player)
        goals=int(player.get("goals") or 0);assists=int(player.get("assists") or 0)
        points=int(player.get("points") if player.get("points") is not None else goals+assists)
        row["goals"]=goals;row["assists"]=assists;row["points"]=points
        row["official_win"]=int(goals>=1 if row["market"]=="GOAL" else points>=1)
        row["shots"]=int(player.get("sog") if player.get("sog") is not None else (player.get("shots") or 0))
        row["toi_min"]=minute(player.get("toi") or player.get("timeOnIce"))
        row["pp_toi_min"]=minute(player.get("powerPlayToi") or player.get("ppTimeOnIce") or player.get("powerPlayTimeOnIce"))
        ev=extract_attempts(pbp,player.get("playerId"))
        row["attempts_pbp"]=len(ev)
        row["missed_attempts"]=sum(e["type"]=="missed-shot" for e in ev)
        row["blocked_attempts"]=sum(e["type"]=="blocked-shot" for e in ev)
        row["close_20ft_proxy"]=sum(e["dist_ft_proxy"] is not None and e["dist_ft_proxy"]<=20 for e in ev)
        row["close_40ft_proxy"]=sum(e["dist_ft_proxy"] is not None and e["dist_ft_proxy"]<=40 for e in ev)
        row["evidence"]=json.dumps(ev,separators=(",",":"))
        out.append(row)
    if not out:raise ValueError("No archived picks for target day")
    outdir.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in out for k in row))
    with (outdir/(day+"-picks.csv")).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(out)
    overview={}
    for group in ["CLASSEUR","EXTRA"]:
        overview[group]={}
        for market in ["POINT","GOAL"]:
            subset=[x for x in out if x["archive_group"]==group and x["market"]==market]
            resolved=[x for x in subset if x["status"]=="SETTLED"]
            overview[group][market]={"original":len(subset),"settled":len(resolved),"wins":sum(x["official_win"] for x in resolved),"unresolved":len(subset)-len(resolved)}
    summary={"date":day,"source":"Official NHL API boxscore and play-by-play","teams":len(team_to_game),
             "picks":len(out),"stats":overview,"limitations":["No official individual xG provided",
             "Distance-based zones are uncalibrated proxies, not xG or dangerous shots",
             "Disallowed goals require separately corroborated official video/event",
             "Do not retrain or modify HENACHEL probabilities on a single slate"]}
    (outdir/(day+"-summary.json")).write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("AUDIT_SUMMARY="+json.dumps(summary,separators=(",",":")),flush=True)
    for x in out:
        print("AUDIT_ROW="+json.dumps({k:x.get(k) for k in ["market","player","team","archive_group","status","goals","assists","points","shots","toi_min","pp_toi_min","attempts_pbp","close_20ft_proxy","official_win"]},separators=(",",":")),flush=True)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--date",required=True);p.add_argument("--picks",required=True);p.add_argument("--output-dir",required=True)
    a=p.parse_args();datetime.date.fromisoformat(a.date)
    collect(a.date,a.picks,pathlib.Path(a.output_dir))
if __name__=="__main__":main()
