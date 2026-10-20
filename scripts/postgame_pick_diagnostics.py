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
def lookup_player(box,team,player,player_id=None):
    for side in ["home","away"]:
        meta=box.get(side+"Team") or {}
        abbr=str(meta.get("abbrev") or meta.get("triCode") or "").upper()
        if abbr!=team.upper():continue
        squad=box.get("playerByGameStats",{}).get(side+"Team") or {}
        rows=sum((squad.get(k) or [] for k in ["forwards","defense","defensemen"]),[])
        if player_id:
            identified=[x for x in rows if str(x.get("playerId"))==str(player_id)]
            if len(identified)==1:return identified[0],"NHL_ID"
            return None,"PLAYER_ID_NOT_IN_TEAM_BOXSCORE"
        if not str(player).strip():return None,"ABSENT_OR_NAME_MISMATCH"
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
        if (play.get("periodDescriptor") or {}).get("periodType")=="SO":continue
        typ=play.get("typeDescKey")
        if typ not in ["shot-on-goal","missed-shot","blocked-shot","goal"]:continue
        d=play.get("details") or {}
        shooter=d.get("shootingPlayerId") or d.get("scoringPlayerId")
        if str(shooter)!=str(pid):continue
        x=d.get("xCoord");y=d.get("yCoord")
        dist=round(math.hypot(89-abs(x),y),1) if isinstance(x,(int,float)) and isinstance(y,(int,float)) else None
        r.append({"type":typ,"dist_ft_proxy":dist,"period":(play.get("periodDescriptor") or {}).get("number")})
    return r
def collect(day,input_csv,outdir,archive_id=""):
    if archive_id and (not archive_id.replace("_","").isalnum() or len(archive_id)>35):
        raise ValueError("Invalid archive ID")
    slate=fetch(f"{API}/score/{day}")
    games=[g for g in slate.get("games",[]) if str(g.get("gameDate"))==day]
    if not games: raise ValueError("No official NHL slate available: "+day)
    team_to_game={}
    for g in games:
        for side in ["home","away"]:
            team=(g.get(side+"Team") or {}).get("abbrev")
            if team:team_to_game[str(team).upper()]=g
    with open(input_csv,encoding="utf-8-sig",newline="") as source:
        picks=list(csv.DictReader(source))
    identities=[(p.get("archive_group"),p.get("market"),p.get("bet_id") or
                 (p.get("id_match"),p.get("id_joueur") or norm(p.get("player")),p.get("team"))) for p in picks]
    if len(identities)!=len(set(identities)):raise ValueError("Duplicate pick identity")
    captured=datetime.datetime.now(datetime.timezone.utc).isoformat()
    cache={};out=[]
    for pick in picks:
        if pick["date"]!=day:continue
        row=dict(pick)
        row.setdefault("archive_provenance","RETROSPECTIVE_RECONSTRUCTION")
        row.update(postgame_captured_at_utc=captured,settlement_status="UNRESOLVED",status="UNRESOLVED",official_win="",goals="",assists="",points="",shots="",toi_min="",pp_toi_min="",
                   missed_attempts="",blocked_attempts="",attempts_pbp="",close_20ft_proxy="",close_40ft_proxy="",
                   xg_individual="NOT_AVAILABLE",disallowed_goal="NOT_VERIFIED",evidence="",source="")
        g=team_to_game.get(row["team"].upper())
        if not g:
            row["status"]="TEAM_NOT_IN_SLATE";out.append(row);continue
        gid=g["id"];row["source"]=f"https://www.nhl.com/gamecenter/{gid}"
        row["id_match_official"]=str(gid)
        # A player appearing in an unfinished boxscore has NOT lost their bet.
        if str(g.get("gameState") or "").upper() not in {"OFF","FINAL"}:
            row["status"]="GAME_NOT_FINAL";out.append(row);continue
        supplied_game_id=str(row.get("id_match") or "").strip()
        if supplied_game_id and supplied_game_id!=str(gid):
            row["status"]="GAME_ID_MISMATCH";out.append(row);continue
        if gid not in cache:
            box=fetch(f"{API}/gamecenter/{gid}/boxscore")
            try: pbp=fetch(f"{API}/gamecenter/{gid}/play-by-play")
            except Exception as exc:pbp={"plays":[],"source_error":str(exc)}
            cache[gid]=(box,pbp)
        box,pbp=cache[gid]
        player,method=lookup_player(box,row["team"],row["player"],str(row.get("id_joueur") or "").strip())
        if player is None:
            row["status"]=method;out.append(row);continue
        observed_toi=minute(player.get("toi") or player.get("timeOnIce"))
        if observed_toi is not None and observed_toi<=0:
            row["status"]="NO_ICE_TIME";out.append(row);continue
        if any(player.get(k) is None for k in ("goals","assists")):
            row["status"]="MISSING_OFFICIAL_OUTCOME";out.append(row);continue
        supplied_id=str(row.get("id_joueur") or "").strip()
        if supplied_id and supplied_id!=str(player.get("playerId")):
            row["status"]="PLAYER_ID_MISMATCH";out.append(row);continue
        row["matched_player_id"]=str(player.get("playerId"))
        row["matching_method"]=method
        row["status"]="SETTLED"
        row["matched_name"]=nm(player)
        row["observed_scorer_team"]=row["team"]
        # This is only an uncalibrated geographical proxy.
        # Blocked shots may be recorded at the blocker location.
        row["geometric_zone_is_xg"]=False
        goals=int(player.get("goals") or 0);assists=int(player.get("assists") or 0)
        points=int(player.get("points") if player.get("points") is not None else goals+assists)
        row["goals"]=goals;row["assists"]=assists;row["points"]=points
        row["official_win"]=int(goals>=1 if row["market"]=="GOAL" else points>=1)
        row["settlement_status"]="WIN" if row["official_win"] else "LOSS"
        shots=player.get("sog",player.get("shots"))
        row["shots"]=int(shots) if shots is not None else ""
        row["toi_min"]=minute(player.get("toi") or player.get("timeOnIce"))
        row["pp_toi_min"]=minute(player.get("powerPlayToi") or player.get("ppTimeOnIce") or player.get("powerPlayTimeOnIce"))
        row["boxscore_source"]=f"{API}/gamecenter/{gid}/boxscore"
        row["pbp_source"]=f"{API}/gamecenter/{gid}/play-by-play"
        if pbp.get("source_error") or not pbp.get("plays"):
            row["pbp_error"]=pbp.get("source_error") or "EMPTY_OR_MISSING_PLAYS"
            out.append(row);continue
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
    filename=day+(("-"+archive_id) if archive_id else "")+"-picks.csv"
    with (outdir/filename).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(out)
    overview={}
    for group in ["CLASSEUR","EXTRA","CANONICAL"]:
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
    summary["archive_id"]=archive_id or "manual"
    summary["archive_provenance"]="mixed—see row-level archive_provenance"
    (outdir/(day+(("-"+archive_id) if archive_id else "")+"-summary.json")).write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("AUDIT_SUMMARY="+json.dumps(summary,separators=(",",":")),flush=True)
    for x in out:
        print("AUDIT_ROW="+json.dumps({k:x.get(k) for k in ["market","player","team","archive_group","status","goals","assists","points","shots","toi_min","pp_toi_min","attempts_pbp","close_20ft_proxy","official_win"]},separators=(",",":")),flush=True)
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--date",required=True);p.add_argument("--picks",required=True);p.add_argument("--output-dir",required=True)
    p.add_argument("--archive-id",default="",help="Append source ID (e.g. canonical), avoiding collisions")
    a=p.parse_args();datetime.date.fromisoformat(a.date)
    collect(a.date,a.picks,pathlib.Path(a.output_dir),a.archive_id)
if __name__=="__main__":main()
