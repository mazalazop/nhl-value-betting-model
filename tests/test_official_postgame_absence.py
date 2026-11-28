"""Official NHL-confirmed postgame absences require match+player proof and final boxscore."""
import csv
import importlib.util
import json
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("postgame_pick_diagnostics_absence",ROOT/"scripts/postgame_pick_diagnostics.py")
diagnostics=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=diagnostics
spec.loader.exec_module(diagnostics)

DAY="2026-11-27"
GAME_ID=2026020411
PLAYER_ID=8474141

def fixture(tmp_path, include_player=False, evidence=True):
    picks=tmp_path/"picks.csv"
    with picks.open("w",newline="",encoding="utf-8") as file:
        w=csv.DictWriter(file,fieldnames=["date","market","player","team","id_joueur","id_match","archive_group"])
        w.writeheader();w.writerow(dict(date=DAY,market="POINT",player="Patrick Kane",team="CHI",
                                        id_joueur=str(PLAYER_ID),id_match=str(GAME_ID),archive_group="CANONICAL"))
    official=tmp_path/"official";official.mkdir()
    if evidence:
        (official/(DAY+".json")).write_text(json.dumps({
            "date":DAY,"scope":"postgame_only_not_pregame_eligibility",
            "records":[{"game_id":str(GAME_ID),"player_id":str(PLAYER_ID),"team":"CHI",
                "absence_confirmed":True,"proof_grade":"A",
                "official_match_recap_url":"https://www.nhl.com/news/blackhawks-recap",
                "official_injury_release_url":"https://www.nhl.com/blackhawks/news/official-injury",
                "reason":"Official NHL IR confirmation and did not play"}]}))
    score={"games":[dict(id=GAME_ID,gameDate=DAY,gameState="OFF",
                homeTeam={"abbrev":"CHI"},awayTeam={"abbrev":"NYR"})]}
    hockey=[]
    if include_player:
        hockey=[dict(playerId=PLAYER_ID,name={"default":"P. Kane"},
                     goals=0,assists=1,sog=0,toi="15:01")]
    box={"id":GAME_ID,"gameState":"OFF",
         "homeTeam":{"abbrev":"CHI"},"awayTeam":{"abbrev":"NYR"},
         "playerByGameStats":{"homeTeam":{"forwards":hockey,"defense":[]},
                              "awayTeam":{"forwards":[],"defense":[]}}}
    def fake(url):
        if "/score/" in url:return score
        if "/boxscore" in url:return box
        return {"plays":[]}
    return picks,official,fake,score

def read_result(out):
    with (out/(DAY+"-canonical-picks.csv")).open(newline="",encoding="utf-8") as file:
        return list(csv.DictReader(file))[0]

def test_official_confirmed_absence_is_not_a_losing_bet(tmp_path,monkeypatch):
    picks,official,fake,_=fixture(tmp_path)
    monkeypatch.setattr(diagnostics,"fetch",fake)
    out=tmp_path/"out"
    diagnostics.collect(DAY,picks,out,archive_id="canonical",absence_evidence_dir=official)
    row=read_result(out)
    assert row["status"]==row["settlement_status"]=="ABSENT_CONFIRMED"
    assert row["proof_grade"]=="A"
    assert "nhl.com" in row["absence_official_match_recap_url"]
    assert row["official_win"]=="" and row["points"]==""
    report=json.loads((out/(DAY+"-canonical-summary.json")).read_text())
    assert report["stats"]["CANONICAL"]["POINT"]["settled"]==0
    assert report["stats"]["CANONICAL"]["POINT"]["unresolved"]==1

def test_player_present_in_final_boxscore_supersedes_absence_annotation(tmp_path,monkeypatch):
    picks,official,fake,_=fixture(tmp_path,include_player=True)
    monkeypatch.setattr(diagnostics,"fetch",fake)
    out=tmp_path/"out"
    diagnostics.collect(DAY,picks,out,archive_id="canonical",absence_evidence_dir=official)
    row=read_result(out)
    assert row["status"]=="SETTLED" and row["official_win"]=="1"
    assert row.get("proof_grade","")=="" and row.get("absence_reason","")==""

def test_no_manual_evidence_cannot_guess_absence(tmp_path,monkeypatch):
    picks,official,fake,_=fixture(tmp_path,evidence=False)
    monkeypatch.setattr(diagnostics,"fetch",fake)
    out=tmp_path/"out"
    diagnostics.collect(DAY,picks,out,archive_id="canonical",absence_evidence_dir=official)
    row=read_result(out)
    assert row["status"]=="PLAYER_ID_NOT_IN_TEAM_BOXSCORE"
    assert row["official_win"]==""

def test_undocumented_player_id_cannot_be_substituted(tmp_path,monkeypatch):
    picks,official,fake,_=fixture(tmp_path)
    blob=json.loads((official/(DAY+".json")).read_text())
    blob["records"][0]["player_id"]="8479999"
    (official/(DAY+".json")).write_text(json.dumps(blob))
    monkeypatch.setattr(diagnostics,"fetch",fake)
    out=tmp_path/"out"
    diagnostics.collect(DAY,picks,out,archive_id="canonical",absence_evidence_dir=official)
    row=read_result(out)
    assert row["status"]=="PLAYER_ID_NOT_IN_TEAM_BOXSCORE"

def test_bad_unofficial_source_fails_closed(tmp_path,monkeypatch):
    picks,official,fake,_=fixture(tmp_path)
    blob=json.loads((official/(DAY+".json")).read_text())
    blob["records"][0]["official_match_recap_url"]="https://rumors.example.com/fake"
    (official/(DAY+".json")).write_text(json.dumps(blob))
    monkeypatch.setattr(diagnostics,"fetch",fake)
    with pytest.raises(ValueError,match="Unverified"):
        diagnostics.collect(DAY,picks,tmp_path/"out",archive_id="canonical",absence_evidence_dir=official)

def test_unfinished_game_is_never_an_absence_confirmation(tmp_path,monkeypatch):
    picks,official,fake,score=fixture(tmp_path)
    score["games"][0]["gameState"]="LIVE"
    monkeypatch.setattr(diagnostics,"fetch",fake)
    out=tmp_path/"out"
    diagnostics.collect(DAY,picks,out,archive_id="canonical",absence_evidence_dir=official)
    row=read_result(out)
    assert row["status"]=="GAME_NOT_FINAL" and row["official_win"]==""

def test_wrong_day_proof_fails_closed(tmp_path,monkeypatch):
    picks,official,fake,_=fixture(tmp_path)
    blob=json.loads((official/(DAY+".json")).read_text())
    blob["date"]="2026-11-26"
    (official/(DAY+".json")).write_text(json.dumps(blob))
    monkeypatch.setattr(diagnostics,"fetch",fake)
    with pytest.raises(ValueError,match="Invalid official absence evidence"):
        diagnostics.collect(DAY,picks,tmp_path/"out",archive_id="canonical",absence_evidence_dir=official)
