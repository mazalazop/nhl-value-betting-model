"""Postgame quality controls never infer goals from chances or future labels."""
import csv,importlib.util,json,sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
def load(name):
    path=ROOT/"scripts"/(name+".py")
    spec=importlib.util.spec_from_file_location("test_mod_"+name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module
diagnostics=load("postgame_pick_diagnostics")
ledger=load("postgame_import_ledger")
longitudinal=load("postgame_longitudinal")

def csvfile(path,rows):
    with path.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)

def test_official_point_and_goal_no_fictional_challenge(tmp_path,monkeypatch):
    inp=tmp_path/"picks.csv"
    common=dict(date="2026-10-17",player="Tim Stutzle",team="OTT",
                archive_group="EXTRA",archive_provenance="RETROSPECTIVE_RECONSTRUCTION",
                odds="2.80",signal="DISCUSSION")
    csvfile(inp,[dict(common,market="GOAL"),dict(common,market="POINT")])
    box={"homeTeam":{"abbrev":"PHI"},"awayTeam":{"abbrev":"OTT"},
         "playerByGameStats":{"homeTeam":{"forwards":[],"defense":[]},"awayTeam":{
          "forwards":[{"name":{"default":"T. Stützle"},"playerId":123,
                       "goals":0,"assists":1,"points":1,"sog":1,"toi":"23:03"}],"defense":[]}}}
    pbp={"plays":[{"typeDescKey":"shot-on-goal",
                  "details":{"shootingPlayerId":123,"xCoord":80,"yCoord":5},
                  "periodDescriptor":{"number":2}}]}
    def fake(url):
        if url.endswith("/score/2026-10-17"):
            return {"games":[{"id":2026020001,"gameDate":"2026-10-17","gameState":"OFF",
                               "homeTeam":{"abbrev":"PHI"},"awayTeam":{"abbrev":"OTT"}}]}
        if url.endswith("/boxscore"):return box
        if url.endswith("/play-by-play"):return pbp
        raise AssertionError(url)
    monkeypatch.setattr(diagnostics,"fetch",fake)
    diagnostics.collect("2026-10-17",inp,tmp_path/"out")
    with (tmp_path/"out/2026-10-17-picks.csv").open() as f:
        saved=list(csv.DictReader(f))
    assert [x["status"] for x in saved]==["SETTLED","SETTLED"]
    assert [x["official_win"] for x in saved]==["0","1"]
    assert saved[0]["shots"]=="1"
    assert saved[0]["attempts_pbp"]=="1"
    assert saved[0]["disallowed_goal"]=="NOT_VERIFIED"
    assert saved[0]["xg_individual"]=="NOT_AVAILABLE"
    assert saved[0]["archive_provenance"]=="RETROSPECTIVE_RECONSTRUCTION"
    s=json.loads((tmp_path/"out/2026-10-17-summary.json").read_text())
    assert s["stats"]["EXTRA"]["POINT"]["wins"]==1

def test_do_not_settle_an_unfinished_game(tmp_path,monkeypatch):
    path=tmp_path/"picks.csv"
    csvfile(path,[dict(date="2026-10-17",market="POINT",player="P",team="OTT",
                       archive_group="CLASSEUR",signal="S",odds="1.5")])
    monkeypatch.setattr(diagnostics,"fetch",
      lambda url: {"games":[{"id":2,"gameDate":"2026-10-17","gameState":"LIVE",
                      "homeTeam":{"abbrev":"PHI"},"awayTeam":{"abbrev":"OTT"}}]}
      if url.endswith("/score/2026-10-17") else (_ for _ in ()).throw(AssertionError("premature boxscore")))
    diagnostics.collect("2026-10-17",path,tmp_path/"out")
    with (tmp_path/"out/2026-10-17-picks.csv").open() as f:
        r=list(csv.DictReader(f))[0]
    assert r["status"]=="GAME_NOT_FINAL"
    assert r["official_win"]==""

def test_canonical_import_validates_pregame_identity_and_prices():
    base=dict(date_match="2026-10-17",market="player_points",stat="points",threshold="1",
              recommended_flag="1",captured_at="2026-10-17T15:00:00Z",
              start_time_utc="2026-10-17T19:00:00Z",odds_decimal="1.45",
              model_probability=".7",bet_id="unique",id_match="2026020011",
              id_joueur="100",player_name="Example Player",team="NJD",bookmaker="Unibet")
    out,excluded=ledger.extract([base])
    assert len(out["2026-10-17"])==1
    assert out["2026-10-17"][0]["archive_provenance"]=="PREGAME_CANONICAL_LEDGER"
    assert out["2026-10-17"][0]["model_probability"]=="0.7"
    for reason,bad in [("not_provably_pregame",dict(base,captured_at="2026-10-17T19:05:00Z")),
                       ("bad_price_or_probability",dict(base,odds_decimal="1.35")),
                       ("not_recommended",dict(base,recommended_flag="0"))]:
        out,why=ledger.extract([bad])
        assert not out
        assert why[reason]==1

def test_longitudinal_no_retrospective_roi_or_model_score(tmp_path):
    d=tmp_path/"results";d.mkdir()
    rows=[dict(date="2026-10-17",archive_group="CLASSEUR",market="GOAL",
               player="Player A",team="OTT",signal="REBONd",status="SETTLED",official_win="0",
               shots="5",attempts_pbp="8",odds="2.8",model_probability=".8",
               archive_provenance="RETROSPECTIVE_RECONSTRUCTION"),
          dict(date="2026-10-17",archive_group="EXTRA",market="POINT",
               player="Player B",team="OTT",signal="DISCUSSION",status="SETTLED",official_win="1",
               shots="1",attempts_pbp="2",archive_provenance="RETROSPECTIVE_RECONSTRUCTION")]
    csvfile(d/"2026-10-17-picks.csv",rows)
    report=longitudinal.build(d,tmp_path/"summary")
    one=report["windows"]["last_date"]["CLASSEUR"]["GOAL"]
    assert one["wins"]==0 and one["failed_with_activity"]==1
    assert one["brier"] is None and one["unit_stake_simulated_roi"] is None
    assert report["windows"]["last_date"]["EXTRA"]["POINT"]["wins"]==1

def test_longitudinal_canonical_evaluation_and_duplicate_guard(tmp_path):
    d=tmp_path/"results";d.mkdir()
    rows=[dict(date="2026-10-17",archive_group="CANONICAL",market="POINT",bet_id="a",
               player="A",team="OTT",signal="CANONICAL_PREGAME",status="SETTLED",official_win="1",
               shots="2",attempts_pbp="4",odds="1.5",captured_at="2026-10-17T14:00Z",
               start_time_utc="2026-10-17T19:00Z",model_probability=".6",
               archive_provenance="PREGAME_CANONICAL_LEDGER")]
    csvfile(d/"2026-10-17-canonical-picks.csv",rows)
    summary=longitudinal.build(d,tmp_path/"summary")
    t=summary["windows"]["since_2026_10_17"]["CANONICAL"]["POINT"]
    assert t["brier"]==pytest.approx(.16)
    assert t["unit_stake_simulated_roi"]==pytest.approx(.5)
    csvfile(d/"2026-10-18-canonical-picks.csv",[dict(rows[0],date="2026-10-18")])
    # Re-use of bet_id is only possible on a DIFFERENT date; independent identities.
    assert len(longitudinal.load(d.glob("*-picks.csv")))==2
    csvfile(d/"2026-10-17-duplicate-picks.csv",rows)
    with pytest.raises(ValueError,match="Duplicate"):
        longitudinal.load(d.glob("*-picks.csv"))

@pytest.mark.parametrize("group,cap",[("EXTRA","2026-10-17T14:00Z"),("CANONICAL","2026-10-17T20:00Z"),("CANONICAL","2026-10-17T14:00")])
def test_provenance_string_alone_does_not_authorize_metrics(group,cap):
    row=dict(archive_group=group,archive_provenance="PREGAME_CANONICAL_LEDGER",
             status="SETTLED",official_win="1",model_probability=".6",odds="2",
             captured_at=cap,start_time_utc="2026-10-17T19:00Z")
    stats=longitudinal.metrics([row])
    assert stats['calibrated_n']==stats['priced_pregame_n']==0

@pytest.mark.parametrize("toi",[None,"00:00"])
def test_unavailable_pbp_and_shots_remain_unknown(tmp_path,monkeypatch,toi):
    path=tmp_path/'input.csv'
    csvfile(path,[dict(date='2026-10-17',market='GOAL',player='Name',team='OTT',
                       id_joueur='42',archive_group='CANONICAL')])
    def fake(url):
        if '/score/' in url:return {'games':[dict(id=1,gameDate='2026-10-17',gameState='OFF',homeTeam={'abbrev':'OTT'})]}
        if url.endswith('boxscore'):return {'homeTeam':{'abbrev':'OTT'},'playerByGameStats':{'homeTeam':{'forwards':[
            dict(playerId=42,name={'default':'Other spelling'},goals=0,assists=1,toi=toi)]}}}
        raise RuntimeError('unavailable')
    monkeypatch.setattr(diagnostics,'fetch',fake)
    diagnostics.collect('2026-10-17',path,tmp_path/'out')
    with (tmp_path/'out/2026-10-17-picks.csv').open() as f:
        row=list(csv.DictReader(f))[0]
    if toi=='00:00':
        assert row['settlement_status']=='UNRESOLVED' and row['official_win']==''
        return
    assert row['settlement_status']=='LOSS'
    assert row['shots']==row['attempts_pbp']==row['missed_attempts']==''
    assert row['matching_method']=='NHL_ID' and row['postgame_captured_at_utc']

def test_conflicting_duplicate_bet_cannot_be_silently_dropped():
    x=dict(date_match='2026-10-17',market='player_points',stat='points',threshold='1',
        recommended_flag='1',captured_at='2026-10-17T12:00Z',start_time_utc='2026-10-17T20:00Z',
        odds_decimal='1.5',model_probability='.5',bet_id='x',id_match='1',id_joueur='2',
        player_name='A',team='OTT',bookmaker='B')
    with pytest.raises(ValueError,match='Conflicting'):ledger.extract([x,dict(x,model_probability='.6')])
    assert ledger.extract([x,x])[1]['duplicate_bet_id']==1

def test_collector_preserves_previous_results_and_partial_pbp(tmp_path,monkeypatch):
    path=tmp_path/'input.csv';out=tmp_path/'out';out.mkdir()
    csvfile(path,[dict(date='2026-10-17',market='GOAL',player='A',team='OTT',id_joueur='42',archive_group='CANONICAL')])
    original=b'previous archive bytes\n';(out/'2026-10-17-picks.csv').write_bytes(original)
    box={'id':1,'gameState':'OFF','homeTeam':{'abbrev':'OTT'},'playerByGameStats':{'homeTeam':{'forwards':[
        dict(playerId=42,name={'default':'A'},goals=0,assists=0,sog=3,toi='15:00')]}}}
    def fake(url):
        if '/score/' in url:return {'games':[dict(id=1,gameDate='2026-10-17',gameState='OFF',homeTeam={'abbrev':'OTT'})]}
        if url.endswith('boxscore'):return box
        return {'plays':[{'typeDescKey':'shot-on-goal','details':{'shootingPlayerId':42}}]}
    monkeypatch.setattr(diagnostics,'fetch',fake)
    diagnostics.collect('2026-10-17',path,out)
    assert any(p.read_bytes()==original for p in (out/'revisions').glob('*-picks.csv'))
    with (out/'2026-10-17-picks.csv').open() as f:row=list(csv.DictReader(f))[0]
    assert row['official_win']=='0' and row['attempts_pbp']=='' and row['pbp_error']=='SHOT_COUNT_MISMATCH'
    box['playerByGameStats']['homeTeam']['forwards'][0]['sog']=1
    diagnostics.collect('2026-10-17',path,out)
    with (out/'2026-10-17-picks.csv').open() as f:row=list(csv.DictReader(f))[0]
    assert row['attempts_pbp']=='1' and row['close_20ft_proxy']==''
