"""Guard the historical 03-16 Oct backfill without adding any new workflow."""
import datetime
import pathlib
import sys

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]/"scripts"))
import postgame_import_ledger as ledger
import postgame_batch as batch
import postgame_longitudinal as longitudinal


def test_early_october_import_stays_pregame_and_priced():
    base=dict(date_match="2026-10-03",market="player_points",stat="points",threshold="1",
              recommended_flag="1",captured_at="2026-10-03T10:00:00Z",
              start_time_utc="2026-10-03T20:00:00Z",odds_decimal="1.45",
              model_probability=".60",bet_id="oct03",id_match="2026020001",
              id_joueur="123",player_name="Test Skater",team="NJD",bookmaker="Unibet")
    accepted, excluded = ledger.extract([base,dict(base,date_match="2026-10-02",bet_id="oct02")])
    assert list(accepted) == ["2026-10-03"]
    assert len(accepted["2026-10-03"]) == 1
    assert excluded["before_scope_or_invalid_date"] == 1


def test_historical_backfill_runs_once_and_has_explicit_refresh(tmp_path,monkeypatch):
    inp=tmp_path/"input";out=tmp_path/"results";summary=tmp_path/"longitudinal"
    inp.mkdir();out.mkdir()
    (inp/"2026-10-03-canonical.csv").write_text("date\\n")
    (inp/"2026-10-17-canonical.csv").write_text("date\\n")
    called=[]
    def fake_collect(day,path,target,source):
        called.append(day)
        (target/(day+"-canonical-picks.csv")).write_text("date\\n")
    monkeypatch.setattr(batch,"collect",fake_collect)
    monkeypatch.setattr(batch,"build",lambda *a:{"archive_dates":["2026-10-03","2026-10-17"]})
    cutoff=datetime.date(2026,11,3)
    batch.run(inp,out,summary,cutoff)
    assert called==["2026-10-03","2026-10-17"]
    called.clear();batch.run(inp,out,summary,cutoff)
    assert called==["2026-10-17"]
    called.clear();batch.run(inp,out,summary,cutoff,refresh_historical=True)
    assert called==["2026-10-03","2026-10-17"]


def test_cumulative_window_preserves_17_oct_legacy_definition():
    rows=[dict(date="2026-10-03",archive_group="CANONICAL",market="POINT",
               status="SETTLED",official_win="1",player="A",team="OTT"),
          dict(date="2026-10-17",archive_group="CANONICAL",market="POINT",
               status="SETTLED",official_win="0",player="B",team="OTT")]
    summary,_,_=longitudinal.report(rows)
    assert summary["windows"]["since_2026_10_03"]["CANONICAL"]["POINT"]["settled"]==2
    assert summary["windows"]["since_2026_10_17"]["CANONICAL"]["POINT"]["settled"]==1
    assert summary["windows"]["since_2026_10_17"]["CANONICAL"]["POINT"]["wins"]==0
