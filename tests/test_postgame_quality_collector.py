"""Offline unit tests for postgame audit. No network and no model training."""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("postgame",ROOT/"scripts/postgame_pick_diagnostics.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_nhl_abbreviated_siblings_do_not_get_mixed():
    box={"awayTeam":{"abbrev":"NJD"},
         "playerByGameStats":{"awayTeam":{"forwards":[
             {"playerId":1001,"name":{"default":"J. Hughes"},"goals":0,"points":0},
             {"playerId":1002,"name":{"default":"L. Hughes"},"goals":1,"points":1}
         ]}}}
    jack,status=m.lookup_player(box,"NJD","Jack Hughes")
    luke,_=m.lookup_player(box,"NJD","Luke Hughes")
    assert (jack["playerId"],luke["playerId"])==(1001,1002)
    assert status=="INITIAL_SURNAME"
    assert m.lookup_player(box,"WSH","Jack Hughes")[0] is None

def test_nhl_tkachuk_initial_prevents_sibling_misattribution():
    box={"homeTeam":{"abbrev":"FLA"},"playerByGameStats":{
        "homeTeam":{"forwards":[
            {"playerId":1,"name":{"default":"M. Tkachuk"},"assists":3},
            {"playerId":2,"name":{"default":"B. Tkachuk"},"goals":2}]}}}
    matt,_=m.lookup_player(box,"FLA","Matthew Tkachuk")
    brady,_=m.lookup_player(box,"FLA","Brady Tkachuk")
    assert (matt["playerId"],brady["playerId"])==(1,2)

def test_ambiguous_identical_initial_names_are_not_guessed():
    box={"homeTeam":{"abbrev":"TST"},"playerByGameStats":{
         "homeTeam":{"forwards":[{"name":{"default":"A. Smith"}},
                                 {"name":{"default":"A. Smith"}}]}}}
    x,state=m.lookup_player(box,"TST","Adam Smith")
    assert x is None and state=="ABSENT_OR_NAME_MISMATCH"

def test_geometric_proxy_is_explicitly_not_expected_goals():
    pbp={"plays":[{"typeDescKey":"shot-on-goal","details":{"shootingPlayerId":10,"xCoord":85,"yCoord":3}},
                  {"typeDescKey":"blocked-shot","details":{"shootingPlayerId":10,"xCoord":30,"yCoord":30}},
                  {"typeDescKey":"goal","details":{"scoringPlayerId":9,"xCoord":88,"yCoord":1}}]}
    attempts=m.extract_attempts(pbp,10)
    assert len(attempts)==2
    assert attempts[0]["dist_ft_proxy"]<=20
    assert attempts[1]["dist_ft_proxy"]>40

def test_id_wins_over_name_and_never_falls_back_to_another_player():
    box={"homeTeam":{"abbrev":"OTT"},"playerByGameStats":{"homeTeam":{"forwards":[
        {"playerId":42,"name":{"default":"Different spelling"}}]}}}
    assert m.lookup_player(box,"OTT","Old name","42")[1]=="NHL_ID"
    assert m.lookup_player(box,"OTT","Different spelling","43")[0] is None

def test_shootout_attempts_are_not_goal_market_events():
    pbp={"plays":[{"typeDescKey":"goal","periodDescriptor":{"periodType":"SO"},
                   "details":{"scoringPlayerId":42}}]}
    assert m.extract_attempts(pbp,42)==[]
