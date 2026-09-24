import importlib.util
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "model" / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

payload_mod = load("payload", "12_build_api_payload.py")
validation_mod = load("validation", "13_validate_daily_payload.py")

def test_payload_contract_and_sorting():
    df = pd.DataFrame([
        {"player_name":"B","team":"BBB","opponent":"AAA","date_match":"2026-10-14","market":"point_1_plus","odds_decimal":2.0,"model_probability":0.61,"edge_probability":0.11},
        {"player_name":"A","team":"AAA","opponent":"BBB","date_match":"2026-10-14","market":"point_1_plus","odds_decimal":1.8,"model_probability":0.72,"edge_probability":0.08},
    ])
    payload = payload_mod.build(df, slate_date="2026-10-13")
    assert payload["slate_date"] == "2026-10-13"
    assert payload["pick_count"] == 2
    assert payload["picks"][0]["player"] == "A"
    assert payload["picks"][0]["rank"] == 1
    assert payload["points_pick_count"] == 2
    assert payload["goals_pick_count"] == 0
    assert validation_mod.validate(payload) == []

def test_payload_ranks_independently_by_market_and_allows_same_player():
    df = pd.DataFrame([
        {"player_name":"Same Player","team":"AAA","opponent":"BBB","date_match":"2026-10-14","market":"point_1_plus","pick_market_group":"points","odds_decimal":2.0,"model_probability":0.80,"edge_probability":0.30},
        {"player_name":"Same Player","team":"AAA","opponent":"BBB","date_match":"2026-10-14","market":"player_to_score_including_ot","pick_market_group":"goals","odds_decimal":2.5,"model_probability":0.70,"edge_probability":0.30},
        {"player_name":"Another Goal","team":"BBB","opponent":"AAA","date_match":"2026-10-14","market":"player_to_score_including_ot","pick_market_group":"goals","odds_decimal":2.2,"model_probability":0.65,"edge_probability":0.20},
    ])
    payload = payload_mod.build(df, slate_date="2026-10-13")
    points = [p for p in payload["picks"] if p["market_group"] == "points"]
    goals = [p for p in payload["picks"] if p["market_group"] == "goals"]
    assert [p["rank"] for p in points] == [1]
    assert [p["rank"] for p in goals] == [1, 2]
    assert {p["player"] for p in points} & {p["player"] for p in goals} == {"Same Player"}
    assert validation_mod.validate(payload) == []
