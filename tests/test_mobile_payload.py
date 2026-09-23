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
    assert validation_mod.validate(payload) == []
