import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "model" / "07_build_daily_bets.py"


def load_module():
    spec = importlib.util.spec_from_file_location("daily_bets", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def base_rows():
    return pd.DataFrame(
        [
            {
                "bet_id": "a",
                "run_date": "2026-09-23",
                "bet_status": "pending",
                "result": "",
                "actual_stat_value": None,
                "settled_at": "",
                "recommended_flag": False,
                "recommendation_rank": None,
                "date_match": "2026-09-23",
                "player_name": "Player A",
                "team": "TOR",
                "opponent": "MTL",
                "bookmaker": "Unibet",
                "market": "player_points",
                "stat": "points",
                "threshold": 1,
                "odds_decimal": 1.80,
                "implied_probability": 1 / 1.80,
                "model_probability": 0.70,
                "edge_probability": 0.70 - 1 / 1.80,
                "hard_exclude_hot_streak_pre": 0,
                "toi_last_game_minutes": 16.0,
            },
            {
                "bet_id": "b",
                "run_date": "2026-09-23",
                "bet_status": "pending",
                "result": "",
                "actual_stat_value": None,
                "settled_at": "",
                "recommended_flag": False,
                "recommendation_rank": None,
                "date_match": "2026-09-23",
                "player_name": "Player B",
                "team": "BOS",
                "opponent": "NYR",
                "bookmaker": "Unibet",
                "market": "player_points",
                "stat": "points",
                "threshold": 1,
                "odds_decimal": 1.20,
                "implied_probability": 1 / 1.20,
                "model_probability": 0.58,
                "edge_probability": 0.58 - 1 / 1.20,
                "hard_exclude_hot_streak_pre": 0,
                "toi_last_game_minutes": 13.0,
            },
            {
                "bet_id": "c",
                "run_date": "2026-09-23",
                "bet_status": "pending",
                "result": "",
                "actual_stat_value": None,
                "settled_at": "",
                "recommended_flag": False,
                "recommendation_rank": None,
                "date_match": "2026-09-23",
                "player_name": "Player C",
                "team": "EDM",
                "opponent": "VAN",
                "bookmaker": "Unibet",
                "market": "player_points",
                "stat": "points",
                "threshold": 1,
                "odds_decimal": 1.70,
                "implied_probability": 1 / 1.70,
                "model_probability": 0.88,
                "edge_probability": 0.88 - 1 / 1.70,
                "hard_exclude_hot_streak_pre": 1,
                "toi_last_game_minutes": 18.0,
            },
            {
                "bet_id": "d",
                "run_date": "2026-09-23",
                "bet_status": "pending",
                "result": "",
                "actual_stat_value": None,
                "settled_at": "",
                "recommended_flag": False,
                "recommendation_rank": None,
                "date_match": "2026-09-23",
                "player_name": "Player D",
                "team": "COL",
                "opponent": "DAL",
                "bookmaker": "Unibet",
                "market": "player_points",
                "stat": "points",
                "threshold": 1,
                "odds_decimal": 1.90,
                "implied_probability": 1 / 1.90,
                "model_probability": 0.91,
                "edge_probability": 0.91 - 1 / 1.90,
                "hard_exclude_hot_streak_pre": 1,
                "toi_last_game_minutes": 17.0,
            },
        ]
    )


def test_selection_applies_odds_probability_edge_and_hot_streak_rules():
    m = load_module()
    df = base_rows()
    selected, stats = m.build_daily_bets(
        df,
        run_date="2026-09-23",
        max_picks=10,
        min_odds=1.01,
        min_model_proba=0.50,
        min_edge=0.0,
        value_threshold=0.02,
        hot_streak_exception_proba=0.90,
        one_pick_per_player=True,
        disable_hot_streak_exclude=False,
    )
    assert selected["player_name"].tolist() == ["Player D", "Player A"]
    assert stats["rows_removed_hot_streak"] == 1


def test_selection_is_capped_at_ten_and_one_pick_per_player():
    m = load_module()
    df = base_rows()
    extra = df.iloc[[0]].copy()
    extra["bet_id"] = "duplicate-market"
    extra["market"] = "same_market"
    extra["odds_decimal"] = 2.00
    extra["implied_probability"] = 0.50
    extra["edge_probability"] = 0.20
    df = pd.concat([df, extra], ignore_index=True)

    selected, _ = m.build_daily_bets(
        df,
        run_date="2026-09-23",
        max_picks=10,
        min_odds=1.01,
        min_model_proba=0.50,
        min_edge=0.0,
        value_threshold=0.02,
        hot_streak_exception_proba=0.90,
        one_pick_per_player=True,
        disable_hot_streak_exclude=False,
    )
    assert len(selected) <= 10
    assert selected["player_name"].is_unique
