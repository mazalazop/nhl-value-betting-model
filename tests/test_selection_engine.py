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
                "no_point_drought_alert_pre": 0,
                "no_point_streak_excess_pre": 0.0,
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
        max_picks=5,
        min_odds=1.01,
        min_model_proba=0.50,
        min_edge=-1.0,
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


def test_drought_alert_can_break_a_small_probability_gap():
    m = load_module()
    df = base_rows().iloc[[0, 1]].copy()
    df.loc[df["player_name"] == "Player A", "model_probability"] = 0.69
    df.loc[df["player_name"] == "Player B", "model_probability"] = 0.70
    df.loc[df["player_name"] == "Player A", "edge_probability"] = -0.05
    df.loc[df["player_name"] == "Player B", "edge_probability"] = -0.05
    df.loc[df["player_name"] == "Player A", "no_point_drought_alert_pre"] = 1
    df.loc[df["player_name"] == "Player A", "no_point_streak_excess_pre"] = 1.5
    selected, _ = m.build_daily_bets(
        df, "2026-09-23", 5, 1.01, 0.50, -1.0, 0.02, 0.90, True, False
    )
    assert selected.iloc[0]["player_name"] == "Player A"


def test_selection_never_exceeds_five_for_a_single_market():
    m = load_module()
    df = pd.concat([base_rows()] * 4, ignore_index=True)
    df["bet_id"] = [f"b{i}" for i in range(len(df))]
    df["player_name"] = [f"Player {i}" for i in range(len(df))]
    selected, _ = m.build_daily_bets(
        df, "2026-09-23", 5, 1.01, 0.50, -1.0, 0.02, 0.90, True, False
    )
    assert len(selected) <= 5


def test_combined_selector_keeps_five_points_and_five_goals_independent():
    import importlib.util
    combined_path = ROOT / "model" / "07b_build_combined_daily_bets.py"
    spec = importlib.util.spec_from_file_location("combined_daily_bets", combined_path)
    combined = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(combined)

    points = pd.concat([base_rows()] * 3, ignore_index=True)
    points["bet_id"] = [f"p{i}" for i in range(len(points))]
    points["player_name"] = [f"Point Player {i}" for i in range(len(points))]
    points["run_date"] = "2026-09-23"
    points["hard_exclude_hot_streak_pre"] = 0

    goals = points.copy()
    goals["bet_id"] = [f"g{i}" for i in range(len(goals))]
    goals["player_name"] = [f"Goal Player {i}" for i in range(len(goals))]
    goals["market"] = "player_to_score_including_ot"
    goals["goal_drought_alert_pre"] = 0
    goals["goal_streak_excess_pre"] = 0.0
    goals["pick_market_group"] = "goals"

    point_path = ROOT / "outputs" / "_test_points.csv"
    goal_path = ROOT / "outputs" / "_test_goals.csv"
    point_path.parent.mkdir(parents=True, exist_ok=True)
    points.to_csv(point_path, index=False)
    goals.to_csv(goal_path, index=False)

    try:
        import sys
        old_argv = sys.argv
        sys.argv = [
            "07b_build_combined_daily_bets.py",
            "--run-date", "2026-09-23",
            "--point-csv", str(point_path),
            "--goal-csv", str(goal_path),
            "--max-points", "5",
            "--max-goals", "5",
        ]
        combined.main()
        result = pd.read_csv(ROOT / "outputs" / "07_daily_bets.csv")
        assert len(result[result["pick_market_group"] == "points"]) <= 5
        assert len(result[result["pick_market_group"] == "goals"]) <= 5
        assert result[result["pick_market_group"] == "points"]["recommendation_rank"].between(1, 5).all()
        assert result[result["pick_market_group"] == "goals"]["recommendation_rank"].between(1, 5).all()
    finally:
        sys.argv = old_argv
        point_path.unlink(missing_ok=True)
        goal_path.unlink(missing_ok=True)


def test_combined_selector_allows_same_player_across_markets_and_missing_goal_file(tmp_path):
    import importlib.util
    import sys

    combined_path = ROOT / "model" / "07b_build_combined_daily_bets.py"
    spec = importlib.util.spec_from_file_location("combined_daily_bets_missing_goal", combined_path)
    combined = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(combined)

    points = base_rows().iloc[[0, 3]].copy()
    points["player_name"] = ["Same Player", "Point Player"]
    points["bet_id"] = ["p_same", "p_other"]
    points["hard_exclude_hot_streak_pre"] = 0
    point_path = tmp_path / "points.csv"
    points.to_csv(point_path, index=False)
    missing_goal_path = tmp_path / "missing_goals.csv"

    old_argv = sys.argv
    output_path = ROOT / "outputs" / "07_daily_bets.csv"
    try:
        sys.argv = [
            "07b_build_combined_daily_bets.py",
            "--run-date", "2026-09-23",
            "--point-csv", str(point_path),
            "--goal-csv", str(missing_goal_path),
            "--max-points", "5",
            "--max-goals", "5",
        ]
        combined.main()
        result = pd.read_csv(output_path)
        assert len(result) == 2
        assert set(result["pick_market_group"]) == {"points"}
    finally:
        sys.argv = old_argv


def test_combined_selector_allows_same_player_in_points_and_goals(tmp_path):
    import importlib.util
    import sys

    combined_path = ROOT / "model" / "07b_build_combined_daily_bets.py"
    spec = importlib.util.spec_from_file_location("combined_daily_bets_cross_market", combined_path)
    combined = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(combined)

    points = base_rows().iloc[[0]].copy()
    points["player_name"] = "Same Player"
    points["bet_id"] = "p_same"
    points["hard_exclude_hot_streak_pre"] = 0

    goals = points.copy()
    goals["bet_id"] = "g_same"
    goals["market"] = "player_to_score_including_ot"
    goals["goal_drought_alert_pre"] = 0
    goals["goal_streak_excess_pre"] = 0.0

    point_path = tmp_path / "points.csv"
    goal_path = tmp_path / "goals.csv"
    points.to_csv(point_path, index=False)
    goals.to_csv(goal_path, index=False)

    old_argv = sys.argv
    try:
        sys.argv = [
            "07b_build_combined_daily_bets.py",
            "--run-date", "2026-09-23",
            "--point-csv", str(point_path),
            "--goal-csv", str(goal_path),
            "--max-points", "5",
            "--max-goals", "5",
        ]
        combined.main()
        result = pd.read_csv(ROOT / "outputs" / "07_daily_bets.csv")
        assert len(result) == 2
        assert set(result["pick_market_group"]) == {"points", "goals"}
        assert result["player_name"].tolist().count("Same Player") == 2
    finally:
        sys.argv = old_argv
