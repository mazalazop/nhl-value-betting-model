#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, numbers
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "outputs" / "07_daily_bets.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "api" / "daily_picks.json"
MODEL_VERSION = "henachel-2026-27-v1"
SCHEMA_VERSION = "1.0.0"

def finite(v: Any) -> Any:
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, numbers.Integral):
        return int(v)
    if isinstance(v, numbers.Real):
        value = float(v)
        return value if math.isfinite(value) else None
    return v
def col(df: pd.DataFrame, names: list[str]) -> str | None:
    return next((n for n in names if n in df.columns), None)

def num(row: pd.Series, names: list[str]) -> float | None:
    c = col(row.to_frame().T, names)
    if not c:
        return None
    v = pd.to_numeric(pd.Series([row[c]]), errors="coerce").iloc[0]
    return finite(float(v)) if pd.notna(v) else None

def textv(row: pd.Series, names: list[str], default: str = "") -> str:
    c = col(row.to_frame().T, names)
    if not c or pd.isna(row[c]):
        return default
    return str(row[c]).strip()

def datev(v: Any) -> str | None:
    d = pd.to_datetime(v, errors="coerce")
    return None if pd.isna(d) else d.strftime("%Y-%m-%d")

def build(df: pd.DataFrame, slate_date: str | None = None) -> dict[str, Any]:
    date_col = col(df, ["date_match", "match_date", "game_date"])
    prob_col = col(df, ["model_probability", "probability", "proba_model", "p_model"])
    edge_col = col(df, ["edge_probability", "edge", "edge_pct"])
    odds_col = col(df, ["odds_decimal", "odds", "cote"])
    if df.empty:
        return {
            "schema_version": SCHEMA_VERSION,
            "model_version": MODEL_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "timezone": "Europe/Paris",
            "slate_date": slate_date,
            "pick_count": 0,
            "points_pick_count": 0,
            "goals_pick_count": 0,
            "picks": [],
        }
    if not date_col or not prob_col or not odds_col:
        raise ValueError("07_daily_bets.csv ne contient pas les colonnes minimales date/probabilité/cote.")

    w = df.copy()
    w["_p"] = pd.to_numeric(w[prob_col], errors="coerce")
    w["_e"] = pd.to_numeric(w[edge_col], errors="coerce") if edge_col else 0.0
    w["_o"] = pd.to_numeric(w[odds_col], errors="coerce")
    w = w.sort_values(["_p", "_e", "_o"], ascending=False, kind="stable")
    if "pick_market_group" in w.columns:
        w["_market_rank"] = w.groupby("pick_market_group", sort=False).cumcount() + 1
    else:
        w["_market_rank"] = range(1, len(w) + 1)

    dates = pd.to_datetime(w[date_col], errors="coerce").dropna()
    picks = []
    for _, r in w.iterrows():
        player = textv(r, ["player_name", "nom", "player"])
        odds = num(r, ["odds_decimal", "odds", "cote"])
        prob = num(r, ["model_probability", "probability", "proba_model", "p_model"])
        if not player or odds is None or prob is None or odds <= 1:
            continue
        picks.append({
            "rank": int(r["_market_rank"]),
            "player": player,
            "player_id": (int(r[col(df, ["id_joueur", "player_id"])]) if pd.api.types.is_integer_dtype(type(r[col(df, ["id_joueur", "player_id"])])) else str(r[col(df, ["id_joueur", "player_id"])])) if col(df, ["id_joueur", "player_id"]) and not pd.isna(r[col(df, ["id_joueur", "player_id"])]) else None,
            "team": textv(r, ["team", "team_player_match"]),
            "opponent": textv(r, ["opponent", "adversaire_match"]),
            "match_date": datev(r[date_col]),
            "market": textv(r, ["market", "stat", "bet_type"], "point_1_plus"),
            "market_group": textv(r, ["pick_market_group"], "points"),
            "odds": odds,
            "model_probability": prob,
            "implied_probability": num(r, ["implied_probability", "proba_implicite"]),
            "edge": num(r, ["edge_probability", "edge", "edge_pct"]) or 0.0,
            "ev_per_unit": num(r, ["ev_per_unit", "ev"]),
            "recent_form": num(r, ["recent_form", "point_hit_rate_last_5", "point_hit_rate_last_10"]),
            "toi_last_game_minutes": num(r, ["toi_last_game_minutes", "toi_dernier_match", "temps_de_glace"]),
            "drought_alert": num(r, ["no_point_drought_alert_pre", "goal_drought_alert_pre"]),
            "drought_excess": num(r, ["no_point_streak_excess_pre", "goal_streak_excess_pre"])
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "model_version": MODEL_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "timezone": "Europe/Paris",
        "slate_date": slate_date or (dates.min().strftime("%Y-%m-%d") if not dates.empty else None),
        "pick_count": len(picks),
        "points_pick_count": sum(1 for p in picks if p.get("market_group") == "points"),
        "goals_pick_count": sum(1 for p in picks if p.get("market_group") == "goals"),
        "picks": picks,
    }

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    p.add_argument("--output-json", default=str(DEFAULT_OUTPUT))
    p.add_argument("--slate-date", default=None, help="Date métier France du slate, YYYY-MM-DD.")
    a = p.parse_args()
    src, dst = Path(a.input_csv), Path(a.output_json)
    if not src.exists():
        raise FileNotFoundError(f"Input introuvable: {src}")
    payload = build(pd.read_csv(src, low_memory=False), slate_date=a.slate_date)
    if payload["pick_count"] and not payload["slate_date"]:
        raise ValueError("Picks présents mais slate_date introuvable.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Payload API écrit: {dst}")
    print(f"Slate France: {payload['slate_date']}")
    print(f"Picks: {payload['pick_count']}")

if __name__ == "__main__":
    main()
