"""User's >=1.40 rule applies to bets, never to the probability ranking."""
import pandas as pd
import pytest
from conftest import load_script
from test_history import candidate

picks=load_script("07_build_daily_bets")

@pytest.mark.parametrize("price,expect", [
    (1.39,0), (1.40,1), (1.41,1), (None,0), (float("nan"),0), (1.01,0)
])
def test_probability_does_not_override_minimum_odds(price,expect):
    c=candidate().copy()
    c["run_date"]="2026-10-02"
    c["model_probability"]=.999
    c["odds_decimal"]=price
    c["value_gap"]=c["model_probability"]-c["implied_probability"]
    selected,stats=picks.build_daily_bets(c,"2026-10-02",10,1.4,.90,.02,True,True)
    assert len(selected)==expect
    assert stats["rows_removed_low_odds"]==1-expect

def test_lower_cli_minimum_cannot_undercut_user_hard_floor():
    c=candidate().copy();c["run_date"]="2026-10-02";c["model_probability"]=.99
    c["odds_decimal"]=1.35
    c["value_gap"]=c["model_probability"]-c["implied_probability"]
    result,_=picks.build_daily_bets(c,"2026-10-02",10,1.2,.90,.02,True,True)
    assert result.empty

def test_negative_ev_is_allowed_with_valid_price():
    c=candidate().copy();c["run_date"]="2026-10-02"
    c["odds_decimal"]=1.41;c["model_probability"]=.4;c["edge_probability"]=-.15
    c["value_gap"]=c["model_probability"]-c["implied_probability"]
    result,_=picks.build_daily_bets(c,"2026-10-02",10,1.4,.90,.02,True,True)
    assert len(result)==1
    assert result.iloc[0].is_value_bet==0
