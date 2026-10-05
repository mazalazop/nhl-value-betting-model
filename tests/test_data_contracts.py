import numpy as np
import pandas as pd
import pytest
from conftest import load_script
from fixtures import games_fixture


def test_scores_are_not_finality():
    m=load_script('00b_build_base_match_fusionnee')
    df=games_fixture().drop_duplicates('id_match').head(5).copy()
    df['status']=['OFF','FINAL','LIVE','FUT','UNKNOWN']
    assert len(m.played_matches_only(df)) == 2


def test_missing_stat_is_not_zero():
    m=load_script('00b_build_base_match_fusionnee')
    assert m.get_stat_int({},'points') is None
    assert m.get_stat_int({'points':0},'points') == 0


@pytest.mark.parametrize('problem', ['duplicate','points','date','team','missing_target'])
def test_bad_training_rows_rejected(problem):
    from henachel.data import validate_player_games
    df=games_fixture().head(4).copy()
    if problem=='duplicate': df=pd.concat([df,df.iloc[:1]],ignore_index=True)
    if problem=='points': df.loc[0,'points']=99
    if problem=='date': df.loc[0,'date_match']=pd.NaT
    if problem=='team': df.loc[0,'team_player_match']='BOS'
    if problem=='missing_target': df.loc[0,'points']=np.nan
    with pytest.raises(ValueError): validate_player_games(df)


def test_partial_non_target_data_allowed():
    from henachel.data import validate_player_games
    df=games_fixture()
    validate_player_games(df)


def test_status_classification():
    from henachel.data import game_state
    assert game_state('OFF') == 'final'
    assert game_state('LIVE') == 'live'
    assert game_state('FUT','PPD') == 'postponed'
    assert game_state('CANCELLED') == 'cancelled'
    assert game_state(None) == 'unknown'
